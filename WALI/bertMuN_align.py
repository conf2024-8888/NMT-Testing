import nltk
from tqdm import tqdm
import numpy as np
import string
import math
import torch
import sys
import torch.nn.functional as F
# from stanfordcorenlp import StanfordCoreNLP
import os
import time
import random
from copy import deepcopy
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM
# from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer


os.environ["CUDA_VISIBLE_DEVICES"]="0"

K_Number = 100
Max_Mutants = 5

ft = time.time()
#tokenizer = TreebankWordTokenizer()
#detokenizer = TreebankWordDetokenizer()

#nlp = StanfordCoreNLP("stanford-corenlp-full-2018-02-27", port=34139, lang="en")

def check_tree (ori_tag, line):
    tag = line.strip()
    tag = nlp.pos_tag(tag)
    #print (tag)
    #print (ori_tag)
    #print ("-----------------")
    if len(tag) != len(ori_tag):
        return False
    for i in range(len(tag)):
        if tag[i][1] != ori_tag[i][1]:
            return False
    return True


def bertInit():
    #config = Ber
    berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")#'/data/szy/bertlarge')
    bertori = BertModel.from_pretrained("bert-large-cased")#'/data/szy/bertlarge')
    #berttokenizer = RobertaTokenizer.from_pretrained('bert-large-uncased')
    #bertmodel = RoBertaForMaskedLM.from_pretrained('/data/szy/bertlarge')
    bertmodel.eval().cuda().to(torch.device("cuda:0"))
    bertori.eval().cuda().to(torch.device("cuda:0"))
    
    return bertmodel, berttokenizer, bertori

# tokenizer = TreebankWordTokenizer()

lcache = []


def BertM(bert, berttoken, inpori, bertori, mutate_idx):
    sentence = inpori

    tokens = berttoken.tokenize(sentence)
    batchsize = 1000 // len(tokens)
    #print(tokens[mutate_idx])

    gen = []
    ltokens = ["[CLS]"] + tokens + ["[SEP]"]

    try:
        encoding = [berttoken.convert_tokens_to_ids(ltokens[0:mutate_idx] + ["[MASK]"] + ltokens[mutate_idx + 1:])]#.cuda()
    except:
        return " ".join(tokens), gen
    p = []
    #print("encoding :", len(encoding))
    for i in range(0, len(encoding)):
        tensor = torch.tensor(encoding[i: min(len(encoding), i + batchsize)]).cuda()
        #print("tensor size", tensor.size())
        pre = F.softmax(bert(tensor)[0], dim=-1).data.cpu() # logits of the bertmodel, (1,42,28996)
        p.append(pre)

    pre = torch.cat(p, 0) # concate on 0 dimension 
    #print("pre size", pre.size()) #(40, 42, 28996), (token, layer, dimension in bert)

    tarl = [[tokens, -1]]


    # delete for loop 
    topk = torch.topk(pre[0][mutate_idx +1], K_Number)#.tolist()
    
    value = topk[0].numpy() #top values
    topk = topk[1].numpy().tolist() # index of the top values
    
    #print (topk)
    topkTokens = berttoken.convert_ids_to_tokens(topk)
    # tarl = []
    #print(topkTokens)
    for index in range(len(topkTokens)):
        #print(topkTokens[index], value[index])
        if value[index] < 0.005:
            break
        tt = topkTokens[index]
        #print (tt)
        if tt in string.punctuation or "#" in tt :
            continue
        if tt.strip() == tokens[mutate_idx].strip():
            continue
        # pos tag
        # if nltk.pos_tag([tt])[0][1][:1] != nltk.pos_tag([tokens[mutate_idx]])[0][1][:1]:
        #     continue
        l = deepcopy(tokens)
        l[mutate_idx] = tt
        tarl.append([l, mutate_idx, value[index]])
    #print("tarl:", tarl)
        
    if len(tarl) == 0:
        return " ".join(tokens), gen
    
    lDB = []
 
    for i in range(0, len(tarl), batchsize):
        lDB.append(bertori(torch.tensor([berttoken.convert_tokens_to_ids(["[CLS]"] + l[0] + ["[SEP]"]) for l in tarl[i: min(i + batchsize, len(tarl))]]).cuda())[0].data.cpu().numpy())
    lDB = np.concatenate(lDB, axis=0)
            

    lDA = lDB[0]
    assert len(lDB) == len(tarl)
    tarl = tarl[1:]
    lDB = lDB[1:]
    for t in range(len(lDB)):
        DB = lDB[t][tarl[t][1]]
        DA = lDA[tarl[t][1]]

        cossim = np.sum(DA * DB) / (np.sqrt(np.sum(DA * DA)) * np.sqrt(np.sum(DB * DB)))

        if cossim < 0.85:
            continue

        #sen = " ".join(tarl[t][0])# + "\t!@#$%^& " + str(math.exp(value[index]))#.replace(" ##", "")
        sen = " ".join(tarl[t][0]).replace(" ##", "").replace(" - ", "-").replace(" ' ", "'").replace(" .", ".").replace(" , ", ", ")
          
        gen.append([cossim, sen])
    # if len(lcache) > 4:
    #     lcache = lcache[1:]    

    # lcache.append([inpori, " ".join(tokens), gen])
    gen.sort(key=lambda x: x[0], reverse=True)
    return " ".join(tokens).replace(" ##", "").replace(" - ", "-").replace(" ' ", "'").replace(" .", ".").replace(" , ", ", "), gen#.replace(" ##", "")#, gen



# import pandas as pd
# replaced_csv = pd.DataFrame()
INDEX = 0
def align_index(bertmodel, berttoken, line, bertori, token_index, top_tokens, logits):
    result = []
    history_index = []
    tokens = berttoken.tokenize(line)
    # print(tokens)
    tar=""
    replaced = []
    replaced_tok = []
    pos_tag = []
    logit_lst = []

    for i in range(len(token_index)):
        idx = max(token_index[i]-1, 0)
        idx = min(idx, len(tokens)-1)
        if tokens[idx].lower() == top_tokens[i].lower():
            if idx in history_index:
                continue
            history_index.append(idx)
            tar, gen = BertM(bertmodel, berttoken, line, bertori, idx)
            # print(idx)
            # print(tokens[idx])
            # print(gen)
            if len(gen) > 0 and gen[0] not in result:
                result+=gen
                replaced+=[idx]*len(gen)
                replaced_tok+=[tokens[idx].replace("##", "")]*len(gen)
                pos_tag+=[nltk.pos_tag([tokens[idx].replace("##", "")])[0][1]]*len(gen)
                logit_lst+=[logits[i]]*len(gen)
                continue
        
        else:
            idx = min(token_index[i]+5, len(tokens)-1)
            while idx>max(token_index[i]-5, 0):
                if top_tokens[i].lower() in tokens[idx].lower():
                    if idx in history_index:
                        idx -=1
                        continue
                    history_index.append(idx)
                    tar, gen = BertM(bertmodel, berttoken, line, bertori, idx)
                    # print(tokens[idx])
                    # print(gen)
                    if len(gen) > 0 and gen[0] not in result:
                        result+=gen
                        replaced+=[idx]*len(gen)
                        replaced_tok+=[tokens[idx].replace("##", "")]*len(gen)
                        pos_tag+=[nltk.pos_tag([tokens[idx].replace("##", "")])[0][1]]*len(gen)
                        logit_lst+=[logits[i]]*len(gen)
                    break
                idx -=1
        if len(result) > 5:
            result=result[:5]
            replaced=replaced[:5]
            replaced_tok=replaced_tok[:5]
            pos_tag=pos_tag[:5]
            logit_lst=logit_lst[:5]
            break
    global INDEX
   
    added_csv = pd.DataFrame({"index":[INDEX]*len(result),"test_sen": [i[1] for i in result], "simi": [i[0] for i in result], "token_idx": replaced, "tokens":replaced_tok, "pos": pos_tag, "logits": logit_lst})
    INDEX +=1
    return tar, result, added_csv
    
# f = open(sys.argv[2], "w")
# fline = open(sys.argv[3], "w")


import json
# with open(sys.argv[1]) as f:
with open("./en_token_al_stop.json") as f:
    sent_data = json.load(f)
f = open("./f_en_mu_align_stop.txt", "w")
fline = open("./f_en_mu_align_stop.index", "w")
bertmodel, berttoken, bertori = bertInit()

for i in tqdm(range(len(sent_data))):
    line = sent_data[i]["en"]
    index= sent_data[i]["token_index"]
    top_tokens = sent_data[i]["top_tokens"]
    logits = sent_data[i]["logits"]
    #tag = nlp.pos_tag(line)

    tar, gen,added_csv = align_index(bertmodel, berttoken, line, bertori, index, top_tokens, logits)
    #gen = sorted(gen)[::-1]
    count = 0
    for sen in gen:
        f.write(tar.strip() + "\n")
        #print(sen[1].strip())
        f.write(sen[1].strip() + "\n")
        fline.write(str(i) + "\n")
        # count += 1
        # if count >= Max_Mutants:
        #     break
#     replaced_csv = pd.concat([replaced_csv, added_csv])
# replaced_csv.to_csv(f"compare_replace.csv", index=False)
# print(f".........compare_replace.csv generated!" )
f.close()
fline.close()
print (time.time() - ft)
