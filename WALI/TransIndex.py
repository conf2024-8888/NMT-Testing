import sys
from datasets import load_dataset, load_metric
import transformers
import os
from tqdm import tqdm

from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline, MarianTokenizer, MarianTokenizer, TFMarianMTModel, AutoModelForSeq2SeqLM
import torch
mode_name = '../transformer'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model =AutoModelForSeq2SeqLM.from_pretrained(mode_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(mode_name, return_tensors="pt")
from datasets import load_dataset, load_metric
raw_datasets = load_dataset("wmt17", "zh-en")
metric = load_metric("sacrebleu")

from datasets.arrow_dataset import Dataset
prefix = ""
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "zh"
def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["translation"] = examples["translation"]
    return model_inputs

from datasets import load_dataset, load_metric
def preprocess_dataset(num_ex):
    raw_datasets = load_dataset("wmt17", "zh-en")   
    preprocess = preprocess_function(raw_datasets['test'][num_ex:num_ex+1])
    return Dataset.from_dict(preprocess)
    

import re
from nltk.corpus import stopwords
import string 
import nltk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

def plot_alignment(en_sen, ch_sen, attention_matrix, value_matrix, logits):
    font_path = ''
    fontP = fm.FontProperties(fname=font_path)

    fig = plt.figure(figsize=(15,25))

    ax = fig.add_subplot(111)

    cax = ax.matshow(attention_matrix, cmap='Blues')

    ax.tick_params(labelsize=12)
    ax.set_xticklabels(['']+[t.lower() for t in en_sen], 
                        rotation=90)
    ax.set_yticklabels(['']+ch_sen, fontproperties=fontP)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(True)
    
    atten_values = []
    
    for i in range(len(value_matrix)):
        for j in range(len(value_matrix[0])):
            c = round(value_matrix[i,j].item(), 3)
            l = round(logits[i]*100, 3)
            if attention_matrix[i,j].item() == 1:
                atten_values.append(value_matrix[i,j])
                if value_matrix[i,j].item() < 0.2:
                    ax.text(j, i, str(c)+"/"+str(l), va='center', ha='center', color='red')
                else:
                    ax.text(j, i, str(c)+"/"+str(l), va='center', ha='center', color='green')
    plt.show()
    plt.close(fig)
    return atten_values

nltk.download('stopwords')
sw_nltk = stopwords.words('english')
def is_stopword(token_id, tokenizer):
    word = tokenizer.decode(token_id)
    if word in string.punctuation:
        return True
    word = word.lower()
    word = re.sub(r'[^\w\s]', '', word)
    stopword = stopwords.words('english') + ["</s>", "<unk>", ">>cmn_Hans<<", "<pad>"]
    if word in stopword or not word:
        return True
    return False
def remove_dup(my_list):
    unique_list = []

    for element in my_list:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list

def alignment(tensor_matrix):
    # get indices of maximum value in each row
    max_indices = torch.argmax(tensor_matrix, dim=1)

    # create output matrix with zeros everywhere
    output_matrix = torch.zeros_like(tensor_matrix)

    # set value of 1 at maximum index of each row
    for i in range(output_matrix.shape[0]):
        output_matrix[ i, max_indices[i] ] = 1
    return output_matrix

def alignment_search_nostop(en_sentence, zh_sentence, k=10):
    token_ids = tokenizer(en_sentence, return_tensors="pt").input_ids.to(device)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(zh_sentence, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outs = model(input_ids=token_ids,decoder_input_ids=labels, output_hidden_states=True, output_attentions=True)
    
    attention = torch.stack(outs.cross_attentions, dim=0).squeeze(1)
    attention = attention[:2].mean(dim=0)[:, :-1,:-1].mean(dim=0)
    align_matrix = alignment(attention)
    sentence = tokenizer.convert_ids_to_tokens(token_ids[0])
    translation = tokenizer.convert_ids_to_tokens(labels[0])
    
    #plot_alignment(sentence, translation, align_matrix, attention)
    logits = []
    for i in range(len(labels[0])):
        logits.append(outs.logits.softmax(axis=-1)[0][i][labels[0][i]].item())
    out_idx_lst = np.argsort(logits[:-1]) #Remove last token
    

    in_idx_lst = []
    for indx in out_idx_lst:
        #if len(np.where(align_matrix[indx] == 1)[0])>0:
        id = np.where(align_matrix[indx] == 1)[0].item()
        if id not in in_idx_lst and not is_stopword(token_ids[0][id], tokenizer):
            in_idx_lst += np.where(align_matrix[indx] == 1)[0].tolist() 
    #print(in_idx_lst)
    #idx =[ i for i in in_idx_lst if not is_stopword(token_ids[0][i], tokenizer)]
    #idx = remove_dup(idx)
    
    idx = in_idx_lst[:min(len(idx), k)]
    input_tokens = [token_ids[0][i] for i in idx ]
    return idx, [tokenizer.decode(t) for t in input_tokens if tokenizer.decode(t) not in string.punctuation]
def punctuation(token):
    return not all(char in string.punctuation for char in token) and token not in ["</s>", "<unk>", ">>cmn_Hans<<", "<pad>", "."] and len(token)>1
def alignment_search(en_sentence, zh_sentence, k=5):
    token_ids = tokenizer(en_sentence, return_tensors="pt").input_ids.to(device)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(zh_sentence, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outs = model(input_ids=token_ids,decoder_input_ids=labels, output_hidden_states=True, output_attentions=True)
    
    attention = torch.stack(outs.cross_attentions, dim=0).squeeze(1).cpu()
    attention = attention[:2].mean(dim=0)[:, :-1,:-1].mean(dim=0)
    
    align_matrix = alignment(attention)
    
    sentence = tokenizer.convert_ids_to_tokens(token_ids[0])
    translation = tokenizer.convert_ids_to_tokens(labels[0])
    
    logits = []
    for i in range(len(labels[0])):
        logits.append(outs.logits.softmax(dim=-1)[0][i][labels[0][i]].item())
    
    #atten_values = plot_alignment(sentence, translation, align_matrix, attention, logits)
    
    
    out_idx_lst = np.argsort(logits[:-1]) #Remove last token
    
    in_idx_lst = []
    logt_idx = []
    for indx in out_idx_lst:
        #if len(np.where(align_matrix[indx] == 1)[0])>0:
        id = np.where(align_matrix[indx] == 1)[0].item()
        tok = tokenizer.decode(token_ids[0][id])
        #print(tokenizer.decode(token_ids[0][id]))
        if id not in in_idx_lst and punctuation(tok):
            in_idx_lst.append(id) 
            logt_idx.append(indx)
    #print(in_idx_lst)
    # idx =[ i for i in in_idx_lst if tokenizer.decode(token_ids[0][i]) not in list(string.punctuation) + ["</s>", "<unk>", ">>cmn_Hans<<", "<pad>", "."]]
    
    #idx = remove_dup(idx)
    idx = in_idx_lst[:min(len(in_idx_lst), k)]
    logt_idx = logt_idx[:min(len(in_idx_lst), k)]
    input_tokens = [token_ids[0][i] for i in idx ]
    logit_lst = [logits[i] for i in logt_idx]
    return idx, [tokenizer.decode(t) for t in input_tokens], logit_lst

def loop_data_nostop(dataset):
    for pair in tqdm(dataset['translation']):
        #pair["token_index"], pair["top_tokens"] = gradient_search_nostop(pair["en"], pair["zh"])
        pair["token_index"], pair["top_tokens"], pair["logits"] = alignment_search(pair["en"], pair["zh"])
        #print(dataset['translation'][0])
    return dataset['translation']
result = loop_data_nostop(raw_datasets["test"][:])

import json

with open("./en_token_al_stop.json", "w") as f:
    json.dump(result, f)
    
    