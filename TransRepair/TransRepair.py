import subprocess
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import nltk
nltk.download('averaged_perceptron_tagger')
import numpy as np
# from google.cloud import translate
from google.cloud import translate_v2 as translate
import jieba
import os, requests, uuid, json
from tqdm import tqdm
import pandas as pd
# initialization
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()
dataset = 'wmt17'
software = 'transformer'
similarity_threshold = 0.96
#translate_client = translate.Client()	# initialize the Google translate client
apikey = ''	
approach = ''
mutant_limit=5

# result_stats_filename = f"result_stats_{dataset}_{approach}_{software}.csv"
# result_stats = pd.DataFrame()

# distance_result_filename=f"issues_{dataset}_{approach}_{software}.csv"
# distance_result = pd.DataFrame()

# load the similar word dictionary
SIM_DICT_FILE = "word_op.txt"
if software == "transformer" and approach != "":
    with open(f"./en_token_{approach}_{dataset}.json", "r") as f:
        sent_data = json.load(f)

#Bing translation
def BingTranslate(api_key, text, language_from, language_to):
    # If you encounter any issues with the base_url or path, make sure
    # that you are using the latest endpoint: https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-translate
    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&language='+ language_from +'&to=' + language_to
    constructed_url = base_url + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Ocp-Apim-Subscription-Region': 'eastus',
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    if type(text) is str:
        text = [text]

    body = [{'text': x} for x in text]
    # You can pass more than one object in body.
    
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()


    return [i["translations"][0]["text"] for i in response]


def getLevenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def normalizedED(seq1, seq2):

    dist = getLevenshtein(seq1, seq2)
    normalized_dist = 1 - (dist/max(len(seq1), len(seq2)))

    return normalized_dist


def getSubSentSimilarity(subsentL1, subsentL2):
    similarity = -1

    for subsent1 in subsentL1:
        for subsent2 in subsentL2:
            currentSim = normalizedED(subsent1.split(' '), subsent2.split(' '))
            if currentSim>similarity:
                similarity = currentSim

    return similarity

def lcs(X, Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n + 1) for i in range(m + 1)] 
  
    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 

def lcs_sim_d(x, y):
    assert len(x) == len(y)
    ma = -1000
    #print (x)
    assert len(x) > 0
    for i in range(len(x)):
        lcs_vari = float(lcs(x[i], y[i]))
        try:
            lcs_score = (lcs_vari) / max(len(x[i]), len(y[i]))
        except:
            lcs_score = 1
        ma = max(ma, lcs_score)

    return ma
import collections

f = open("idf_dic.txt", "r")
line = f.readline()
f.close()

idf = eval(line)
def compute_tfidf(line):
    words = line.strip().split()
    counter = collections.Counter()
    for word in words:
        #if word in [',', '.', ':', '"', "'", '', ' ', '', '。', '：', '，', '）', '（', '！', '?', '”', '“', "’", "‘", "；"]:
        #    continue
        counter[word] += 1

    ret_dic = {}
    for item in counter:
        search_index = item
        if item not in idf:
            item = "<unk>"

        ret_dic[search_index] = (counter[search_index] / len(words)) * idf[item]

    return ret_dic
import math
def compute_cos(dic_a, dic_b):
    dot = 0
    l_a = 0
    l_b = 0
    for item in dic_a:
        l_a += dic_a[item] * dic_a[item]

    for item in dic_b:
        l_b += dic_b[item] * dic_b[item]
        if item in dic_a:
            dot += dic_a[item] * dic_b[item]

    #print (dic_a)
    #print (dic_b)
    #print (dot / math.exp(((math.log(l_a) / 2) + (math.log(l_b) / 2))))
    
    try:
        assert dot / math.exp(((math.log(l_a) / 2) + (math.log(l_b) / 2))) <= 1
    except:
        return 1
    return dot / math.exp(((math.log(l_a) / 2) + (math.log(l_b) / 2)))#dot / (math.sqrt(l_a) * math.sqrt(l_b))


def tf_cos_d_sim(x, y):
    ma = -100000
    for i in range(len(x)):
        x_now = x[i]
        y_now = y[i]
        dic_a = compute_tfidf(" ".join(x_now))
        dic_b = compute_tfidf(" ".join(y_now))
        ma = max(ma, compute_cos(dic_a, dic_b))
    return ma

def _get_ngrams(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams up to max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_with_padding = collections.Counter()
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      #print (segment[i: i + order])
      if "<padding>" in segment[i:i + order]:
        #print ("1")
        ngram_with_padding[ngram] += 1
      else:
        ngram_counts[ngram] += 1

  if len(ngram_with_padding) > 0:
      return ngram_counts, ngram_with_padding
  return ngram_counts, ngram_with_padding

def overlap_with_padding(ngram, collect, count_pad):
    _sum = 0
    p_ngram = []
    for f_index in range(count_pad + len(collect)):
        s_index = f_index - count_pad + 1
        l = []
        from copy import deepcopy
        n_ngram = []
        for t in ngram:
            n_ngram.append(deepcopy(t))
        #n_ngram = deepcopy(ngram)

        for index in range(max(0, s_index), min(f_index + 1, len(ngram))):
            n_ngram[index] = "<padding>"

        if "<padding>" not in n_ngram:
            continue
        n_ngram = tuple(n_ngram)
        #print (n_ngram)
        if n_ngram not in p_ngram:
            p_ngram.append(n_ngram)
            _sum += collect[n_ngram]
    #print (_sum)
    return _sum

def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.
  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    count_pad = 0
    for i in references:
        if i == "<padding>":
            count_pad += 1
    
    translation_length += len(translations)
    #print (references)
    #print (max_order)
    ref_ngram_counts, ngram_with_padding = _get_ngrams(references, max_order)
    translation_ngram_counts, _ = _get_ngrams(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram, count in translation_ngram_counts.items():
        if ngram in overlap:
            overlap[ngram] += overlap_with_padding(ngram, ngram_with_padding, count_pad)
        else:
            overlap[ngram] = overlap_with_padding(ngram, ngram_with_padding, count_pad)

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
      possible_matches_by_order[len(ngram) - 1] += overlap_with_padding(ngram, ngram_with_padding, count_pad)
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
    
    #for ngram in ngram_with_padding:
    #    possible_matches_by_order[len(ngram) - 1] += ngram_with_padding[ngram]
  
  precisions = [0] * max_order
  smooth = 1.0
  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum/max_order)

  if use_bp:
    if not reference_length:
      bp = 1.0
    else:
      ratio = translation_length / reference_length
      if ratio <= 0.0:
        bp = 0.0
      elif ratio >= 1.0:
        bp = 1.0
      else:
        bp = math.exp(1 - 1. / ratio)
  bleu = geo_mean * bp
  return np.float32(bleu)

import re
import sys
import six
import unicodedata
class UnicodeRegex(object):
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars("P")
    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

  def property_chars(self, prefix):
    return "".join(six.unichr(x) for x in range(sys.maxunicode)
                   if unicodedata.category(six.unichr(x)).startswith(prefix))

uregex = UnicodeRegex()

def bleu_tokenize(string):
  r"""Tokenize a string following the official BLEU implementation.
  See https://github.com/moses-smt/mosesdecoder/"
           "blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).
  Note that a number (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized,
  i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
  does not match this case (unless we add a space after each sentence).
  However, this error is already in the original mteval-v14.pl
  and we want to be consistent with it.
  Args:
    string: the input string
  Returns:
    a list of tokens
  """
  string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
  string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
  string = uregex.symbol_re.sub(r" \1 ", string)
  return string.split()


def com_bleu_2 (old, new):
    en_lines = []
    or_lines = []
    en_new_line = bleu_tokenize(new)
    en_line = bleu_tokenize(old)
    or_lines.append(en_line)
    en_lines.append(en_new_line)
    score = compute_bleu(or_lines, en_lines)#nltk.tran
    #l.append([score, score, score, en_old[i], en[i], en_new[i], new[i], ora[i]])
    return score#sum_ / len(en)

def bleu_sim_d(x, y):
    assert len(x) == len(y)
    ma = -1000
    #print (x)
    assert len(x) > 0
    for i in range(len(x)):
        lcs_vari = float(com_bleu_2(" ".join(x[i]), " ".join(y[i])))
        lcs_score = (lcs_vari) #/ max(len(x[i]), len(y[i]))
        ma = max(ma, lcs_score)
        lcs_vari = float(com_bleu_2(" ".join(y[i]), " ".join(x[i])))
        lcs_score = (lcs_vari) #/ max(len(x[i]), len(y[i]))
        ma = max(ma, lcs_score)

    return ma
def Edit_Distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
def ed_sim_d(x, y):
    max_ed = -1e18
    for i in range(len(x)):
        ed = Edit_Distance(x[i], y[i])
        try:
            ed = 1. - float(ed) / max(len(x[i]), len(y[i]))
        except:
            ed = 1
        max_ed = max(ed, max_ed)
    #return 1. - float(min_ed) / max(len(x[i]), len(y[i]))

    # if min_ed == 1e18:
    #     min_ed = 0
    
    #return 1. - float(min_ed) / max(len(x[i]), len(y[i]))
    return max_ed

def wordDiffSet(sentence1, sentence2):

    file1 = "temptest5.txt"
    file2 = "temptest6.txt"

    set1 = set()
    set2 = set()

    with open(file1, 'w') as f:
        f.write(sentence1)

    with open(file2, 'w') as f:
        f.write(sentence2)

    p = subprocess.run(["wdiff", file1, file2], stdout= subprocess.PIPE)
    wdstr = p.stdout.decode("utf-8")

    #print (wdstr)


    idxL1 = []
    idxL2 = []

    startIdx = -1
    endIdx = -1
    for idx, c in enumerate(wdstr):
        if c=='[':
            startIdx = idx
        elif c==']':
            endIdx = idx
            idxL1.append((startIdx, endIdx))
        elif c=='{':
            startIdx = idx
        elif c=='}':
            endIdx = idx
            idxL2.append((startIdx, endIdx))


    for idxPair in idxL1:
        wordsS = wdstr[idxPair[0]+2:idxPair[1]-1]
        wordsL = wordsS.split(' ')
        set1 |= set(wordsL)

    for idxPair in idxL2:
        wordsS = wdstr[idxPair[0]+2:idxPair[1]-1]
        wordsL = wordsS.split(' ')
        set2 |= set(wordsL)

    return (set1, set2)


def getSubSentenceList(sentence1, sentence2, set1, set2):
    # obtain the diff words
    (set1, set2) = wordDiffSet(sentence1, sentence2)

    # generate sub sentences
    subsentL1 = []
    subsentL2 = []

    removeIdx1 = []
    removeIdx2 = []

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    sentence1L = tokenizer.tokenize(sentence1)
    sentence2L = tokenizer.tokenize(sentence2)

    for idx, word in enumerate(sentence1L):
        if word in set1:
            removeIdx1.append(idx)

    for idx, word in enumerate(sentence2L):
        if word in set2:
            removeIdx2.append(idx)

    for idx in removeIdx1:
        tokens = tokenizer.tokenize(sentence1)
        tokens.pop(idx)
        subsent = detokenizer.detokenize(tokens)
        subsentL1.append(subsent)

    for idx in removeIdx2:
        tokens = tokenizer.tokenize(sentence2)
        tokens.pop(idx)
        subsent = detokenizer.detokenize(tokens)
        subsentL2.append(subsent)

    return (subsentL1, subsentL2)

def map_tokens(sent, tokens, token_index):
    input_ids = trans_tokenizer(sent, return_tensors="pt").input_ids
    trans_tokens = trans_tokenizer.batch_decode(input_ids[0], skip_special_tokens=True)
    trans_tokens = trans_tokens[:-1]

    trans_idx=0
    char_idx = 0
    trans_char = 0
    idx =0
    token_map={}
    while trans_idx < len(trans_tokens) and idx < len(tokens):
        if len(trans_tokens[trans_idx]) == 0:
            trans_idx+=1
            print("Warning! Empty token!")
            continue

        if char_idx >= len(tokens[idx]):
            if not trans_idx in token_map:
                token_map[trans_idx] = idx
            char_idx = 0
            idx +=1
            if idx >= len(tokens):
                break
        if trans_char >= len(trans_tokens[trans_idx]):
            if not trans_idx in token_map:
                token_map[trans_idx] =  idx
            trans_char = 0
            trans_idx +=1
            if len(trans_tokens[trans_idx]) == 0:
                trans_idx+=1
                print("Warning! Empty token!")
                continue
            if trans_idx >= len(trans_tokens):
                break
        if tokens[idx][char_idx] == trans_tokens[trans_idx][trans_char]:
            char_idx += 1
            trans_char += 1
    return [token_map[i] for i in token_index]


def generate_sentences(sent, sent_idx):
    tokens = tokenizer.tokenize(sent)
    pos_inf = nltk.tag.pos_tag(tokens)

    replaced_token_index = []
    replaced_token = []
    pos_tag = []
    log_grad_lst = []
    log_grad = None

    if software == 'transformer' and approach != "":
        token_idx, top_tokens = sent_data[sent_idx]["token_index"], sent_data[sent_idx]["top_tokens"]
        token_idx = map_tokens(sent, tokens, token_idx)
        pos_inf = [pos_inf[i] for i in token_idx]
        
        if approach == "gri":
            log_grad =  dict(zip(token_idx, sent_data[sent_idx]["grad"]))
        elif approach == "wali":
            log_grad = dict(zip(token_idx, sent_data[sent_idx]["logits"]))

        new_sentences, masked_indexes = [], []
        for idx, (word, tag) in enumerate(pos_inf):
            if word in sim_dict:
                masked_indexes.append((token_idx[idx], tag))

    else:
        new_sentences, masked_indexes = [], []
        for idx, (word, tag) in enumerate(pos_inf):
            # only replace noun, adjective, number
            if word in sim_dict and (tag.startswith('NN') or tag.startswith('JJ') or tag=='CD'):
            #if word in sim_dict:
                masked_indexes.append((idx, tag))

    for (masked_index, tag) in masked_indexes:
        original_word = tokens[masked_index]

        # generate similar sentences
        for similar_word in sim_dict[original_word]:
            if len(new_sentences) >= mutant_limit:
                break
            tokens[masked_index] = similar_word			
            new_pos_inf = nltk.tag.pos_tag(tokens)

            # check that tag is still same type
            if (new_pos_inf[masked_index][1].startswith(tag[:2])):
                pos_tag.append(tag)
                replaced_token.append(original_word)
                replaced_token_index.append(masked_index)
                if log_grad:
                    log_grad_lst.append(log_grad[masked_index])
                new_sentence = detokenizer.detokenize(tokens)
                new_sentences.append(new_sentence)
                
        tokens[masked_index] = original_word
    # global result_stats
    # if new_sentences:
    #     if approach == "gri":
    #         added_csv = pd.DataFrame({"index":[sent_idx]*len(new_sentences),"test_sen": new_sentences,  "token_idx": replaced_token_index, "tokens":replaced_token, "pos": pos_tag, "grad": log_grad_lst})
    #         result_stats = pd.concat([result_stats, added_csv])
    #     elif approach == "wali":
    #         added_csv = pd.DataFrame({"index":[sent_idx]*len(new_sentences),"test_sen": new_sentences, "token_idx": replaced_token_index, "tokens":replaced_token, "pos": pos_tag, "logits": log_grad_lst})
    #         result_stats = pd.concat([result_stats, added_csv])
    #     else:
    #         added_csv = pd.DataFrame({"index":[sent_idx]*len(new_sentences),"test_sen": new_sentences,  "token_idx": replaced_token_index, "tokens":replaced_token, "pos": pos_tag})
    #         result_stats = pd.concat([result_stats, added_csv])
    return new_sentences



sim_dict = {}
with open(SIM_DICT_FILE, 'r') as f:
    lines = f.readlines()
    for l in lines:
        sim_dict[l.split()[0]] = l.split()[1:]
print ("created dictionary")

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch
import os

mode_name = '../transformer'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =AutoModelForSeq2SeqLM.from_pretrained(mode_name).to(device)
trans_tokenizer = AutoTokenizer.from_pretrained(mode_name, return_tensors="pt")


# load input sentences
ori_source_sents = []
with open('./dataset/'+dataset) as file:
    for line in file:
        ori_source_sents.append(line.strip())
print ('input sentences loaded')
print(f"Number of Lines in {dataset}", len(ori_source_sents))

# translate input sentences
ori_target_sents = []
with open('./dataset/'+dataset+"_target") as file:
    for line in file:
        ori_target_sents.append(line.strip())
# for ori_source_sent in tqdm(ori_source_sents):
#     if software=='google':
#         # Google translate
#         translation = translate_client.translate(
#             ori_source_sent,
#             target_language='zh-CN',
#             source_language='en')
#         ori_target_sent = translation['translatedText'].replace("&#39;", "'")
#     elif software=="bing":
#         # Bing translate
#         ori_target_sent = BingTranslate(apikey, ori_source_sent, 'en', 'zh-Hans')[0]
#     else:
#         input_ids = trans_tokenizer(ori_source_sent, return_tensors="pt").input_ids.to(device)
#         outputs = model.generate(input_ids)
#         ori_target_sent = trans_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
    
#     ori_target_sents.append(ori_target_sent)

# with open('./dataset/'+dataset+"_target", "w") as f:
#     for i in ori_target_sents:
#         f.write(i+"\n")
def read_from_wdiff(old_sentence, new_sentence):
    #print (old_sentence)
    f = open("temptest5.txt", "w")
    f.write(old_sentence.strip())
    f.close()
    f = open("temptest6.txt", "w")
    f.write(new_sentence.strip())
    f.close()

    diff = os.popen("wdiff temptest5.txt temptest6.txt")
    lines = diff.readlines()
    #print (lines)
    assert len(lines) == 1
    diff.close()

    return lines[0]
def sentences_from_wdiff(wdiff_line):
    count_o = []
    count_n = []
    bf = ""
    #for line in lines:
    now = wdiff_line
    now = bf + " " + now
    now = re.sub(r"([\[])([\-])", r" \1\2 ", now)
    now = re.sub(r"([\-])([\]])", r" \1\2 ", now)
    now = re.sub(r"([\{])([\+])", r" \1\2 ", now)
    now = re.sub(r"([\+])([\}])", r" \1\2 ", now)
    # print (now)
    
    # stable
    o = []
    n = []
    #print ("-----------------")
    words = now.strip().split()
    old_tokens = []
    new_tokens = []

    in_old = 0
    in_new = 0
    tokens = words
    for i in range(len(words)):
        if in_old > 0 and tokens[i] not in ["{+", "+}", "[-", "-]"]:
            old_tokens.append(tokens[i])
        elif in_new > 0 and tokens[i] not in ["{+", "+}", "[-", "-]"]:
            new_tokens.append(tokens[i])
        elif tokens[i] not in ["{+", "+}", "[-", "-]"]:
            old_tokens.append(tokens[i])
            new_tokens.append(tokens[i])

        if tokens[i] == "{+":
            in_new += 1
        elif tokens[i] == "+}":
            in_new -= 1

        if tokens[i] == "[-":
            in_old += 1
        elif tokens[i] == "-]":
            in_old -= 1

        #assert in_new < 2 and in_new > -1 and in_old < 2 and in_old > -1 and in_new + in_old < 2
    o.append(old_tokens)
    n.append(new_tokens)
    #o.append(words)
    #n.append(words)
    #print (" ".join(words))
    for i in range(len(words)):
        if words[i] == "[-":
            for t in range(i + 1, len(words)):
                if words[t] == "-]":
                    sentence = []
                    in_new = 0
                    if t - i >= 5:
                        break
                    for k in range(len(words)):
                        if k >= i and k <=t:
                            continue
                        elif words[k] == "{+":
                            in_new += 1
                        elif words[k] == "+}":
                            in_new -= 1
                            assert in_new >= 0
                            continue
                        if in_new > 0:
                            continue
                        if words[k] in ["[-", "-]"]:
                            continue
                        else:
                            sentence.append(words[k])
                    #print (" ".join(sentence))
                    o.append(sentence)
                    break

        if words[i] == "{+":
            for t in range(i + 1, len(words)):
                if words[t] == "+}":
                    sentence = []
                    in_new = 0
                    if t - i >= 5:
                        break
                    for k in range(len(words)):
                        if k >= i and k <=t:
                            continue
                        elif words[k] == "[-":
                            in_new += 1
                        elif words[k] == "-]":
                            in_new -= 1
                            assert in_new >= 0
                            continue
                        if in_new > 0:
                            continue
                        if words[k] in ["{+", "+}"]:
                            continue
                        else:
                            sentence.append(words[k])
                    #print (" ".join(sentence))
                    n.append(sentence)
                    break

    for i in o:
        for t in n:
            count_o.append(i)
            count_n.append(t)

    return count_o, count_n
# for each input sentence, generate similar sentences and test cases
suspicous_issueL = []
count = 0
print("new sent generating...")
sent_idx=0
for ori_source_sent, ori_target_sent in tqdm(zip(ori_source_sents, ori_target_sents)):

    new_sentsL = generate_sentences(ori_source_sent, sent_idx)
    print("new",len(new_sentsL))
    for new_source in new_sentsL:
        # collect the target sent from machine translation software
        if software=='google':
            # Google translate
            translation = translate_client.translate(
                new_source,
                target_language='zh-CN',
                source_language='en')
            new_target = translation['translatedText'].replace("&#39;", "'")
        elif software =="bing":
            # Bing translate
            new_target = BingTranslate(apikey, new_source, 'en', 'zh-Hans')[0]
        else:
            input_ids = trans_tokenizer(new_source, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids)
            new_target = trans_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            

        # obtain the segmented one for Chinese
        # print("Original: ", ori_target_sent)
        # print("New: ", new_target)
        sentence1 = ' '.join(jieba.cut(ori_target_sent))
        sentence2 = ' '.join(jieba.cut(new_target))

        old_line = ori_target_sent.strip()#old chinese line
        new_line = new_target.strip()#new chinese
        wdiff = read_from_wdiff(" ".join(old_line), " ".join(new_line))
        #print("wdiff:", wdiff)

        old_tokens, new_tokens = sentences_from_wdiff(wdiff)

        # obtain different words by wdiff
        set1, set2 = wordDiffSet(sentence1, sentence2)

        # get sub sentences
        # subsentL1, subsentL2 = getSubSentenceList(sentence1, sentence2, set1, set2)


        lcs_d_sim = lcs_sim_d(old_tokens, new_tokens)
        tf_d_sim = tf_cos_d_sim(old_tokens, new_tokens)
        bleu_d_sim = bleu_sim_d(old_tokens, new_tokens)
        #ed_sim = getSubSentSimilarity(subsentL1, subsentL2)
        ed_sim = ed_sim_d(old_tokens, new_tokens)
        lcs_flag = (1 if lcs_d_sim< 0.963 else 0)
        tf_flag = (1 if tf_d_sim< 0.999 else 0)
        bleu_flag = (1 if bleu_d_sim< 0.906 else 0)
        ed_flag = (1 if ed_sim< 0.963 else 0)

        if lcs_flag or tf_flag or bleu_flag or ed_flag:
            suspicous_issueL.append((str(count), ori_source_sent, ori_target_sent, new_source, new_target))
            # suspicous_issueL.append((str(count), str(similarity), ori_source_sent, ori_target_sent, new_source, new_target))
            count += 1
            # added_row = pd.DataFrame({"index":[sent_idx],"test_sen": [new_source], "test_sen_trans": [new_target], "LCS_res":[lcs_flag],"ED_res": [ed_flag],	"TF-IDF_res":[tf_flag], "BLEU_res":[bleu_flag]})
            # distance_result = pd.concat([distance_result, added_row])
    sent_idx+=1

# result_stats.to_csv(result_stats_filename, index=False)
# print(f"{result_stats_filename} is saved in current dir.")
# distance_result.to_csv(distance_result_filename, index=False)
# print(f"{distance_result_filename} is saved in current dir.")

writebugs = open(f'./results/{dataset}_{software}_{approach}_bug.txt', 'w')

for (ID, ori_source_sent, ori_target_sent, new_source, new_target) in suspicous_issueL:
    writebugs.write('Issue: '+ID+'\n')
    # writebugs.write(similaritystr+'\n')
    writebugs.write(ori_source_sent+'\n')
    writebugs.write(ori_target_sent+'\n')
    writebugs.write(new_source+'\n')
    writebugs.write(new_target+'\n')

writebugs.close()

