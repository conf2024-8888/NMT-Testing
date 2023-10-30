import sys
from datasets import load_dataset, load_metric
import transformers
import os
from tqdm import tqdm

from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline, MarianTokenizer, MarianTokenizer, TFMarianMTModel, AutoModelForSeq2SeqLM
import torch

import re
from nltk.corpus import stopwords
import string 
import nltk
import numpy as np

mode_name = '../transformer'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =AutoModelForSeq2SeqLM.from_pretrained(mode_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(mode_name, return_tensors="pt")

nltk.download('stopwords')
sw_nltk = stopwords.words('english')
def is_stopword(word):
    #word = tokenizer.decode(token_id)
    if word in string.punctuation:
        return True
    word = word.lower()
    word = re.sub(r'[^\w\s]', '', word)
    stopword = stopwords.words('english') + ["</s>", "<unk>", ">>cmn_Hans<<", "<pad>"]
    if word in stopword or not word:
        return True
    return False 
def gradient_search_nostop(en_sentence, zh_sentence, k=5):
    token_ids = tokenizer(en_sentence, return_tensors="pt").input_ids.to(device)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(zh_sentence, return_tensors="pt").input_ids.to(device)
    # get your token embeddings
    token_embeds=model.get_input_embeddings().weight[token_ids].clone()
    token_embeds.retain_grad() 
    # get model output that contains loss value
    outs = model(inputs_embeds=token_embeds,labels=labels, output_hidden_states=True, output_attentions=True)
    loss=outs.loss
    loss.backward(retain_graph=True)
    grad=token_embeds.grad.cpu()
    grad_norm = torch.norm(grad, dim=2).squeeze(dim=0)
    
    # idx = significant_large_elements(grad_norm)
    # for i in range(len(token_ids[0])):
    #     current_token = token_ids[0][i]
    #     if current_token in string.punctuation:
    #         grad_norm[i] = 0
    idx = np.argsort(grad_norm).tolist()[::-1]
    
    grad_norm = np.asarray(grad_norm)
    grad_norm = grad_norm.astype(float)
    idx =[ i for i in idx if punctuation(tokenizer.decode(token_ids[0][i])) and 
          not is_stopword(tokenizer.decode(token_ids[0][i])) and i != 0]
    idx=idx[:min(len(idx), k)]
    grad_lst = [grad_norm[i] for i in idx]
    #print(grad_lst)
    input_tokens = [token_ids[0][i] for i in idx]
    return idx, [tokenizer.decode(t) for t in input_tokens], grad_lst
def punctuation(token):
    return not all(char in string.punctuation for char in token) and token not in ["</s>", "<unk>", ">>cmn_Hans<<", "<pad>", "."] and len(token)>1

def significant_large_elements(arr):
    arr = np.asarray(arr)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    threshold = median +  1.5*mad
    significant_indices = np.where(arr > threshold)[0]
    if len(significant_indices) == 0:
        return np.argsort(arr).tolist()[::-1][:5]
    significant_values = arr[significant_indices]
    sorted_indices = significant_values.argsort()[::-1]
    return significant_indices[sorted_indices].tolist()

def gradient_search(en_sentence, zh_sentence, k=5):
    token_ids = tokenizer(en_sentence, return_tensors="pt").input_ids.to(device)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(zh_sentence, return_tensors="pt").input_ids.to(device)
    # get your token embeddings
    token_embeds=model.get_input_embeddings().weight[token_ids].clone()
    token_embeds.retain_grad() 
    # get model output that contains loss value
    outs = model(inputs_embeds=token_embeds,labels=labels, output_hidden_states=True, output_attentions=True)
    loss=outs.loss
    loss.backward(retain_graph=True)
    grad=token_embeds.grad.cpu()
    grad_norm = torch.norm(grad, dim=2).squeeze(dim=0)
    
    # idx = significant_large_elements(grad_norm)
    # for i in range(len(token_ids[0])):
    #     current_token = token_ids[0][i]
    #     if current_token in string.punctuation:
    #         grad_norm[i] = 0
    idx = np.argsort(grad_norm).tolist()[::-1]
    
    grad_norm = np.asarray(grad_norm)
    grad_norm = grad_norm.astype(float)
    idx =[ i for i in idx if punctuation(tokenizer.decode(token_ids[0][i])) and i != 0]
    idx=idx[:min(len(idx), k)]
    grad_lst = [grad_norm[i] for i in idx]
    #print(grad_lst)
    input_tokens = [token_ids[0][i] for i in idx]
    return idx, [tokenizer.decode(t) for t in input_tokens], grad_lst


# def loop_data_nostop(dataset):
#     for pair in tqdm(dataset['translation']):
#         #pair["token_index"], pair["top_tokens"] = gradient_search_nostop(pair["en"], pair["zh"])
#         pair["token_index"], pair["top_tokens"], pair["grad"] = gradient_search(pair["en"], pair["zh"])
#         #print(dataset['translation'][0])
#     return dataset['translation']
result = []
dataset="wmt17"
with open(f"dataset/{dataset}", "r") as f:
    sentences = f.readlines()
for en_sent in tqdm(sentences):
    pair = {}
    pair["en"] = en_sent
    input_ids = tokenizer(en_sent, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    translated_sent = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    pair["zh"] = translated_sent
    pair["token_index"], pair["top_tokens"], pair["grad"] = gradient_search_nostop(pair["en"], pair["zh"], k=10)
    result.append(pair)

import json

with open(f"./en_token_gri_{dataset}.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
    