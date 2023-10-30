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
nltk.download('stopwords')
sw_nltk = stopwords.words('english')

# import sys
# mad_fact = float(sys.argv[1])


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
def gradient_search_nostop(en_sentence, zh_sentence, k=10):
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
    
    for i in range(len(token_ids[0])):
        current_token = token_ids[0][i]
        if is_stopword(current_token, tokenizer):
            grad_norm[i] = 0
            
    idx = significant_large_elements(grad_norm)
    # idx = np.argsort(grad_norm).tolist()[::-1]
    idx =[ i for i in idx if not is_stopword(token_ids[0][i], tokenizer)]

    idx = idx[:min(len(idx), k)]
    input_tokens = [token_ids[0][i] for i in idx ]
    return idx, [tokenizer.decode(t) for t in input_tokens if tokenizer.decode(t) not in string.punctuation]
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

def gradient_search(en_sentence, zh_sentence, k=10):
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
    
    idx = significant_large_elements(grad_norm)
    # for i in range(len(token_ids[0])):
    #     current_token = token_ids[0][i]
    #     if is_stopword(current_token, tokenizer):
    #         grad_norm[i] = 0
    #idx = np.argsort(grad_norm).tolist()[::-1]
    
    grad_norm = np.asarray(grad_norm)
    grad_norm = grad_norm.astype(float)
    idx =[ i for i in idx if punctuation(tokenizer.decode(token_ids[0][i]))]
    idx=idx[:min(len(idx), k)]
    grad_lst = [grad_norm[i] for i in idx]
    #print(grad_lst)
    input_tokens = [token_ids[0][i] for i in idx]
    return idx, [tokenizer.decode(t) for t in input_tokens], grad_lst


def loop_data_nostop(dataset):
    for pair in tqdm(dataset['translation']):
        #pair["token_index"], pair["top_tokens"] = gradient_search_nostop(pair["en"], pair["zh"])
        pair["token_index"], pair["top_tokens"], pair["grad"] = gradient_search(pair["en"], pair["zh"])
        #print(dataset['translation'][0])
    return dataset['translation']
result = loop_data_nostop(raw_datasets["test"][:])

import json

with open("./en_token_stop.json", "w") as f:
    json.dump(result, f)
    
    