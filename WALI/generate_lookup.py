from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline, MarianTokenizer, MarianTokenizer, TFMarianMTModel, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
mode_name = '../transformer'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model =AutoModelForSeq2SeqLM.from_pretrained(mode_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(mode_name, return_tensors="pt")
# from datasets import load_dataset, load_metric
# raw_datasets = load_dataset("wmt17", "zh-en", cache_dir="/data2/hanyings/.cache")

with open("./f_en_mu_align_stop.txt") as f:
    enlines = f.readlines()

with open("./f_en_mu_align_stop.index") as f:
    index = f.readlines()


look = open("./lookup_align.txt", "w")


enline_mut = [enlines[i].strip() for i in range(len(enlines)) if i%2!=0 or i==0 or index[i//2]!= index[i//2-1]]
#print(enline_mut)
for i in tqdm(range(len(enline_mut))):
    # if i==0 or index[i//2]!= index[i//2-1]:
    #     print(enline_mut[i])
    #     idx = int(index[i//2])
    #     print(idx)
    #     print(raw_datasets['test'][idx]['translation']["zh"])
    #print(enline_mut[i])
    input_ids = tokenizer(enline_mut[i], return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print(translated)
    look.write(enline_mut[i]+ "\n")
    look.write(translated[0]+ "\n")

