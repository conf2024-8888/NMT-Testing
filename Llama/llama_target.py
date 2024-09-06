import ollama
from tqdm import tqdm
import re
import pandas as pd
response = ollama.chat(model='llama3', messages=[
    {"role": "system", "content": "I give you a sentence, give me 10 word where the replacement of the word with synonym would most likely to induce bug in translation."},
    {"role": "user", "content": "The old rule started to seem dated and out of place."},
])
print(response['message']['content'])

dataset="politics"
with open(f"./data/{dataset}", "r") as f:
    sentences = f.readlines()

index=0
result =[]
for en_sent in tqdm(sentences):
    pair = {}
    response = ollama.chat(model='llama3', messages=[
    {"role": "system", "content": "I give you a sentence, give me 10 word where the replacement of the word with synonym would most likely to induce bug in translation."},
    {"role": "user", "content": en_sent},
]) 
    res = response['message']['content']
    # print("THis", en_sent)
    # print(res)
    pair["en"] = en_sent
    res_num = 1
    tokens = []
    #print(res)
    while True:
        try:
            # print(res_num)
            # print(re.search(f"{res_num}\. (.*)\n", res).group(1))
            if res_num ==10:
                token = re.search(f"{res_num}\. (.*)", res).group(1).split("->")[0].split("(")[0]
            else:
                token = re.search(f"{res_num}\. (.*)\n", res).group(1).split("->")[0].split("(")[0]
            tokens.append(token)
            res_num+=1
            print(tokens)

        except:
            print("THIS: ", en_sent)
            break
            
    pair["top_tokens"] = tokens

    index+=1
    result.append(pair)
                
import json
with open(f"en_token_llama_{dataset}.json", "w") as f:
    json.dump(result, f)



