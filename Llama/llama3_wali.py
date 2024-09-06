import ollama
from tqdm import tqdm
import re
import pandas as pd
response = ollama.chat(model='llama3', messages=[
    {"role": "system", "content": "when I give you a sentence, you can only replace 'rule' with its synonym and try not to change the semantic of the phrase, output ONLY the generated sentence."},
    {"role": "user", "content": "The old rule started to seem dated and out of place."},
])
print(response['message']['content'])

result = pd.DataFrame()
dataset='business'
approach = 'wali'
import json
with open(f"./en_token_{approach}_{dataset}.json", "r") as f:
        sent_data = json.load(f)
index = 0
for en_sent in tqdm(sent_data):
    test_sen = []
    grad = []
    for i in en_sent["top_tokens"][:5]:

        response = ollama.chat(model='llama3', messages=[
        {"role": "system", "content": f"I give you a sentence, you can only replace {i} with its synonym and try not to change the semantic of the phrase, output ONLY the generated sentence."},
        {"role": "user", "content": en_sent["en"]},
    ])
        
        res = response['message']['content']
        # print("THis", en_sent)    with open(f"./en_token_{approach}_{dataset}_{token_num}_8.json", "r") as f:
        print(res)
        test_sen.append(res)
        grad.append(en_sent["logits"])
        
    added_csv = pd.DataFrame({"index": [index]*len(test_sen), "test_sen":test_sen, "logits": grad})
    index+=1
    # pair["test_sen"] = [res1, res2, res3, res4, res5]
    #print(added_csv)
    result = pd.concat([result, added_csv])
    
        
    
result.to_csv(f"test_case_{dataset}_{approach}.csv", index=False)



