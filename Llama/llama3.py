import ollama
from tqdm import tqdm
import re
import pandas as pd
response = ollama.chat(model='llama3', messages=[
    {"role": "system", "content": "when I give you a sentence, you can only replace ONE random word with its synonym and try not to change the semantic of the phrase, generate FIVE sentences as required"},
    {"role": "user", "content": "The old rule started to seem dated and out of place."},
])
print(response['message']['content'])

result = pd.DataFrame()

dataset="business"
with open(f"data/{dataset}", "r") as f:
    sentences = f.readlines()
index = 0
for en_sent in tqdm(sentences):
    pair = {}
    while True:
        response = ollama.chat(model='llama3', messages=[
        {"role": "system", "content": "I give you a sentence, you can only replace ONE random word with its synonym and try not to change the semantic of the phrase, generate FIVE sentences as required. For each alternative sentence you can replace different word"},
        {"role": "user", "content": en_sent},
    ])
        res = response['message']['content']
        # print("THis", en_sent)
        # print(res)
        
        try:
            res1 = re.search("1. (.*)\n", res).group(1).split("(")[0]
            #token = re.search("Replaced (.*)\n", res)
            res2 = re.search("2. (.*)\n", res).group(1).split("(")[0]
            res3 = re.search("3. (.*)\n", res).group(1).split("(")[0]
            res4 = re.search("4. (.*)\n", res).group(1).split("(")[0]
            res5 = re.search("5. (.*)", res).group(1).split("(")[0]
        except:
            
            print("THis", en_sent)

            continue
        else:
            added_csv = pd.DataFrame({"index": [index]*5, "test_sen":[res1, res2, res3, res4, res5]})
            index+=1
            # pair["test_sen"] = [res1, res2, res3, res4, res5]
            #print(added_csv)
            result = pd.concat([result, added_csv])
            break
    
    
result.to_csv(f"test_case_{dataset}.csv", index=False)



