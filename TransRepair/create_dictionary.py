import spacy
import numpy as np
from scipy import spatial
from collections import defaultdict
from nltk.corpus import words
from tqdm import tqdm
import nltk

nltk.download('words')
valid_eng_words = set(words.words())

# Download glove at https://nlp.stanford.edu/projects/glove/, unzip, and put files in GLOVE_LOCATION
GLOVE_LOCATION = "./"
SIM_DICT_FILE = "similarity_dict_200d.txt"

# install spacy and download model with `python -m spacy download en_core_web_md`
spacy_dict = spacy.load('en_core_web_md')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
spacy_vocab = set(spacy_dict.vocab.strings)
print("SPacy ", len(spacy_vocab))

glove_dict = {}
def get_spacy_vec(word):
  return spacy_dict(word).vector

with open(GLOVE_LOCATION + "glove.6B.200d.txt", 'r') as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        if word.isalpha() and len(word) >= 3 and word in valid_eng_words and word in spacy_vocab \
        and not np.count_nonzero(get_spacy_vec(word)) == 0 and not word in spacy_stopwords:
            glove_dict[word] = vector
        

print(len(glove_dict))
#print(glove_dict.keys())
def get_glove_vec(word):
    return glove_dict[word]




def similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print("creating dictionary...")

filtered_words = []

similarity_dict = defaultdict(list)


sorted_glove_key = sorted(glove_dict.keys())
for idx, word in enumerate(sorted_glove_key):
    # if len(word) < 2 \
    #     or np.count_nonzero(get_spacy_vec(word)) == 0:
    #     continue
    print("now at ", idx, word)
    for other_word in sorted_glove_key[idx+1:]:
        # already encountered...
        # if len(word) < 2 or other_word <= word or other_word not in spacy_dict.vocab:
        #     continue
        # if other_word :
        #     continue
        similarity1 = similarity(get_glove_vec(word), get_glove_vec(other_word))
        similarity2 = similarity(get_spacy_vec(word), get_spacy_vec(other_word))

        if similarity1 > 0.8 and similarity2 > 0.8:
            print('Simi: ', word, other_word)
            similarity_dict[word].append(other_word)
            similarity_dict[other_word].append(word)
    
print("simi_dict:", similarity_dict[:10])
# def to_file(d):
#     return '\n'.join("%s %s" % (key, ' '.join(map(str, values))) for key, values in d.items())

# with open(SIM_DICT_FILE, 'w') as f:
# 	f.write(to_file(similarity_dict))

def to_line(k, v):
    return ("%s %s" % (k, ' '.join(map(str, v))))

with open(SIM_DICT_FILE, 'w') as f:
    for key, values in similarity_dict.items():
        currentLine = to_line(key, values)
        f.write(currentLine+'\n')
