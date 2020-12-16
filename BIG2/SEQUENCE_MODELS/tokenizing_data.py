import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


with open("data/ThreeMenInABoat.txt","r") as f:
    string_data = f.read().replace("\n"," ")

string_data = string_data.replace("--"," ")

sentence_data=string_data.split(".")

#print(sentence_data)

additional_filters='-''""'

token = Tokenizer(num_words=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + additional_filters,
                  lower=True,
                  split=" ",
                  char_level=False,
                  oov_token="UNK",
                  document_count=0)
token.fit_on_texts(sentence_data)

tokenizer_config = token.get_config()
print(tokenizer_config.keys())

#print(tokenizer_config["word_index"])

#print("\n\n\n\n\n\n\n")

import json
word_counts = json.loads(tokenizer_config['word_counts'])
#print(word_counts)
print(word_counts["the"])

index_word = json.loads(tokenizer_config['index_word'])
word_index=json.loads(tokenizer_config["word_index"])
#print(sentence_data)

print(sentence_data[:5])

sentence_seq = token.texts_to_sequences(sentence_data)
print(sentence_seq[0:5])

senetn = token.sequences_to_texts(sentence_seq)
print(senetn[:5])