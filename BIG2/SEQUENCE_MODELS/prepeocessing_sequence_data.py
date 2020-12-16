import tensorflow as tf
import tensorflow.keras.datasets.imdb as imdb


(Xtrain,ytrain),(Xtest,ytest)=imdb.load_data(num_words=1000,oov_char=2)

print(Xtrain.shape)

imdb_word_index=imdb.get_word_index()
#print(imdb_word_index)

print(imdb_word_index["the"])

inv_imdb_index={value:key for key,value in imdb_word_index.items()}

print(inv_imdb_index)
