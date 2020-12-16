import tensorflow as tf
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense,LSTM,GRU
import numpy as np


simple_Rnn=SimpleRNN(units=16,)


sequence = tf.constant([[[1.,1.],[2.,2.],[3.,3.]]])
print(sequence.shape)
layer_out= simple_Rnn(sequence)
print(layer_out)


def get_and_pad_imdb_dataset(num_words=1000,maxlen=2000,index_form=2):

    from tensorflow.keras.datasets import imdb

    (Xtrain,ytrain),(Xtest,ytest)=imdb.load_data(num_words=num_words,
                                                 skip_top=0,
                                                 oov_char=2,
                                                 index_from=index_form,maxlen=maxlen)

    Xtrain=tf.keras.preprocessing.sequence.pad_sequences(Xtrain,maxlen=None,
                                                         padding="pre",
                                                         truncating="pre",
                                                         value=0)

    Xtest = tf.keras.preprocessing.sequence.pad_sequences(Xtest, maxlen=None,
                                                           padding="pre",
                                                           truncating="pre",
                                                           value=0)
    return (Xtrain,ytrain),(Xtest,ytest)


def imdb_word_index(num_words=10000,index_from=2):
    imdb_word_index1=tf.keras.datasets.imdb.get_word_index()
    imdb_word_index1={key:value+index_from for key,value in imdb_word_index1.items() if value+index_from<num_words}
    return imdb_word_index1


(Xtrain,ytrain),(Xtest,ytest )=get_and_pad_imdb_dataset(maxlen=250)
print(Xtrain.shape,ytrain.shape)

imdb_word_index = imdb_word_index()
#print(imdb_word_index)

max_index = max(imdb_word_index.values())
embeddding_dim=16

model = tf.keras.models.Sequential([
    Embedding(input_dim=max_index+1,output_dim=embeddding_dim,mask_zero=True),
    LSTM(16),
    Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

history = model.fit(Xtrain,ytrain,epochs=5,validation_split=0.15,batch_size=32)

inv_imdb_index = {value:key for key,value in imdb_word_index.items()}

ex=Xtest[1]
print(Xtest.shape)

print(ex)

real =[inv_imdb_index[num] for num in ex if num>2]

ex=ex[np.newaxis,:]
print(ex.shape)
print(real)
print(model.predict(ex))