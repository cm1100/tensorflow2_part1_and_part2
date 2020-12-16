import tensorflow as tf
from tensorflow.keras.layers import LSTM,GRU,Bidirectional,Dense,Embedding



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

(Xtrain,ytrain),(Xtest,ytest)=get_and_pad_imdb_dataset()
imdb_index = imdb_word_index()

max_value=max(imdb_index.values())
embedding_dim=16

model=tf.keras.Sequential([

    Embedding(input_dim=max_value+1,output_dim=embedding_dim,mask_zero=True),
    Bidirectional(layer=LSTM(8,return_sequences=True),merge_mode="sum",backward_layer=GRU(8,return_sequences=True,go_backwards=True)),
    LSTM(8),
    Dense(1,activation="sigmoid")

])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
his=model.fit(Xtrain,ytrain,validation_split=0.15,epochs=5)


