import tensorflow as tf
from tensorflow.keras.layers import Embedding
import pandas as pd
import matplotlib.pyplot as plt

def get_and_pad_imdb_dataset(num_words=1000,maxlen=2000,index_form=2):

    from tensorflow.keras.datasets import imdb

    (Xtrain,ytrain),(Xtest,ytest)=imdb.load_data(num_words=num_words,
                                                 skip_top=0,
                                                 oov_char=2,
                                                 index_from=index_form,maxlen=None)

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

imdb_index=imdb_word_index()

#print(imdb_index)
print(Xtrain.shape,Xtest.shape)
rev_imdb_index = {value:key for key,value in imdb_index.items()}
#print(rev_imdb_index)

print(Xtrain[0])

l1=[rev_imdb_index[index] for index in Xtrain[0] if index>2]
print(l1)

max_index_value = max(imdb_index.values())
embedding_dim=16


model =tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2494,),name="inp_1"),
    tf.keras.layers.Embedding(input_dim=max_index_value+1,output_dim=embedding_dim,mask_zero=False,name="embd"),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Flatten(name="flat_1"),
    tf.keras.layers.Dense(1,activation="sigmoid",name="last")
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss="binary_crossentropy",metrics=["accuracy"])
print(model.summary())

hist = model.fit(Xtrain,ytrain,epochs=10,batch_size=32,validation_split=0.15)


print(hist.history.keys())
df = pd.DataFrame(hist.history)
print(df.head())
df.plot(y="accuracy")
df.plot(y="val_accuracy")

plt.show()

weights = model.get_layer("embd").get_weights()[0]

print(model.get_layer("last").weights[0].shape)
print(model.get_layer("embd").output)
#print(model.get_layer("inp_1").weights[0].shape)



import io
from os import path

'''
out_v = io.open("files/vecs.tsv","w+",encoding="utf-8")
out_m = io.open("files/meta.tsv","w+",encoding="utf-8")

k=0
for word,token in imdb_index.items():
    if k!=0:
        out_m.write("\n")
        out_v.write("\n")

    out_v.write("\t".join([str(x) for x in weights[token]]))
    out_m.write(word)
    k+=1
out_v.close()
out_m.close()
'''