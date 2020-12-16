import tensorflow as tf
import tensorflow.keras.datasets.imdb as imdb
import numpy as np

(Xtrain,ytrain),(Xtest,ytest)=imdb.load_data()

print(len(Xtrain[0]))

padded_Xtrain = tf.keras.preprocessing.sequence.pad_sequences(Xtrain,maxlen=300,padding="post",truncating="pre")
print(padded_Xtrain.shape)


padded_Xtrain=np.expand_dims(padded_Xtrain,-1)
tf_Xtrain = tf.convert_to_tensor(padded_Xtrain,dtype="float32")
masking_layer = tf.keras.layers.Masking(mask_value=0.0)

masked_Xtrain=masking_layer(tf_Xtrain)##adds a property to the Xtrain

print(masked_Xtrain._keras_mask[0])
print(masked_Xtrain[0])