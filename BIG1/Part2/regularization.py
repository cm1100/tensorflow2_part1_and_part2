import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import  Dense,Dropout
from tensorflow.keras.models import Sequential
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



dataset = load_diabetes()

X=dataset["data"]
y=dataset["target"]

y=(y-np.mean(y))/np.std(y)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)

wd=0.001
rate=0.5
model = Sequential()
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu",input_shape=(Xtrain.shape[1],)))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dense(1))

print(model.summary())

model.compile(optimizer="adam",loss="mse",metrics=["mae"])

hist = model.fit(Xtrain,ytrain,epochs=100,validation_split=0.15,batch_size=64)

print(hist.history)

print(model.evaluate(Xtest,ytest))