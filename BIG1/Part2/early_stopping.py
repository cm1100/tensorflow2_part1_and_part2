import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np




dataset = load_diabetes()

X=dataset["data"]
y=dataset["target"]

y=(y-np.mean(y))/np.std(y)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
print(Xtrain.shape,Xtest.shape,ytrain.shape,ytest.shape)

model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(Xtrain.shape[1],)),
    Dense(64,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])


model.compile(optimizer="adam",loss="mse")
'''
history = model.fit(Xtrain,ytrain,
                    epochs=1000,
                    validation_split=0.15,batch_size=64,verbose=False,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,monitor="val_loss",min_delta=0.001,mode="min")])
'''

##  Learning rate scheduler callback
'''
def lrschedular(epoch,lr):
    if epoch%2==0:
        return lr
    else:
        return lr+epoch/1000

history = model.fit(Xtrain,ytrain,callbacks=[tf.keras.callbacks.LearningRateScheduler(lrschedular,verbose=1)],verbose=False,epochs=10)
'''

## CSV loger callback-> to write into csv file

#history=model.fit(Xtrain,ytrain,epochs=10,callbacks=[tf.keras.callbacks.CSVLogger("results.csv")])

## Lambda Callbacks
'''
epoch_callback =tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch,logs: print(f"starting epoch {epoch}"))


history = model.fit(Xtrain,ytrain,callbacks=[epoch_callback],epochs=10)

'''

## To reduce learning rate on a plateau

history = model.fit(Xtrain,ytrain,epochs=10,callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.2,verbose=2)],verbose=False)

from tensorflow.keras.layers import LeakyReLU