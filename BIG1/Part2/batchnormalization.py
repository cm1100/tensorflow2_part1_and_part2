import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import  Dense,Dropout
from tensorflow.keras.models import Sequential
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,BatchNormalization
import  pandas as pd


dataset = load_diabetes()

X=dataset["data"]
y=dataset["target"]

y=(y-np.mean(y))/np.std(y)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
print(Xtrain.shape,Xtest.shape,ytrain.shape,ytest.shape)

model = Sequential([
    Dense(64,activation="relu",input_shape=(Xtrain.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256,activation="relu")
])

#print(model.summary())

model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95, #The hyperparameter momentum is the weighting given to the previous running mean when re-computing it with an extra minibatch. By default, it is set to 0.99.
    epsilon=0.005, #The hyperparameter  𝜖 is used for numeric stability when performing the normalisation over the minibatch. By default it is set to 0.001.
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.5),
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
    #The parameters  𝛽 β and  𝛾 γ are used to implement an affine transformation after normalisation. By default,  𝛽
#β is an all-zeros vector, and  𝛾
#γ is an all-ones vector.

)
)

model.add(Dense(1))

print(model.summary())

model.compile(optimizer="adam",loss="mse",metrics=["accuracy"])

history = model.fit(Xtrain,ytrain,validation_split=0.15,epochs=100,batch_size=64)

df = pd.DataFrame(history.history)

print(df.head())