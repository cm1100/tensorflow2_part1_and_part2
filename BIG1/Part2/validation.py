import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.datasets import  load_diabetes

dataset= load_diabetes()
print(type(dataset))

#print(dataset["DESCR"])
print(dataset.keys())

X=dataset["data"]
y=dataset["target"]

#y=(y-np.mean(y))/np.std(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

model = Sequential([
    Dense(128,activation="relu",input_shape=(X_train.shape[1],)),
    Dense(128,activation="relu"),
    Dense(128,activation="relu"),
    Dense(128,activation="relu"),
    Dense(128,activation="relu"),
    Dense(128,activation="relu"),
    Dense(1)
])

print(model.summary())

model.compile(optimizer="adam",loss="mse",metrics=["mae"])
history=model.fit(X_train,y_train,epochs=1000,validation_split=0.15)

print(model.evaluate(X_test,y_test,verbose=2))