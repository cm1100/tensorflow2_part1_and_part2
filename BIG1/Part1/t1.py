import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten

model =Sequential([
    Flatten(input_shape=(28,28)),

    Dense(16,activation="relu",name="layer1"),
    Dense(16,activation="relu"),
    Dense(10,activation="softmax")
])


#print(model.weights)

print(model.summary())

print(np.zeros((2,5)))