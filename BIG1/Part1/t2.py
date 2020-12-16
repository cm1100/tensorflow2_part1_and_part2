import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D


model = Sequential([
    Conv2D(16,(3,3),padding="SAME",strides=2,activation="relu",input_shape=(28,28,1),data_format="channels_last"),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10,activation="softmax")
])


print(model.summary())