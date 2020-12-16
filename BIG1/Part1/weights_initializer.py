import tensorflow as tf
import  numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,Flatten

model=Sequential([
    Conv1D(filters=16,kernel_size=3,input_shape=(128,64),kernel_initializer="random_uniform",bias_initializer="zeros",activation="relu"),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(units=64,activation="relu",kernel_initializer="he_uniform",bias_initializer="ones")

])

model.add(Dense(64,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),activation="relu"))


model.add(Dense(64,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0,seed=None),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),activation="relu"))

#including your own weights
import tensorflow.keras.backend as K
def my_init(shape,dtype=None):
    return K.random_normal(shape,dtype=dtype)

model.add(Dense(64,kernel_initializer=my_init))

print(model.summary())

import matplotlib.pyplot as plt

fig,axs=plt.subplots(5,2,figsize=(12,16))
fig.subplots_adjust(hspace=0.5,wspace=0.5)

weight_layers=[layers for layers in model.layers if len(layers.weights)>0]

for i,l in enumerate(weight_layers):
    for j in [0,1]:
        axs[i,j].hist(l.weights[j].numpy().flatten(),align="left")
        axs[i,j].set_title(l.weights[j].name)

plt.show()


