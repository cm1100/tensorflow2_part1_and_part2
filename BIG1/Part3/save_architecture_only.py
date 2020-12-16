import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorflow.keras.callbacks import Callback
import json



(X1train,y1train),(X1test,y1test) = tf.keras.datasets.cifar10.load_data()

print(X1train.shape,y1train.shape,X1test.shape,y1test.shape)

X1train=X1train/255
X1test=X1test/255

X1train=X1train[:10000]
y1train=y1train[:10000]
X1test=X1test[:1000]
y1test=y1test[:1000]
print(X1train.shape,y1train.shape,X1test.shape,y1test.shape)


def get_model():

    model = tf.keras.models.Sequential([

        Conv2D(16,(3,3),input_shape=(32,32,3),activation="relu",name="conv_1"),
        Conv2D(8,(3,3),activation="relu",name="conv_2"),
        MaxPooling2D((4,4),name="pool_1"),
        Flatten(name="flatten"),
        Dense(32,activation="relu",name="dense_1"),
        Dense(10,activation="softmax",name="dense_2")


    ])

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

model = get_model()

config = model.get_config()
print(config)

# new model created from the config history
model_new = tf.keras.Sequential.from_config(config)
print(model_new.summary())

#how to save in file
json_str = model.to_json()
print(json_str)

with open("model.json", "w") as f:
    json.dump(json_str,f)
    f.close()
del json_str

with open("model.json", "r") as f:
    json_str=json.load(f)

model_new_2 = tf.keras.models.model_from_json(json_str)
print(model_new_2.summary())

