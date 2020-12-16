import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import os



(X1train,y1train),(X1test,y1test) = tf.keras.datasets.cifar10.load_data()

print(X1train.shape,y1train.shape,X1test.shape,y1test.shape)

X1train=X1train/255
X1test=X1test/255

X1train=X1train[:10000]
y1train=y1train[:10000]
X1test=X1test[:1000]
y1test=y1test[:1000]

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

    return  model


model=get_model()
print(model.summary())

#print(model.evaluate(Xtrain,ytrain))

checkpoint_path="model_checkpoint/checkpoint"
checkpoint=ModelCheckpoint(filepath=checkpoint_path,save_freq="epoch",save_weights_only=True,verbose=1)

#hist = model.fit(x=X1train,y=y1train,epochs=3,callbacks=[checkpoint])
#os.system("ls -lh")

model1=get_model()

model1.load_weights(checkpoint_path)


print(model1.evaluate(X1test,y1test))