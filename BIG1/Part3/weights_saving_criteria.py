import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorflow.keras.callbacks import Callback



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
class att(Callback):

    #def set_model(self, model):
        #self.model=model

    def on_epoch_begin(self, epoch, logs=None):
        print(f"epoch is {epoch}")
        if epoch==5:
            print("training stopped")
            self.model.stop_training=True

model =get_model()
print(model.summary())

checkpoint_path="train_run1/checkpoint_{epoch}__{accuracy}"



obj=att()

checkpoint_path1="best_save/checkpoint_{val_accuracy}"
check = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,save_freq=300,verbose=1)

checkpoint_path1="best_save/checkpoint_{val_accuracy}"
check_new=ModelCheckpoint(filepath=checkpoint_path1,save_weights_only=True,save_freq="epoch",monitor="val_accuracy",mode="max",verbose=1,save_best_only=True)
hist= model.fit(X1train,y1train,batch_size=10,callbacks=[obj,check_new],epochs=10,validation_split=0.15)