import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


train_path = "data/flowers-recognition-split/train"
test_path="data/flowers-recognition-split/val"


datagenerator = ImageDataGenerator(rescale=1/255)
classes=["daisy","dandelion","rose","sunflower","tulip"]


train_generatoe=datagenerator.flow_from_directory(train_path,batch_size=64,classes=classes,target_size=(16,16))

test_generatoe=datagenerator.flow_from_directory(test_path,batch_size=64,classes=classes,target_size=(16,16))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Input,Flatten

model = Sequential([
    Conv2D(8,(32,32),padding="same",activation="relu",input_shape=(16,16,3)),
    MaxPooling2D((4,4)),
    Conv2D(8,(8,8),padding="same",activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(4,(4,4),padding="same",activation="relu"),
    Flatten(),
    Dense(16,activation="relu"),
    Dense(8,activation="relu"),
    Dense(5,activation="relu")
])

optim = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optim,loss="categorical_crossentropy",metrics=["accuracy"])
print(model.summary())

steps_per_epoch = train_generatoe.n//train_generatoe.batch_size
val_steps = test_generatoe.n//test_generatoe.batch_size

print(steps_per_epoch,val_steps)

hist = model.fit(train_generatoe,steps_per_epoch=steps_per_epoch,epochs=5)


hist1= model.evaluate(test_generatoe,steps=val_steps)
print(hist1)

