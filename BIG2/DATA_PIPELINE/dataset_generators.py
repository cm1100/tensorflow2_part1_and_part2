import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

headers=["season","age","diseases","trauma","surgery","fever","alcohol","smoking","sitting","output"]

fertility = pd.read_csv("data/fertility_diagnosis.txt",names=headers)

print(fertility.shape)

print(fertility.head())

fertility["output"]=fertility["output"].map(lambda x: 0.0 if x=="N" else 1.0)

print(fertility.head())

fertility=fertility.astype("float32")

fertility=fertility.sample(frac=1).reset_index(drop=True)

fertility=pd.get_dummies(fertility,prefix="season",columns=["season"]) ## to convert into one_hot vector

print(fertility.head())

fertility.columns=[col for col in fertility.columns if col!="output"]+["output"]
print(fertility.head())

fertility=fertility.to_numpy()

training=fertility[:70]
validation=fertility[70:]
print(training.shape)


training_features =training[:,:-1]
training_labels=training[:,-1]

testing_features=validation[:,:-1]
testing_labels=validation[:,-1]

def get_generator(features,labels,batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield (features[n*batch_size:(n+1)*batch_size],labels[n*batch_size:(n+1)*batch_size])


train_generator = get_generator(features=training_features,labels=training_labels,batch_size=10)

print(next(train_generator))

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Input,BatchNormalization

input_shape=(12,)
output_shape=(1,)

input1=Input(input_shape)
batch1=BatchNormalization(momentum=0.8)(input1)
dense_1=Dense(100,activation="relu")(batch1)
batch_2=BatchNormalization(momentum=0.8)(dense_1)
output = Dense(1,activation="sigmoid")(batch_2)

model=Model(inputs=input1,outputs=output)

print(model.summary())

optimizer = tf.keras.optimizers.Adam(lr=1e-2)

model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])

batch_size=5

train_steps=len(training)//batch_size

epochs=3


for epoch in range(epochs):
    train_generator=get_generator(features=training_features,labels=training_labels,batch_size=5)
    validation_generator=get_generator(features=training_features,labels=training_labels,batch_size=29)
    model.fit_generator(train_generator,steps_per_epoch=train_steps,validation_data=validation_generator,validation_steps=1)



def get_generator_cyclic(features,labels,batch_size):
    while True:
        for n in range(len(features)//batch_size):
            yield (features[n*batch_size:(n+1)*batch_size],labels[n*batch_size:(n+1)*batch_size])
            permuted = np.random.permutation(len(features))
            features=features[permuted]
            labels=labels[permuted]


generator=get_generator_cyclic(training_features,training_labels,batch_size)

validation=get_generator(testing_features,testing_labels,batch_size=30)

hist = model.fit(generator,steps_per_epoch=train_steps,epochs=3,validation_data=validation,validation_steps=1)


validation2 = get_generator(testing_features,testing_labels,batch_size=30)

prediction=model.predict_generator(validation2,steps=1)
print(prediction)