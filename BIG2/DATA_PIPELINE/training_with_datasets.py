import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("data/Bank/bank-full.csv",delimiter=";")

print(data.shape)

print(set(data["marital"]))

features=["age","job","marital","education","default","balance","housing","loan","contact","campaign","pdays","poutcome"]
output=["y"]

bank_dataframe = data.filter(features+output)

print(bank_dataframe.shape)

categorical_features = ["default","housing","job","loan","education","contact","poutcome"]

obj=LabelEncoder()
for fea in categorical_features:

    bank_dataframe[fea]=obj.fit_transform(bank_dataframe[fea])

print(bank_dataframe.head())

bank_dataframe=bank_dataframe.sample(frac=1).reset_index(drop=True)
bank_dataset = tf.data.Dataset.from_tensor_slices(dict(bank_dataframe))


def check_divorced(bank_dataset):
    for x in bank_dataset:
        if x["marital"]!="divorced":
            print("found")
            return

    print("not")

check_divorced(bank_dataset)
bank_dataset=bank_dataset.filter(lambda x : tf.equal(x['marital'],tf.constant(['divorced']))[0])
#check_divorced(bank_dataset)

def map_label(x):
    x["y"]=0 if (x["y"]==tf.constant(["no"],dtype=tf.string)) else 1
    return x

bank_dataset=bank_dataset.map(map_label)

print(bank_dataset.element_spec)

bank_dataset=bank_dataset.map(lambda x: {key:val for key,val in x.items() if key!="marital"})
print(bank_dataset.element_spec)


def map_feature_label(bank_dataset):
    features = [[bank_dataset["age"]],[bank_dataset["balance"]],[bank_dataset["campaign"]],[bank_dataset["contact"]],
                [bank_dataset["default"]],[bank_dataset["education"]],[bank_dataset["housing"]],[bank_dataset["job"]],
                [bank_dataset["loan"]],[bank_dataset["pdays"]],[bank_dataset["poutcome"]]]

    return (tf.concat(features,axis=0),bank_dataset["y"])


bank_dataset=bank_dataset.map(map_feature_label)
print(bank_dataset.element_spec)
dataset_len=0
for _ in bank_dataset:
    dataset_len+=1


training_elements = int(dataset_len*0.7)
train_dataset=bank_dataset.take(training_elements)
validation_dataset = bank_dataset.skip(training_elements)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Concatenate,BatchNormalization,Input

model = Sequential([
    Input(shape=(11,)),
    BatchNormalization(momentum=0.8),
    Dense(400,activation="relu"),
    BatchNormalization(momentum=0.8),
    Dense(400,activation="relu"),
    BatchNormalization(momentum=0.8),
    Dense(1,activation="sigmoid")
])

optim = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=optim,loss="binary_crossentropy",metrics=["accuracy"])
print(model.summary())

training_datset=train_dataset.batch(20,drop_remainder=True)
validation_dataset=validation_dataset.batch(100)

training_datset=training_datset.shuffle(1000)
his=model.fit(training_datset,validation_data=validation_dataset,epochs=10)

plt.plot(his.epoch,his.history["accuracy"],label="training")
plt.plot(his.epoch,his.history["val_accuracy"],label="validation")
plt.show()