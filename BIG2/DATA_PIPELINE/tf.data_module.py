import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np


x=np.zeros((100,10,2,2))

dataset1=tf.data.Dataset.from_tensor_slices(x)

print(dataset1)
print(dataset1.element_spec)

x2=[np.zeros((10,1)),np.zeros((10,1)),np.zeros((10,1))]
#print(x2)

dataset2 = tf.data.Dataset.from_tensor_slices(x2)

#for x in dataset2:
   # print(x)

dataset_zipped = tf.data.Dataset.zip((dataset1,dataset2))

print(dataset_zipped.element_spec)


mnist = tf.keras.datasets.mnist.load_data()
(Xtrain,ytrain),(Xtest,ytest)=mnist
print(Xtrain.shape,ytrain.shape)

mnist_dataset = tf.data.Dataset.from_tensor_slices((Xtrain,ytrain))
print(mnist_dataset.element_spec)

for x in mnist_dataset.take(2):
    print(x[0].shape,x[1].shape)


text_files = sorted([f.path for f in os.scandir("data/shakespeare")])

print(text_files)
contents=[]
with open(text_files[0],"r") as f:
    contents=[f.readline() for i in range(5)]

for line in contents:
    print(line)


shakespearw_df = tf.data.TextLineDataset(text_files)

print(shakespearw_df.element_spec)


for x in shakespearw_df.take(2):
    print(x)

text_file_datset=tf.data.Dataset.from_tensor_slices(text_files)
files=[file for file in text_file_datset]
print(files)


interleaved_shakespeare_dataset=text_file_datset.interleave(tf.data.TextLineDataset,cycle_length=9)
print(interleaved_shakespeare_dataset)

for m in interleaved_shakespeare_dataset.take(2):
    print(m)