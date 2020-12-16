import tensorflow as tf

t1= tf.random.uniform([3,2])

tf1=tf.data.Dataset.from_tensor_slices(t1)
tf2=tf.data.Dataset.from_tensors(t1)

print(tf1,tf2)

tensor1=tf.random.uniform([10,2,2])
tensor2=tf.random.uniform([10,1])
tensor3=tf.random.uniform([9,2,2])

# dataset = tf.data.Dataset.from_tensor_slices((tensor1,tensor3))

dataset = tf.data.Dataset.from_tensors((tensor1,tensor3))
for x in dataset:
    #print(x)
    break

import pandas as pd

r1 = pd.read_csv("balloon_dataset.csv")

r1 = dict(r1)
print(r1.keys())

data=tf.data.Dataset.from_tensor_slices(r1)
for x in data:
    print(x)