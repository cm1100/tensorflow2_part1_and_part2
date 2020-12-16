import tensorflow as tf
import numpy as np


strings =tf.Variable(["hello world"],tf.string)
floats=tf.Variable([6.98,78.65],dtype=tf.float64)
complexs = tf.Variable([25.9-7.39j,1.23-4.91j],tf.complex)

print(complexs)

matr= tf.Variable(tf.constant(2.45,shape=(3,3)))
print(matr)

#add and subtract
floats.assign_add(tf.Variable([4.09,5.76],dtype=tf.float64))
print(floats)
#floats.assign_add(1)


x=tf.constant([[1,2,3],[4,5,6],[7,8,9]],tf.float64)
print(x.shape)

coeff =np.arange(16)
print(coeff)
shape=[8,2]
shape1=[4,4]
shape3=[2,2,2,2]
tf1 = tf.constant(coeff,shape=shape,dtype=tf.float64)
tf2=tf.constant(coeff,shape=shape1,dtype=tf.float64)
tf3=tf.constant(coeff,shape=shape3,dtype=tf.float64)

print(tf1)
print(tf2)
print(tf3)

tf1 = tf.reshape(tf1,[2,8])
print(tf1)

print(tf.eye(3))


#tf.concat

t1 = tf.constant(np.arange(784),shape=[28,28],dtype=tf.int64)
print(t1.shape)

t2=tf.expand_dims(t1,0)
t3=tf.expand_dims(t1,1)
t4=tf.expand_dims(t1,2)

print(t2.shape,t3.shape,t4.shape)

#squeezing dimension

t2 = tf.squeeze(t2,0)
t3=tf.squeeze(t3,1)
t4=tf.squeeze(t4,2)
print(t2.shape,t3.shape,t4.shape)

c=tf.constant(np.arange(4),shape=[2,2])
d=tf.constant(np.arange(4),shape=[2,2])
print(c,d)

e=tf.matmul(c,d)
print(e)


#create a tensor with samples from a random distribution

tn = tf.random.normal(shape=[2,2],mean=0,stddev=1)
print(tn)


#create a tensor with the samples from a uniform distribution

tn1 = tf.random.uniform(shape=[2,1],minval=0,maxval=10,dtype=tf.int64)
print(tn1)

#create a tensor with the samples from the possoin distribution
tp = tf.random.poisson([2,2],5)
print(tp)