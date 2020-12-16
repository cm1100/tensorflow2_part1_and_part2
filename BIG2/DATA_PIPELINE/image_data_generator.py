import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(Xtrain,ytrain),(Xtest,ytest)=cifar10.load_data()

num_classes=10

print(ytest.shape)

ytrain=tf.keras.utils.to_categorical(ytrain,num_classes)
ytest=tf.keras.utils.to_categorical(ytest,num_classes)

def get_generator(features,labels,batch_size=1):

    for n in range(len(features)//batch_size):
        yield (features[n*batch_size:(n+1)*batch_size],labels[n*batch_size:(n+1)*batch_size])


x,y = next(get_generator(Xtrain,ytrain))
print(x.shape,y.shape)

train_generator= get_generator(Xtrain,ytrain,batch_size=10)

def monochrome(x):
    def func_bw(a):
        average_color=np.mean(a)
        return [average_color,average_color,average_color]
    x=np.apply_along_axis(func_bw,-1,x)
    return x

image_generator =ImageDataGenerator(preprocessing_function=monochrome,rotation_range=180,rescale=1/255.0)

image_generator.fit(Xtrain)

image_generator_iterable =image_generator.flow(Xtrain,ytrain,batch_size=10,shuffle=False)

image,label=next(image_generator_iterable)

print(image.shape,label.shape)