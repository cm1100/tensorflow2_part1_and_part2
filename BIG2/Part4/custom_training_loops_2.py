import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Softmax,Dense,Dropout
from tensorflow.keras.datasets import reuters
import numpy as np


(Xtrain,ytrain),(Xtest,ytest)=reuters.load_data(num_words=10000)

print(Xtrain.shape,ytrain.shape)

print(len(Xtrain[0]))
print(ytrain[0])


class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']


model =tf.keras.models.Sequential([
    Dense(64,activation="relu"),
    Dropout(0.5),
    Dense(64,activation="relu"),
    Dense(46,activation="softmax")
])

classes =tf.keras.datasets.reuters.get_word_index()
#print(classes)
inv_map ={value:key for key,value in classes.items()}
#print(inv_map)

print(class_names[ytrain[0]])

#print(model.summary())


#print(model.summary())




def bag_of_words(text_samples,elements=10000):
    output = np.zeros((len(text_samples),elements))
    for i,word in enumerate(text_samples):
        output[i][word]=1

    return output


Xtrain=bag_of_words(Xtrain)
Xtest = bag_of_words(Xtest)

print(Xtrain.shape)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

def loss(model,x,y,wd):
    kernel_variables=[]
    for l in model.layers:
        for w in l.weights:
            if "kernel" in w.name:
                kernel_variables.append(w)
    wd_penalty = wd*tf.reduce_sum([tf.reduce_sum(tf.square(k)) for k in kernel_variables])#regularization
    y_pred=model(x)
    return loss_object(y,y_pred)+wd_penalty

optimizer =tf.optimizers.Adam(lr=0.001)

def grad(model,inputs,targets,wd):
    with tf.GradientTape() as tape:
        loss_value =loss(model,inputs,targets,wd)

    return loss_value,tape.gradient(loss_value,model.trainable_variables)

from tensorflow.keras.utils import to_categorical

start_time = time.time()

train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain,ytrain))
train_dataset=train_dataset.batch(32)

train_loss_results=[]
train_accuracy_results=[]

num_epochs=10
weight_decay=0.005

for epoch in range(num_epochs):
    epoch_loss_average = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for x,y in train_dataset:
        loss_value,grads=grad(model,x,y,weight_decay)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))


        epoch_loss_average(loss_value)
        epoch_accuracy(to_categorical(y),model(x))

    train_loss_results.append(epoch_loss_average.result())
    train_accuracy_results.append(epoch_accuracy.result())

    print(f" loss : {epoch_loss_average.result()}  , Accuracy : {epoch_accuracy.result()}")



fig,axis=plt.subplots(2)
for i in range(2):
    axis[0].plot(train_loss_results)

    axis[1].plot(train_accuracy_results)
plt.show()










