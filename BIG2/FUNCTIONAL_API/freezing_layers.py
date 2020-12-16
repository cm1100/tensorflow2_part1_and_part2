import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model =Sequential([
    layers.Dense(input_shape=(4,),units=4,activation="relu",kernel_initializer=tf.keras.initializers.he_normal()),
    layers.Dense(2,activation="relu",bias_initializer="ones",kernel_initializer=tf.keras.initializers.glorot_uniform()),
    layers.Dense(4,activation="softmax")
])

print(model.summary())



W_0 =[e.weights[0].numpy() for e in model.layers]
b_0 =[b.bias.numpy() for b in model.layers]

Xtrain=np.random.random((100,4))
ytrain=np.random.random((100,4))

Xtest=np.random.random((20,4))
ytest = np.random.random((20,4))

#model.compile(optimizer="adam",loss="mae",metrics=["accuracy"])
#hist = model.fit(Xtrain,ytrain,epochs=50,verbose=False)



#print(model.layers[1].weights[0].numpy().shape)

#print(W_1[0].shape,b_1[0].shape)

'''
plt.figure(figsize=(8,8))

for i in range(3):
    delta_1=W_1[i]-W_0[i]
    print(f"difference between biases is {b_1[i]-b_0[i]}")
    axis=plt.subplot(1,3,i+1)
    plt.imshow(delta_1)
    plt.title(f"layer {i+1}")
    plt.axis("off")
plt.colorbar()
plt.show()'''

n_trainable_variables = model.trainable_variables
print(n_trainable_variables)

model.get_layer("dense").trainable=False

model.compile(optimizer="adam",loss="mse",metrics=["accuracy"])
his=model.fit(Xtrain,ytrain,epochs=50)
W_1 =[e.weights[0].numpy() for e in model.layers]
b_1 =[b.bias.numpy() for b in model.layers]

plt.figure(figsize=(8,8))

for i in range(3):
    delta_1=W_1[i]-W_0[i]
    print(f"difference between biases is {b_1[i]-b_0[i]}")
    axis=plt.subplot(1,3,i+1)
    plt.imshow(delta_1)
    plt.title(f"layer {i+1}")
    plt.axis("off")
plt.colorbar()
plt.show()