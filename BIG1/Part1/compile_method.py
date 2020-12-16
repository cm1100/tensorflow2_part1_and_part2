import tensorflow as tf
import  numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([1,2,3,4,5,6,7])
y=np.log(x)
print(x,y)
#x=x.reshape(-1,len(x))
#y=y.reshape(-1,len(y))


model=Sequential([
    Dense(64,activation="tanh",input_shape=(1,)),
    Dense(1)
])
''''
model.compile(optimizer="sgd",#"adam" "rmsprop" "adadelta"  
              loss="binary_crossentropy" # "mean_squared_error" " categorial_cross_entropy"
              ,metrics=["accuracy","mae"]) #stocahstic gradient descent # mae -> mean absolute error'''

# every keyword in the above method is pointing towards a object of the class in keras

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True),
              loss="mean_squared_error",#if the last layer is linear you can use from_ligits=True
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7),tf.keras.metrics.MeanAbsoluteError()])

# model can be compiles in both the ways



print(model.loss)
print(model.optimizer)
print(model.optimizer)
print(model.optimizer.lr)

n=model.fit(x,y,epochs=7)
x1=np.array([10,11,12])
print(model.predict(x1))