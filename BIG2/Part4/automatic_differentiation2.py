import tensorflow as tf
import matplotlib.pyplot as plt

def MakeNoisyData(m,b,n=20):
    x=tf.random.uniform(shape=(n,))
    print(x.shape)
    noise= tf.random.normal(shape=(len(x),))
    print(noise.shape)
    y=m*x+b+noise
    return x,y

m=1
b=2
xtrain,ytrain=MakeNoisyData(m,b)
print(xtrain.shape,ytrain.shape)
#plt.figure(figsize=(20,20))
#plt.scatter(xtrain,ytrain)

from tensorflow.keras.layers import Layer


class Linearlayer(Layer):

    def __init__(self):
        super(Linearlayer, self).__init__()
        self.m=self.add_weight(shape=(1,),initializer="random_normal")
        self.b=self.add_weight(shape=(1,),initializer="zeros")

    def call(self, inputs, **kwargs):

        return self.m*inputs+self.b

linear_regression=Linearlayer()
#print(linear_regression(xtrain))
#print(linear_regression.weights[0].shape)

def SquaredError(y_pred,y_true):
    return tf.reduce_mean(tf.square(y_pred-y_true))

starting_loss = SquaredError(linear_regression(xtrain),ytrain)
print(starting_loss.numpy())

learning_rate=0.05
steps=50

for i in range(steps):

    with tf.GradientTape() as tape:
        predictions = linear_regression(xtrain)
        loss = SquaredError(predictions,ytrain)

    gradient = tape.gradient(loss,linear_regression.trainable_variables)
    linear_regression.m.assign_sub(learning_rate*gradient[0])
    linear_regression.b.assign_sub(learning_rate*gradient[1])

    print(f"steps {i} : loss -> {loss.numpy()}")


latest_pred = linear_regression(xtrain)

plt.scatter(xtrain,ytrain)
plt.plot(xtrain,latest_pred)
plt.show()
