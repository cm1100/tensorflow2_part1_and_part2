import tensorflow as tf
from tensorflow.keras.layers import Layer
'''
class LinearMap(Layer):

    def __init__(self,input_dims,units):
        super(LinearMap, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w=tf.Variable(initial_value=w_init(shape=(input_dims,units)))

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs,self.w)


data = tf.ones(shape=(10,3))
#print(data)
linear_layer = LinearMap(3,2)
print(linear_layer(data).shape)
print(linear_layer.weights)'''

## Shortcut code

class LinearMap(Layer):

    def __init__(self,input_dims,units):
        super(LinearMap, self).__init__()
        self.w =self.add_weight(shape=(input_dims,units),initializer="random_normal")

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs,self.w)

linea_layer = LinearMap(3,2)
data = tf.ones(shape=(10,3))
print(linea_layer(data).shape)
print(linea_layer.weights[0].shape)

class MyModel(tf.keras.Model):

    def __init__(self,hidden_units,output,**kwargs):
        super(MyModel, self).__init__()

        self.dense=tf.keras.layers.Dense(hidden_units,activation="sigmoid")
        self.linear=LinearMap(hidden_units,output)


    def call(self, inputs, training=None, mask=None):
        h = self.dense(inputs)
        return self.linear(h)


my_model=MyModel(hidden_units=64,output=12)
data=tf.ones(shape=(12,12))
out=my_model(data)
#my_model=my_model.build(input_shape=(10,))
#print(my_model.summary())
print(out.shape)
print(my_model.summary())
