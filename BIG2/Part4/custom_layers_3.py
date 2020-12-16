import tensorflow as tf
from tensorflow.keras.layers import Layer

class MyLayer(Layer):

    def __init__(self,units,**kwargs):
        super(MyLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],self.units),initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,),initializer="zeros")

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs,self.w)+self.b

dense_layer = MyLayer(3)

x= tf.ones((3,5))
print(tf.transpose(x))
print(dense_layer(x).shape)

class MyModel(tf.keras.Model):

    def __init__(self,units_1,units_2,**kwargs):
        super(MyModel, self).__init__()
        self.layer1 = MyLayer(units_1)
        self.layer2=MyLayer(units_2)

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x=tf.nn.relu(x)
        x=self.layer2(x)

        return tf.nn.softmax(x)

model = MyModel(32,10)

inputs=tf.ones((10,100))
print(model(inputs).shape)
print(model.summary())
