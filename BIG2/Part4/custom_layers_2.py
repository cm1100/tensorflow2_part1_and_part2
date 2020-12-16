import tensorflow as tf
from tensorflow.keras.layers import Layer,Softmax


class MyLayer(Layer):


    def __init__(self,units,input_dims):
        super(MyLayer, self).__init__()
        self.w = self.add_weight(shape=(input_dims,units),initializer="random_normal",trainable=True)
        self.b = self.add_weight(shape=(units,),initializer="zeros")

        def call(self, inputs, **kwargs):

            return tf.matmul(inputs,self.w)+self.b


dense_layer = MyLayer(3,5)
x=tf.ones((1,5))
print(dense_layer(x).shape)


class MyLayerMean(Layer):


    def __init__(self,units,input_dims):
        super(MyLayerMean, self).__init__()
        self.w = self.add_weight(shape=(input_dims,units),initializer="random_normal",trainable=True)
        self.b = self.add_weight(shape=(units,),initializer="zeros")


        self.sum_activations=tf.Variable(initial_value=tf.zeros((units,)),trainable=False)
        self.number_call =tf.Variable(initial_value=0,trainable=False)

    def call(self, inputs, **kwargs):
        activations = tf.matmul(inputs,self.w)
        self.sum_activations.assign_add(tf.reduce_sum(activations,axis=0))
        self.number_call.assign_add(inputs.shape[0])

        return activations,self.sum_activations/tf.cast(self.number_call,tf.float32)



dense_layer1=MyLayerMean(3,5)
a=tf.ones((4,5))
xm,y= dense_layer1(a)
print(xm.shape,y.shape)
print(xm)


class Dropout(Layer):

    def __init__(self,rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, **kwargs):
        return tf.nn.dropout(inputs,rate=self.rate,)


class MyModel(tf.keras.Model):

    def __init__(self,units_1,input_dims_1,units_2,units_3):
        super(MyModel, self).__init__()

        self.layer1 = MyLayer(units_1,input_dims_1)
        self.dropout=Dropout(0.5)
        self.layer2= MyLayer(units_2,units_1)
        self.dropout2=Dropout(0.5)
        self.layer3=MyLayer(units_3,units_2)
        self.softmax=Softmax()


    def call(self, inputs, training=None, mask=None):
        x= self.layer1(inputs)
        x=tf.nn.relu(x)
        x=self.dropout(x)
        x=self.layer2(x)
        x=tf.nn.relu(x)
        x=self.dropout2(x)
        x=self.layer3(x)
        out= self.softmax(x)

        return out


model = MyModel(64,10000,64,46)
print(model(tf.ones((1,10000))).shape)

print(model.summary())
