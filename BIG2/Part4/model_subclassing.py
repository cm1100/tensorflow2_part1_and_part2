import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Dropout,Softmax,concatenate,Input

class MyModel(Model):
     
    def __init__(self,**kwargs):
        
        super(MyModel, self).__init__()

        self.dense1=Dense(64,activation="relu")
        self.dense2=Dense(10)
        #self.drop=Dropout(0.5)
        self.dense3=Dense(5)
        self.softmax=Softmax()


    def call(self, inputs, training=True, mask=None):

        x=self.dense1(inputs)
        #x=self.drop(x,training)
        y=self.dense2(x)
        y2=self.dense3(x)
        conct=concatenate([y,y2])
        return self.softmax(conct)


model =MyModel()

print(tf.random.uniform([2,10]))
out=model(tf.random.uniform([1,10]))
print(out)
print(model.summary())
print(out.shape)