import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D

a=Input(shape=(128,128,3),name="input_a")
b=Input(shape=(64,64,3),name="input_b")


conv = Conv2D(32,6,padding="same")
conv_out_a=conv(a)
print(type(conv_out_a))

conv_out_b= conv(b)
print(type(conv_out_b))

''' here the conv layer has multiple input and output nodes so we have to do indexing to 
acess those inputs and outputs'''

print(conv.get_input_shape_at(0),conv.get_input_shape_at(1))

print(conv.get_output_at(0),conv.get_output_at(1))

print(conv.get_input_at(0).name,conv.get_input_at(1).name)
