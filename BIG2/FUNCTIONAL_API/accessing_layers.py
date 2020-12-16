from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


vgg_model=load_model("models/vgg19")


vgg_input=vgg_model.input
print(vgg_input)
vgg_layers = vgg_model.layers
#print(vgg_model.summary())

layer_outputs = [layer.output for layer in vgg_layers]
features = Model(inputs=vgg_input,outputs=layer_outputs)

print(features.summary())

img = np.random.random((1,224,224,3)).astype("float32")
extracted_features = features(img)
print(len(extracted_features))
print(len(vgg_model.layers))

img_path="datasets/cool_cat.jpg"
img = image.load_img(img_path,target_size=(224,224,3))
x=image.img_to_array(img)
print(x.shape)
print(type(x))
x=np.expand_dims(x,axis=0)
print(x.shape)
print(type(x))
x=preprocess_input(x)

extracted_features=features(x)

## VISUALIZING input channels
'''
f1 = extracted_features[0]
image1=f1[0,:,:,:]
#plt.imshow(image)
plt.figure(figsize=(15,15))

for i in range(3):
    ax = plt.subplot(1,3,i+1)
    plt.imshow(image1[:,:,i])
    plt.axis("off")
plt.show()'''


## visualizing first layers output

'''
f1 = extracted_features[1]
image1=f1[0,:,:,:]
#plt.imshow(image)
plt.figure(figsize=(15,15))

for i in range(16):
    ax = plt.subplot(4,4,i+1)
    plt.imshow(image1[:,:,i])
    plt.axis("off")
#plt.show()
'''


extraxted_block1_pool=Model(inputs=vgg_input,outputs=features.get_layer("block1_pool").output)
output_block1_pool =extraxted_block1_pool(x)
print(output_block1_pool.shape)

'''
f1 = output_block1_pool
image1=f1[0,:,:,:]
#plt.imshow(image)
plt.figure(figsize=(15,15))

for i in range(16):
    ax = plt.subplot(4,4,i+1)
    plt.imshow(image1[:,:,i])
    plt.axis("off")
plt.show()'''


extracted_features_block5_conv4=Model(inputs=vgg_input,outputs=features.get_layer("block5_conv4").output)
output=extracted_features_block5_conv4(x)

f1 = output
image1=f1[0,:,:,:]
#plt.imshow(image)
plt.figure(figsize=(15,15))

for i in range(16):
    ax = plt.subplot(4,4,i+1)
    plt.imshow(image1[:,:,i])
    plt.axis("off")
#plt.show()

model1= Model(inputs=vgg_input,outputs=vgg_model.get_layer("block5_conv4").output)
print(model1.summary())