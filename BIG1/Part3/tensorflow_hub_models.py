import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import numpy as np


model_url="https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"

model=Sequential([hub.KerasLayer(model_url)])

model.build(input_shape=[None,160,160,3])

print(model.summary())


def get_top_5(image):
    x=img_to_array(image)
    x=x[np.newaxis,...]
    pred = model.predict(x)

    return pred

image = load_img("images/Strawberry-Tuxedo-Cake-4-768x1024.jpg", target_size=(160, 160))

pred = get_top_5(image)

with open("data/imagenet_categories.txt") as f:

    categories = f.read().splitlines()

top= np.argsort(-pred[0])## minus sign is to get maximum first
print(type(top))
print(top)
for i in range(5):
    print(categories[top[i]])

