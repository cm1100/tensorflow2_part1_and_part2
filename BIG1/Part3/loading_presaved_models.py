from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import  preprocess_input,decode_predictions
import numpy as np
import matplotlib.pyplot as plt

model= ResNet50(weights="imagenet")
print(model.summary())

img = load_img("images/Strawberry-Tuxedo-Cake-4-768x1024.jpg", target_size=(224, 224))
print(img)
plt.imshow(img)
plt.show()
img = img_to_array(img)

img = preprocess_input(img[np.newaxis,...])
predictions = model.predict(img)
pred = decode_predictions(predictions,top=5)

print(pred)