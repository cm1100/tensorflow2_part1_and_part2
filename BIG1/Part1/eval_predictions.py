import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D

model = Sequential([
    Conv2D(16,(3,3),input_shape=(28,28,1),activation="relu"),
    MaxPooling2D(3,3),
    Flatten(),
    Dense(10,activation="softmax")
])
opt=tf.keras.optimizers.Adam(learning_rate=0.05)
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy","mae"]
)

#history =model.fit(X_train,y_train,epochs=10,batch_size=16)

#X_train -> (num_samples,num_features)
#y_train -> (num_samples,num_classes)

fashion =tf.keras.datasets.fashion_mnist
(X_train,y_train),(X_test,y_test)=fashion.load_data()

#print(X_train)
print(X_train[0].shape)
print(model.summary())

labels=[
    "tshirt",
    "Trouser",
    "pulover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "boot"
]
print(X_train.shape)
print(len(labels))
X_train1=X_train/255
X_test1=X_test/255

#print(X_train1[1])
#plt.imshow(X_train1[1])
#plt.show()

#model.fit(X_train,y_train,)

history = model.fit(X_train[...,np.newaxis],y_train,epochs=2,batch_size=256,verbose=2)



df = pd.DataFrame(history.history)
history1=model.evaluate(X_test[...,np.newaxis],y_test)
print(f"loss : {history1[0]}, accuracy: {history1[1]} , mae: {history1[2]}")

history2= model.predict(X_test[...,np.newaxis])
print(history2.shape)
pred=np.argmax(history2,axis=1)
print(pred.shape)
print(pred)
pd4= pd.DataFrame(history2)
print(pd4)