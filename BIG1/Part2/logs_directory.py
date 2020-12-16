import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np




dataset = load_diabetes()

X=dataset["data"]
y=dataset["target"]

y=(y-np.mean(y))/np.std(y)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
print(Xtrain.shape,Xtest.shape,ytrain.shape,ytest.shape)

model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(Xtrain.shape[1],)),
    Dense(64,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])


print(model.summary())

model.compile(optimizer="adam",loss="mse",metrics=["mae"])


class LossAndMetricLoss(Callback):

    def on_train_batch_end(self, batch, logs=None):
        if batch%2==0:
            print(f"/n after batch {batch} : loss is {logs['loss']}")

    def on_test_batch_end(self, batch, logs=None):
        print(f"/n after batch {batch} : loss is {logs['loss']}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"/n epoch {epoch} : average loss is {logs['loss']} , mean absolute error is {logs['mae']}")
        if epoch==1:
            print(logs.keys())

    def on_predict_batch_end(self, batch, logs=None):
        print(f"/n finished predicton on batch {batch} ")


his = model.fit(Xtrain,ytrain,epochs=20,callbacks=[LossAndMetricLoss()],verbose=False)

model_eval = model.evaluate(Xtest,ytest,callbacks=[LossAndMetricLoss()],verbose=False)

model_pred = model.predict(Xtest,callbacks=[LossAndMetricLoss()],verbose=False)

