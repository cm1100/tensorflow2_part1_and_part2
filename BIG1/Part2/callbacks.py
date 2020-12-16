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

class TrainingCallback(Callback):

    def on_train_begin(self, logs=None):
        print("starting training...")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"starting epoch {epoch}")



    def on_train_batch_begin(self, batch, logs=None):
        print(f"training : starting batch {batch}")

    def on_train_batch_end(self, batch, logs=None):
        print(f"finishing batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"ending epoch {epoch}")

    def on_train_end(self, logs=None):
        print(f"ending training")


class TestingCallback(Callback):

    def on_test_begin(self, logs=None):
        print("starting testing...")





    def on_test_batch_begin(self, batch, logs=None):
        print(f"testing : starting batch {batch}")

    def on_test_batch_end(self, batch, logs=None):
        print(f"finishing batch {batch}")



    def on_test_end(self, logs=None):
        print(f"ending testing")


def predictioncallback(Callback):
    pass
# same as the other two callbacks




rate=0.5
wd=0.01

model = Sequential()
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu",input_shape=(Xtrain.shape[1],)))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dropout(rate=rate))
model.add(Dense(128,kernel_regularizer=tf.keras.regularizers.l2(wd),activation="relu"))
model.add(Dense(1))


model.compile(optimizer="adam",loss="mse",metrics=["mae"])

history = model.fit(Xtrain,ytrain,callbacks=[TrainingCallback()],epochs=10)

print(history.history)


pred = model.evaluate(Xtest,ytest,callbacks=[TestingCallback()])
