import tensorflow as tf
from tensorflow.keras.layers import Dense,Softmax
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD


def MakeNoisyData(m,b,n=20):
    x=tf.random.uniform(shape=(n,))
    print(x.shape)
    noise= tf.random.normal(shape=(len(x),))
    print(noise.shape)
    y=m*x+b+noise
    return x,y

m=1;b=2
Xtrain,ytrain =MakeNoisyData(m,b)


model =tf.keras.Sequential([
    Dense(12,activation="relu",input_shape=(1,)),
    Dense(12,activation="relu"),
    Dense(1)
])

loss =MeanSquaredError()
print(model.summary())

optimizer = SGD(learning_rate=0.05,momentum=0.9)

for i in range(50):
    with tf.GradientTape() as tape:
        current_loss = loss(ytrain,model(Xtrain))
        grad= tape.gradient(current_loss,model.trainable_variables)

    optimizer.apply_gradients(zip(grad,model.trainable_variables))
    print(current_loss)
