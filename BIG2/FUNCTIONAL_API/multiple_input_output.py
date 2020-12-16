import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import Input,layers

pd_dat = pd.read_csv("datasets/diagnosis.csv")
print(type(pd_dat))

dataset=pd_dat.values
print(dataset.shape)

Xtrain,Xtest,ytrain,ytest = train_test_split(dataset[:,:6],dataset[:,6:],test_size=0.3)
print(Xtrain.shape,Xtest.shape,ytrain.shape,ytest.shape)


temp_train,nocc_train,lump_train,up_train,mict_train,bis_train=np.transpose(Xtrain)
temp_test,nocc_test,lump_test,up_test,mict_test,bis_test = np.transpose(Xtest)
print(Xtrain.shape,up_train.shape)

inflam_train,nephr_train= np.transpose(ytrain)
inflam_test,nephr_test=np.transpose(ytest)


shape_inputs=(1,)
temperature = Input(shape=shape_inputs,name="temp")
nausea_occurence = Input(shape=shape_inputs,name="nocc")
lumbar_pain = Input(shape=shape_inputs,name="lumbp")
urine_pushing = Input(shape=shape_inputs,name="up")
micturition_pains = Input(shape=shape_inputs,name="mict")
bis = Input(shape=shape_inputs,name="bis")

list_inputs=[temperature,nausea_occurence,lumbar_pain,urine_pushing,micturition_pains,bis]

x=layers.concatenate(list_inputs)
print(x.shape)

inflammation_pred = layers.Dense(1,activation="sigmoid",name="inflam")(x)
nephritis_pred = layers.Dense(1,activation="sigmoid",name="nephr")(x)

list_outputs=[inflammation_pred,nephritis_pred]

model=tf.keras.Model(inputs=list_inputs,outputs=list_outputs)
print(model.summary())

#tf.keras.utils.print_model(model,show_shapes=True)


model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
              loss={"inflam":"binary_crossentropy","nephr":"binary_crossentropy"},
              metrics={"inflam":["accuracy"],"nephr":["mae"]},
              loss_weights=(1,0.2))

input_trains={"temp":temp_train,"nocc":nocc_train,"lumbp":lump_train,"up":up_train,"mict":mict_train,"bis":bis_train}
output_trains={"inflam":inflam_train,"nephr":nephr_train}

history = model.fit(input_trains,output_trains,epochs=1000,batch_size=128,verbose=False)

print(history.history.keys())

acc_keys=["inflam_accuracy","nephr_mae"]
loss_keys=["inflam_loss","nephr_loss","loss"]

print(history.history.items())

for k,v in history.history.items():
    if k in acc_keys:
        plt.figure(1)
        plt.plot(v)
    else:
        plt.figure(2)
        plt.plot(v)

plt.show()

test_inputs={"temp":temp_test,"nocc":nocc_test,"lumbp":lump_test,"up":up_test,"mict":mict_test,"bis":bis_test}
test_outputs={"inflam":inflam_test,"nephr":nephr_test}

eval=model.evaluate(x=test_inputs,y=test_outputs)
print(eval)

