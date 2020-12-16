import tensorflow as tf


# @tf.function

# just use it in the starting of the code where you want to make graphs and have the most computational load


import tensorflow as tf
from tensorflow.keras.datasets import reuters
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding,GRU,Bidirectional,Dense


(Xtrain,ytrain),(Xtest,ytest)=reuters.load_data(num_words=10000)

print(Xtest.shape,ytest.shape)



class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']


from tensorflow.keras.preprocessing.sequence import pad_sequences


padded_Xtrain = pad_sequences(Xtrain,maxlen=100,truncating="post")
padded_Xtest = pad_sequences(Xtest,maxlen=100,truncating="post")

Xtrain,Xval,ytrain,yval=train_test_split(padded_Xtrain,ytrain,test_size=0.3)

print(Xtrain.shape,Xval.shape)

dataset = tf.data.Dataset.from_tensor_slices((Xtrain,ytrain))
dataset  = dataset.batch(32)
dataset_val = tf.data.Dataset.from_tensor_slices((Xval,yval))
dataset_val=dataset_val.shuffle(500)
dataset_val= dataset_val.batch(32)

dataset_test = tf.data.Dataset.from_tensor_slices((padded_Xtest,ytest))
dataset_test = dataset_test.batch(32)

class RNNModel(Model):

   def __init__(self,units_1,units_2,num_classes,**kwargs):
      super(RNNModel, self).__init__()
      self.embedding = Embedding(input_dim=10000,output_dim=16,input_length=100)
      self.gru1= Bidirectional(GRU(units_1,return_sequences=True),merge_mode="sum")
      self.gru2=GRU(units_2)
      self.dense=Dense(num_classes,activation="softmax")

   def call(self, inputs, training=None, mask=None):
      x = self.embedding(inputs)
      x=self.gru1(x)
      x=self.gru2(x)
      out= self.dense(x)
      return out


model = RNNModel(32,16,46,name="rnn_model")
optimizers = tf.keras.optimizers.SGD(learning_rate=0.005,momentum=0.9,nesterov=True)

loss =tf.keras.losses.SparseCategoricalCrossentropy()


@tf.function
def grad(model,input,targets,loss):

   with tf.GradientTape() as tape:
      pred = model(input)
      loss_value=loss(targets,pred)

   return pred,loss_value,tape.gradient(loss_value,model.trainable_variables)

train_loss_results=[]
train_roc_loss_results=[]
val_loss_results=[]
val_roc_loss_results=[]


from tensorflow.keras.utils import to_categorical

num_epochs=10
val_steps=10


for epoch in range(num_epochs):
   train_epoch_loss_avg =tf.keras.metrics.Mean()
   train_epoch_roc_loss_avg=tf.keras.metrics.AUC(curve="ROC")

   val_epoch_loss_avg = tf.keras.metrics.Mean()
   val_epoch_loss_roc_avg = tf.keras.metrics.AUC(curve="ROC")
   nm=0


   for x,y in dataset:

      model_pred,loss_value,grads=grad(model,x,y,loss)

      optimizers.apply_gradients(zip(grads,model.trainable_variables))

      train_epoch_loss_avg(loss_value)
      train_epoch_roc_loss_avg(to_categorical(y,num_classes=46),model_pred)
      nm+=1

      #print(f"hi there  {nm}")



   for x,y in dataset_val:

       model_pred= model(x)


       val_epoch_loss_avg(loss(y,model_pred))
       val_epoch_loss_roc_avg(to_categorical(y,num_classes=46),model_pred)
       #print("yo man")

   train_loss_results.append(train_epoch_loss_avg.result().numpy())
   train_roc_loss_results.append(train_epoch_roc_loss_avg.result().numpy())

   val_loss_results.append(val_epoch_loss_avg.result().numpy())
   val_roc_loss_results.append(val_epoch_loss_roc_avg.result().numpy())


   print("Epoch {:03d}: Training loss: {:.3f}, ROC AUC: {:.3%}".format(epoch, train_epoch_loss_avg.result(),
                                                                        train_epoch_roc_loss_avg.result()))
   print("Validation loss: {:.3f}, ROC AUC {:.3%}".format(val_epoch_loss_avg.result(),
                                                                         val_epoch_loss_roc_avg.result()))




print(tf.autograph.to_code(grad.python_function))