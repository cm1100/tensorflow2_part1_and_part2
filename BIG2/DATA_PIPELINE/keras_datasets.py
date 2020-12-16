from tensorflow.keras.datasets import imdb

(Xtrain,ytrain),(Xtest,ytest)=imdb.load_data()

print(Xtrain.shape,ytrain.shape)


(Xtrain,ytrain),(Xtest,ytest)= imdb.load_data(skip_top=50,oov_char=2)

print(Xtrain.shape,ytrain.shape)




