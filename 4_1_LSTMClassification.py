import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import time

#---------------------------------
#LSTM Implementation given by Keras Tensorflow Library
#Parameters are the predetermined architecture of Bi-LSTM, with exception of the 520 hidden units and number of epochs, which can be changed
#-----------------------------------

hiddenUnits = 520
totEpochs = 13

startTime=time.time()
dataMat = np.load('TestMatrices/dataMat.npy')
labelMat = np.load('TestMatrices/labelMat.npy')

#Random 80/20 split into training and test data points
x_train,x_test,y_train,y_test=train_test_split(dataMat,labelMat,train_size=0.8)

print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_test)
size = len(y_test)

model = tf.keras.models.Sequential([
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hiddenUnits, input_shape=(10,260))),
	tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print("---------Training the LSTM model-----------")
model.fit(x_train, y_train,epochs=totEpochs)

exTime = (time.time()-startTime)
print("time = "+str(exTime))

print("----------Evaluation---------")

print("---------Training Data Prediction-------------")
startTime = time.time()

out=model.evaluate(x_train, y_train, verbose=0)

score =100*out[1]
print('Score='+str(score)+'%')
exTime = (time.time()-startTime)
print("time = "+str(exTime))

print("--------Test Data Prediction-----------")
out2=model.evaluate(x_test, y_test, verbose=0)
score =100*out2[1]
print('Score='+str(score)+'%')


#Execution time measuring
exTime2 = (time.time()-startTime)
print("time = "+str(exTime2))
