import sys
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from numpy import array
import tensorflow as tf
import random

import absl.logging                             #suppress keras save warning
absl.logging.set_verbosity(absl.logging.ERROR)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

dataset=""
num_of_TS=-1
mname=""

for i in range(len(sys.argv)-1):
    if sys.argv[i] == "-d":
        dataset=sys.argv[i+1]
    elif sys.argv[i] == "-n":
        num_of_TS=int(sys.argv[i+1])
    elif sys.argv[i] == "-mn":
        mname=sys.argv[i+1]

if dataset=="":
    print("Error,expected dataset name after -d ")
    exit()
elif num_of_TS<0:
    print("Error,expected number of time series or number of graphs for already saved model after -n ")
    exit()


df=pd.read_csv(dataset, header=None, sep='\t',index_col=0)

if num_of_TS > len(df):
  num_of_TS=len(df)

n_steps =60

# df=df.iloc[:num_of_TS]
df=df.T
sc=[]
# set_scaled=sc.fit_transform(df.values)


set_scaled1=[]
for stc in range(len(df.T)):
  sc.append(MinMaxScaler(feature_range = (0,1)))
  set_scaled1.append(sc[stc].fit_transform(df.T.iloc[stc].values.reshape(-1,1)))
set_scaled1=array(set_scaled1)
set_scaled1=set_scaled1.transpose(2,0,1).reshape(-1,set_scaled1.shape[1])
set_scaled=set_scaled1[:num_of_TS]
set_scaled=set_scaled.T

temp=int(0.8 *len(set_scaled))

if mname =="":
  train=set_scaled[:temp]

  X_train = []
  y_train = []
  for j in range(len(train.T)):
      for i in range(n_steps, temp):
          X_train.append(train.T[j][i-n_steps:i])
          y_train.append(train.T[j][i])

  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  # for i in range(len(X_train)):
  #     print(X_train[i], y_train[i])



  model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 32, return_sequences = True, input_shape = (X_train.shape[1], 1),name='l1'))
  model.add(Dropout(0.1,name='d1'))# Adding a second LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 64, return_sequences = True,name='l2'))
  model.add(Dropout(0.1,name='d2'))# Adding a third LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 128,name='l3'))
  model.add(Dropout(0.1,name='d3'))# Adding the output layer
  model.add(Dense(units = 1,name='out'))

  # Compiling the RNN
  model.compile(optimizer = 'adam', loss = 'mean_squared_error')

  # Fitting the RNN to the Training set
  model.fit(X_train, y_train, epochs = 1, batch_size = 64)
  model.save('models/model_sum_'+str(num_of_TS))

  if num_of_TS!=1:
    randlist=[]
    if num_of_TS==2:
        randlist=[0,1]
    else:
        randlist=random.sample(range(0,num_of_TS),3)
    for rand in randlist:
        train=set_scaled[:temp,rand]
        X_train = []
        y_train = []
        for i in range(n_steps, temp):
            X_train.append(train.T[i-n_steps:i])
            y_train.append(train.T[i])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model2 = Sequential()#Adding the first LSTM layer and some Dropout regularisation
        model2.add(LSTM(units = 32, return_sequences = True, input_shape = (X_train.shape[1], 1),name='l1'))
        model2.add(Dropout(0.1,name='d1'))# Adding a second LSTM layer and some Dropout regularisation
        model2.add(LSTM(units = 64, return_sequences = True,name='l2'))
        model2.add(Dropout(0.1,name='d2'))# Adding a third LSTM layer and some Dropout regularisation
        model2.add(LSTM(units = 128,name='l3'))
        model2.add(Dropout(0.1,name='d3'))# Adding the output layer
        model2.add(Dense(units = 1,name='out'))

        # Compiling the RNN
        model2.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Fitting the RNN to the Training set
        model2.fit(X_train, y_train, epochs = 1, batch_size = 64)
        model2.save('models/model_seperate_'+str(rand))
  mname='models/model_sum_'+str(num_of_TS)
  num_of_TS2=10
  if num_of_TS2 > len(df)+1:
    num_of_TS2=len(df)+1
  print("Displaying "+str(num_of_TS2)+" predictions from the "+str(num_of_TS2)+" first stocks of the dataset, for the model you just trained with "+str(num_of_TS)+" stocks")
  num_of_TS=num_of_TS2
  set_scaled=set_scaled1[:num_of_TS]
  set_scaled=set_scaled.T

test=set_scaled[temp-n_steps:]


model=keras.models.load_model(mname)

for stock in range(num_of_TS):
  X_test = []

  for i in range(n_steps,len(test.T[stock])):
      X_test.append(test.T[stock][i-n_steps:i])

  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


  predicted_stock_price = model.predict(X_test)

  trainPredict_dataset_like = np.zeros(shape=(len(predicted_stock_price), len(df.columns)) ) 
  trainPredict_dataset_like[:,0] = predicted_stock_price[:,0] 


  predicted_stock_price = sc[stock].inverse_transform(trainPredict_dataset_like)[:,0]

  # Visualising the results

  plt.plot(range(len(df.iloc[temp:,stock])),df.iloc[temp:,stock],color = "red", label = "Real "+str(df.columns[stock])+" Stock Price")
  plt.plot(range(len(df.iloc[temp:,stock])),predicted_stock_price, color = "blue", label = "Predicted  "+str(df.columns[stock])+" Stock Price")
  plt.xticks(np.arange(0,len(df.iloc[temp:,stock]),50))
  plt.title('Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('Stock Price')
  plt.legend()
  plt.show()


