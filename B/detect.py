import math
import sys
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from numpy import array
import torch
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
import seaborn as sns

import absl.logging                             #suppress keras save warning
absl.logging.set_verbosity(absl.logging.ERROR)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

n =5

RANDOM_SEED = 42
THRESHOLD=0.5
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

File=""
mname=""

for i in range(len(sys.argv)-1):
    if sys.argv[i] == "-d":
        File=sys.argv[i+1]
    elif sys.argv[i] == "-n":
        n=int(sys.argv[i+1])
    elif sys.argv[i] == "-mn":
        mname=sys.argv[i+1]
    elif sys.argv[i] == "-mae":
        THRESHOLD=float(sys.argv[i+1])

if File=="":
    print("Error,expected dataset name after -d ")
    exit()

X1=pd.read_csv(File, header=None, sep='\t',index_col=0)

train1 = X1[0:math.floor(0.8*len(X1))]
test1 = X1[math.floor(0.8*len(X1)):len(X1)]


df=X1.T
sc=[]
set_scaled1=[]
for stc in range(len(df.T)):
  sc.append(MinMaxScaler(feature_range = (0,1)))
  set_scaled1.append(sc[stc].fit_transform(df.T.iloc[stc].values.reshape(-1,1)))
set_scaled1=array(set_scaled1)
set_scaled1=set_scaled1.transpose(2,0,1).reshape(-1,set_scaled1.shape[1])


train = set_scaled1[0:math.floor(0.8*len(set_scaled1))]
test = set_scaled1[math.floor(0.8*len(set_scaled1)):len(set_scaled1)]


if n > len(train):
  n=len(train)


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs, dtype=np.float), np.array(ys, dtype=np.float)

TIME_STEPS = 30

X_test = [0] * len(test)
y_test = [0] * len(test)
for i in range(len(test)):
  X_test[i], y_test[i] = create_dataset(test[i],test[i],TIME_STEPS)
  X_test[i] = np.reshape(X_test[i], (X_test[i].shape[0], X_test[i].shape[1], 1))

if mname =="":
    Xs=[]
    ys=[]
    for i in range(n):
      for j in range(len(train[i]) - TIME_STEPS):
          v = train[i][j:(j + TIME_STEPS)]
          Xs.append(v)
          ys.append(train[i][j + TIME_STEPS])
    X_train, y_train = np.array(Xs, dtype=np.float),np.array(ys, dtype=np.float)
    X_train= np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=64, 
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')


    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        shuffle=False
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    model.save('models/model_sum'+str(n))
    mname='models/model_sum'+str(n)

model=keras.models.load_model(mname)
tms = []
for i in range(len(test)):
  X_test_pred = model.predict(X_test[i])
  test_mae_loss = np.mean(np.abs(X_test_pred - X_test[i]), axis=1) 
  test_mae_loss=sc[i+len(train1)].inverse_transform(test_mae_loss)
  tms.append(test_mae_loss)

test_score_df = [0] * len(test1)
for i in range(len(test1)):
  test_score_df[i] =  pd.DataFrame()

  tms[i] = np.squeeze(tms[i])
  test_score_df[i]['loss'] = tms[i]
  test_score_df[i]['threshold'] = THRESHOLD
  test_score_df[i]['anomaly'] = test_score_df[i].loss > test_score_df[i].threshold
  test_score_df[i]['close'] = test1.iloc[i]

  if i<n:
    plt.plot(test_score_df[i].index, test_score_df[i].loss, label='loss')
    plt.plot(test_score_df[i].index, test_score_df[i].threshold, label='threshold')
    plt.xticks(rotation=25)
    plt.legend();
    anomalies = test_score_df[i][test_score_df[i].anomaly == True]
    print(sum(anomalies.loss)/len(anomalies))
    plt.plot(
        test1.iloc[i].index,
        test1.iloc[i].values,
        label='close'
    );
    sns.scatterplot(
      anomalies.index,
      anomalies.close,
      color=sns.color_palette()[3],
      s=52,
      label='anomaly'
    )
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()
