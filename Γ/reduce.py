import math
import sys
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from numpy import array
import torch
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import absl.logging                             #suppress keras save warning
absl.logging.set_verbosity(absl.logging.ERROR)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

n = 350
n_steps=30
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

File=""
query_set=""
output_query_file=""
output_dataset_file=""

mname=""
for i in range(len(sys.argv)-1):
    if sys.argv[i] == "-d":
        File=sys.argv[i+1]
    elif sys.argv[i] == "-q":
        query_set=sys.argv[i+1]
    elif sys.argv[i] == "-mn":
        mname=sys.argv[i+1]
    elif sys.argv[i] == "-od":
        output_dataset_file=sys.argv[i+1]
    elif sys.argv[i] == "-oq":
        output_query_file=sys.argv[i+1]

if File=="":
    print("Error,expected dataset name after -d ")
    exit()
elif output_query_file=="":
    print("Error,expected output_query_file name after -oq ")
    exit()
elif output_dataset_file=="":
    print("Error,expected output_dataset_file name after -od ")
    exit()
elif query_set=="":
    print("Error,expected query_set name after -q ")
    exit()


X1=pd.read_csv(File, header=None, sep='\t',index_col=0)

X2=pd.read_csv(query_set, header=None, sep='\t',index_col=0)


if n > len(X1):
  n=len(X1)

df=X1.T
sc=[]
set_scaled1=[]
for stc in range(len(df.T)):
  sc.append(MinMaxScaler(feature_range = (0,1)))
  set_scaled1.append(sc[stc].fit_transform(df.T.iloc[stc].values.reshape(-1,1)))
set_scaled1=array(set_scaled1)
set_scaled1=set_scaled1.transpose(2,0,1).reshape(-1,set_scaled1.shape[1])

X1=set_scaled1


df2=X2.T
sc2=[]
set_scaled2=[]
for stc in range(len(df2.T)):
  sc2.append(MinMaxScaler(feature_range = (0,1)))
  set_scaled2.append(sc2[stc].fit_transform(df2.T.iloc[stc].values.reshape(-1,1)))
set_scaled2=array(set_scaled2)
set_scaled2=set_scaled2.transpose(2,0,1).reshape(-1,set_scaled2.shape[1])

X2=set_scaled2


window_length = 10
latent_dim = 3

def chunks(lst, n):
  new = []
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    new.append((lst[i:i + n]))
  return new

x_test = []
for i in range(len(X1)):
  x_test.append(chunks(X1[i], window_length))

x_test2 = []
for i in range(len(X2)):
  x_test2.append(chunks(X2[i], window_length))


x_test = np.array(x_test)
x_test2 = np.array(x_test2)

if mname=="":
  x_train = []
  for j in range(n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(X1[j]), window_length):
      x_train.append((X1[j][i:i + window_length]))
  x_train = np.array(x_train)
  input_window = Input(shape=(window_length,1))
  x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
  #x = BatchNormalization()(x)
  x = MaxPooling1D(2, padding="same")(x) # 5 dims
  x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
  #x = BatchNormalization()(x)
  encoded = MaxPooling1D(2, padding="same")(x) # 3 dims
  encoder = Model(input_window, encoded)

  # 3 dimensions in the encoded layer

  x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
  #x = BatchNormalization()(x)
  x = UpSampling1D(2)(x) # 6 dims
  x = Conv1D(16, 2, activation='relu')(x) # 5 dims
  #x = BatchNormalization()(x)
  x = UpSampling1D(2)(x) # 10 dims
  decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
  autoencoder = Model(input_window, decoded)

  autoencoder.compile(optimizer='adam', loss='mse')
  history = autoencoder.fit(x_train, x_train,
                  epochs=5,
                  batch_size=32,
                  shuffle=True,
                  validation_split=0.2)

  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()
  encoder.save('models/model_sum'+str(n))
  mname='models/model_sum'+str(n)



model=keras.models.load_model(mname)
with open(output_dataset_file, 'w') as f1:
  for k in range(len(x_test)):
    encoded_stocks =model.predict(x_test[k])
    new = []
    for i in encoded_stocks:
      for j in i:
        new.append(j[0])
    trainPredict_dataset_like = np.zeros(shape=(len(new), len(df.columns))) 
    trainPredict_dataset_like[:,0] =new
    new = sc[k].inverse_transform(trainPredict_dataset_like)[:,0]
    f1.write(df.columns[k])
    for i in range(len(new)):
      f1.write('\t'+(str(new[i])))
    f1.write('\t'+'\n')

    # figure, axes = plt.subplots(ncols=2,figsize=(20,10))
    # axes[0].plot(df.T.iloc[k])
    # axes[1].plot(new)

with open(output_query_file, 'w') as f2:
  for k in range(len(x_test2)):
    encoded_stocks =model.predict(x_test2[k])
    new = []
    for i in encoded_stocks:
      for j in i:
        new.append(j[0])
    trainPredict_dataset_like2 = np.zeros(shape=(len(new), len(df2.columns))) 
    trainPredict_dataset_like2[:,0] =new
    new = sc2[k].inverse_transform(trainPredict_dataset_like2)[:,0]
    f2.write(df2.columns[k])
    for i in range(len(new)):
      f2.write('\t'+(str(new[i])))
    f2.write('\t'+'\n')
    # figure, axes = plt.subplots(ncols=2,figsize=(20,10))
    # axes[0].plot(df2.T.iloc[k])
    # axes[1].plot(new)
