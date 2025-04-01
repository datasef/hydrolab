#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 09:24:30 2024

@author: trungtd
"""
import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%% 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  if dropnan:
    agg.dropna(inplace=True)
  return pd.DataFrame(agg.astype('float32'))
#%%
def train_valid_test_split(data, Lag_days, k_tasks):
  data_train_valid = data.iloc[:578448,:] #
  data_test = data.iloc[578448:,:] # 
  data_train_valid.dropna(inplace=True)
  data_test.dropna(inplace=True)
  data_valid, data_train = train_test_split(data_train_valid, test_size=0.2, shuffle= False) #20% used for validation.
  return data_train.values, data_valid.values, data_test.values
#%%
def prepare_data(move_window, Lag_days, k_tasks):
  data = pd.read_csv('Input_data.csv').iloc[:,1:]# 
  # simple min-max scaling.
  scaler = MinMaxScaler()
  scaler.fit(data.iloc[:578448,:]) # min-max scaling without the test dataset.
  q_max = np.max(data.iloc[:,5])
  q_min = np.min(data.iloc[:,5])
  data_scaled = scaler.transform(data)
  # data split
  data_sequence = series_to_supervised(data_scaled, Lag_days, k_tasks)
  data_train, data_valid, data_test = train_valid_test_split(data_sequence, Lag_days, k_tasks)
  
  # Split data into 2 parts for lag-time and forecasting tasks
  train_x1_pr = data_train[:,0::6].reshape(-1, Lag_days+k_tasks, 1)
  train_x2_pr = data_train[:,1::6].reshape(-1, Lag_days+k_tasks, 1)
  train_x3_pr = data_train[:,2::6].reshape(-1, Lag_days+k_tasks, 1)
  train_x4_pr = data_train[:,3::6].reshape(-1, Lag_days+k_tasks, 1)
  train_x5_pr = data_train[:,4::6].reshape(-1, Lag_days+k_tasks, 1)   
  train_inflow = data_train[:,5::6].reshape(-1, Lag_days+k_tasks, 1)
  train_x_inflow = train_inflow[:,:Lag_days,:]
  train_y = train_inflow[:,Lag_days:,:]

  valid_x1_pr = data_valid[:,0::6].reshape(-1, Lag_days+k_tasks, 1)
  valid_x2_pr = data_valid[:,1::6].reshape(-1, Lag_days+k_tasks, 1)
  valid_x3_pr = data_valid[:,2::6].reshape(-1, Lag_days+k_tasks, 1)
  valid_x4_pr = data_valid[:,3::6].reshape(-1, Lag_days+k_tasks, 1)
  valid_x5_pr = data_valid[:,4::6].reshape(-1, Lag_days+k_tasks, 1)
  valid_inflow = data_valid[:,5::6].reshape(-1, Lag_days+k_tasks, 1)
  valid_x_inflow = valid_inflow[:,:Lag_days,:]
  valid_y = valid_inflow[:,Lag_days:,:]


  test_x1_pr = data_test[:,0::6].reshape(-1, Lag_days+k_tasks, 1)
  test_x2_pr = data_test[:,1::6].reshape(-1, Lag_days+k_tasks, 1)
  test_x3_pr = data_test[:,2::6].reshape(-1, Lag_days+k_tasks, 1)
  test_x4_pr = data_test[:,3::6].reshape(-1, Lag_days+k_tasks, 1)
  test_x5_pr = data_test[:,4::6].reshape(-1, Lag_days+k_tasks, 1)
  test_inflow = data_test[:,5::6].reshape(-1, Lag_days+k_tasks, 1)
  test_x_inflow = test_inflow[:,:Lag_days,:]
  test_y = test_inflow[:,Lag_days:,:]


  train_x = [train_x_inflow, train_x1_pr, train_x2_pr, train_x3_pr,train_x4_pr,train_x5_pr]
  valid_x = [valid_x_inflow, valid_x1_pr, valid_x2_pr, valid_x3_pr,valid_x4_pr,valid_x5_pr]
  test_x =  [test_x_inflow,  test_x1_pr,  test_x2_pr,  test_x3_pr,test_x4_pr,test_x5_pr]
  return train_x, train_y, valid_x, valid_y, test_x, test_y, q_max, q_min 
#%% design network for multi-tasks
def Multi_LSTM(Lag_days, k_tasks):
  input_1 = Input(shape=(Lag_days, 1), name='input_1')
  LSTM1 = LSTM(48, return_sequences=False)(input_1)
  # input of 1st pr observation and forecast
  input_2 = Input(shape=((Lag_days+k_tasks), 1), name='input_2') 
  LSTM2 = LSTM(48, return_sequences=False)(input_2)                                    
  # input of 2nd pr observation and forecast, LSTM encoder
  input_3 = Input(shape=((Lag_days+k_tasks), 1), name='input_3') 
  LSTM3 = LSTM(48, return_sequences=False)(input_3)
    # input of 3rd pr observation and forecast, LSTM encoder
  input_4 = Input(shape=((Lag_days+k_tasks), 1), name='input_4')
  LSTM4 = LSTM(48, return_sequences=False)(input_4)
  # input of 4rd pr observation and forecast, LSTM encoder
  input_5 = Input(shape=((Lag_days+k_tasks), 1), name='input_5')
  LSTM5 = LSTM(48, return_sequences=False)(input_5)
  input_6 = Input(shape=((Lag_days+k_tasks), 1), name='input_6')
  LSTM6 = LSTM(48, return_sequences=False)(input_6)

  x = concatenate([LSTM1, LSTM2, LSTM3, LSTM4,LSTM5,LSTM6])
  x = RepeatVector(k_tasks)(x) 

  # LSTM multi-tasks
  x = LSTM(48, return_sequences=True)(x)
  # fully-connected LSTM layers
  dim_dense=[32]
  # final fully-connected LSTM layers for final result
  for dim in dim_dense:
    x = TimeDistributed(Dense(dim, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.1))(x) 
  main_out = TimeDistributed(Dense(1, activation='relu'))(x) 
  main_out = Flatten()(main_out)
  model = Model(inputs=[input_1, input_2, input_3, input_4,input_5, input_6], outputs=main_out)
  return model
#%% KGE, NSE for evaluation metrics
def nse(y_true, y_pred):
  return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)
def kge(y_true, y_pred):
  kge_r = np.corrcoef(y_true,y_pred)[1][0]
  kge_a = np.std(y_pred)/np.std(y_true)
  kge_b = np.mean(y_pred)/np.mean(y_true)
  return 1-np.sqrt((kge_r-1)**2+(kge_a-1)**2+(kge_b-1)**2)
#%% Seting
move_window= 10 # Add moving average inflow for input
k_tasks = 36
Lag_days =24
batch_size = 32
epochs =50
Model_name = 'LSTM'
#%% processing data
x_train, y_train, x_valid, y_valid, x_test, y_test, q_max, q_min = prepare_data(move_window, Lag_days, k_tasks)
model1 = Multi_LSTM(Lag_days, k_tasks)
#%% compile settings
#%% 
#custome loss function (NSE)
def nseloss(y_true, y_pred):
  return K.sum((y_pred-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, cooldown=30, min_lr=1e-8)
earlystoping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(Model_name+'model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
optimizer = RMSprop(learning_rate=0.0001)
model1.compile(optimizer=optimizer, loss='mse')
#%% train model
history = model1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(x_valid, y_valid), callbacks=[reduce_lr, checkpoint], verbose=1)
#%% Save training loss
loss_train = history.history['loss']
loss_valid = history.history['val_loss']
loss_train = pd.DataFrame({'TrainLoss':loss_train})
loss_valid = pd.DataFrame({'TestLoss':loss_valid})
LossEpoches = pd.concat([loss_train, loss_valid], axis=1)
LossEpoches.to_csv(Model_name+'loss.csv', index = True)
#%% Save h5 model
model1.load_weights(Model_name+'model.h5')
y_model_scaled = model1.predict(x_test)
y_model = y_model_scaled*(q_max-q_min)+q_min
y_test = y_test*(q_max-q_min)+q_min
