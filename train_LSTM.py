#!/usr/bin/env python
# coding: utf-8 In[ ]:

import numpy as np 
import os 
os.environ["KERAS_BACKEND"] = "tensorflow" 
import h5py 
import pickle 
import pandas 
import matplotlib.pyplot as plt 

import tensorflow as tf 
from keras import backend as K 
from keras.models import Model 
from keras.layers import Input, Dense, Dropout 
from keras.utils import plot_model 

from utils import load_data

# In[ ]:

from keras.layers import Input, LSTM, RepeatVector, CuDNNLSTM

# In[ ]:

x_train=load_data('AE_training_qcd_preprocessed.h5', 80)
#x_train=normalize(x_train, norm='l2',axis=1)# 0: normalize each feature; 1: normalize each sample

x_train=np.reshape(x_train,(len(x_train), -1, 4))
#x_validation=np.reshape(x_validation, (len(x_validation), -1, 4))

# In[ ]:

input_dim=4 
timesteps=20 
latent_dim=10 

inputs = Input(shape=(None, input_dim)) 
encoded = LSTM(latent_dim)(inputs) 

decoded = RepeatVector(timesteps)(encoded)
#decoded = LSTM(input_dim, return_sequences=True)(decoded) # for CPU users
decoded = CuDNNLSTM(input_dim, return_sequences=True)(decoded) # for GPU users 
sequence_autoencoder = Model(inputs, decoded) 
encoder = Model(inputs, encoded)

# In[ ]:

sequence_autoencoder.compile(optimizer="adam",
                    # change the loss to measuring difference 
                    # between distributions (D_{KL}) 
                    # loss="kullback_leibler_divergence")
                    loss="mean_squared_error")

# In[ ]:

from keras.callbacks import TensorBoard, EarlyStopping 

early_stopping = EarlyStopping(monitor='val_loss', min_delta=5e-4, patience=10) 
history=sequence_autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_split=0.2,
                #validation_data=(x_validation, x_validation), 
                #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) 
                ##http://0.0.0.0:6006
                callbacks=[early_stopping]
                                )

# In[ ]:

sequence_autoencoder.save('AE-LSTM-model-d80-test.h5')
