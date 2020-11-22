#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import os
import csv


# In[3]:


#activity labels as defined in activity_labels.txt
activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
activity_labels = {k:v for k,v in enumerate(activity_labels, start=1)}


# In[4]:


#load data
def load_data(path):
    data = pd.read_csv(path, header=None, delim_whitespace=True)
    return data.values

def load_set(path, x, y):
    data = load_data(path+x)
    labels = load_data(path+y)
    return data, labels

train_data, train_labels = load_set('HAPT Data Set/Train/', 'X_train.txt', 'y_train.txt')
test_data, test_labels = load_set('HAPT Data Set/Test/', 'X_test.txt', 'y_test.txt')

print(f'test_data before mod {test_data.shape}')
print(f'train_data before mod {train_data.shape}')

#reshape the data to add a features dimension (features = 1)
#https://stackoverflow.com/questions/43396572/dimension-of-shape-in-conv1d
train_data = np.expand_dims(train_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

print(f'test_data after mod {test_data.shape}')
print(f'train_data after mod {train_data.shape}')


# In[5]:


#input shape
timesteps = train_data.shape[1] #561 timesteps
features = train_data.shape[2] #1 feature


# In[18]:


model = models.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(timesteps,features)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(activity_labels)))

model.summary()


# In[19]:


# Compiling the model
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[21]:


test_loss,test_acc = model.evaluate(test_data, test_labels, verbose=2)


# In[22]:


pred_outs = model.predict(test_data)
print(pred_outs)


# In[23]:


model.fit(train_data, 
          train_labels, 
          epochs=100, 
          validation_data=(test_data, test_labels))

test_loss,test_acc = model.evaluate( test_data, test_labels, verbose=2)

