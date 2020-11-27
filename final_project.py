#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib inline


# In[2]:


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import time
import math


# In[3]:


#activity labels as defined in activity_labels.txt
activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
#activity_labels = {k:v for k,v in enumerate(activity_labels, start=1)}
#print(activity_labels)


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

print('reshaping data...')

#reshape the data to add a features dimension (features = 1)
#https://stackoverflow.com/questions/43396572/dimension-of-shape-in-conv1d
train_data = np.expand_dims(train_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

print('adjusting labels...')
#reduce the labels by 1 to match with the activity_labels and also to start labels at 0 to 11 instead of from 1 to 12
def adjust_labels (labels):
    for i in range(len(labels)-1):
        labels[i][0] -= 1

adjust_labels(train_labels);
adjust_labels(test_labels);


# In[5]:


#input shape
timesteps = train_data.shape[1] #561 timesteps
features = train_data.shape[2] #1 feature


# In[6]:


model = models.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(timesteps,features)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
#model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(12, activation='relu'))

model.summary()


# In[7]:


# Compiling the model
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[8]:


test_loss,test_acc = model.evaluate(test_data, test_labels, verbose=2)


# In[9]:


pred_outs = model.predict_classes(test_data)

#test if the label matches the prediction
false_pred = 0
true_pred = 0
#look at predictions for the first 25 values
pred_range = 25
for i in range(pred_range):
    if not (0 <= pred_outs[i] or pred_outs[i] <= 11):
        print('prediction out of bounds')
        break
        
    print(f'Test label: {activity_labels[test_labels[i][0]]}')
    print(f'Predicted label:{activity_labels[pred_outs[i]]}')
    
    if pred_outs[i]==test_labels[i][0]:
        print('true\n')
        true_pred += 1
    else:
        print('false\n')
        false_pred += 1
print(f'False predictions:{false_pred}')
print(f'True predictions:{true_pred}')
print(f'Prediction accuraccy for first 25 values: {true_pred/pred_range}')


# In[10]:


#get time of epochs to record training time
#https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
time_callback = TimeHistory()

#train the model
model.fit(train_data, 
          train_labels, 
          epochs=15, 
          validation_data=(test_data, test_labels),
         callbacks=[time_callback])

test_loss,test_acc = model.evaluate( test_data, test_labels, verbose=2)


# In[11]:


training_time = sum(time_callback.times)
print(f"Total training time: {math.floor(training_time/1)}s {math.floor(training_time%1 * 1000)}ms {math.ceil(training_time%(1/1000)*1000)}us")


# In[ ]:




