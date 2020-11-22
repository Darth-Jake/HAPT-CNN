#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

import os


# In[3]:


#activity labels as defined in activity_labels.txt
activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
activity_labels = {k:v for k,v in enumerate(activity_labels, start=1)}


# In[4]:


#load the data
def get_data( data_path ):
    file = open(data_path)
    lines = file.readlines()
    data = []
    for line in lines:
        arr=[]
        for x in line.split():
            arr.append([float(x)])
        data.append(arr)
    return data

def get_labels( path ):
    file = open(path)
    lines = file.readlines()
    data=[]
    for x in lines:
        data.append(int(x))
    return data
    
training_data = get_data('HAPT Data Set/Train/X_train.txt')
training_labels = get_labels('HAPT Data Set/Train/y_train.txt')
test_data = get_data('HAPT Data Set/Test/X_test.txt')
test_labels = get_labels('HAPT Data Set/Test/y_test.txt')


# In[5]:


#test print
for i in range(1):
    print(f'{activity_labels[training_labels[i]]}({training_labels[i]}): {training_data[i]}\n')
    


# In[8]:


model = models.Sequential()
model.add(layers.Conv1D(32, 3, strides=1, activation='relu', input_shape=(561,1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(activity_labels)))

model.summary()


# In[9]:


# Compiling the model
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[10]:


test_loss,test_acc = model.evaluate(test_data, test_labels, verbose=2)


# In[11]:


pred_outs = model.predict(test_data)
print(pred_outs)


# In[12]:


model.fit(training_data, 
          training_labels, 
          epochs=10, 
          validation_data=(test_data, test_labels))

test_loss,test_acc = model.evaluate( test_data, test_labels, verbose=2)


# In[ ]:




