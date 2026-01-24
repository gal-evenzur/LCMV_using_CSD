# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:05:40 2019

@author: shvarta3
"""

from __future__ import print_function
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import gmtime, strftime
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
#import ordinal_categorical_crossentropy3 as OCC3
from keras.constraints import max_norm
from keras import losses

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  

data_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/separate_files_spectrum/feature_vector_'
label_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/separate_files_spectrum/label_'
idx_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/separate_files_spectrum/idx.npy'

val_data_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/val_data_sets/separate_files_spectrum/feature_vector_'
val_label_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/val_data_sets/separate_files_spectrum/label_'
val_idx_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/val_data_sets/separate_files_spectrum/idx.npy'

plt.close("all")                                                                                                           
batch_size = 64
num_classes3 = 3
epochs = int(1)
img_rows, img_cols,img_deep = 14,1025,4
time1=strftime("%d",gmtime())
time2=strftime("%m",gmtime())
norm = max_norm(5.0)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_file, label_file,idx_file, batch_size=batch_size,
                 dim=(img_rows,img_cols,img_deep),
                 num_classes3=num_classes3):
        
        self.data_file=data_file
        self.label_file=label_file
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes3 = num_classes3
        self.size = np.load(idx_file)
        self.on_epoch_end()

    def __len__(self):
        return int(self.size/self.batch_size)
    
    def __getitem__(self, index):
        self.index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y3 = self.__data_generation(self.index)
        return X,y3

    def on_epoch_end(self):
        self.indexes = np.arange(self.size)
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))
        y3 = np.empty((self.batch_size), dtype=int)
        
        for i, idx in enumerate(self.index):
            X[i,]=np.load(self.data_file+str(idx)+'.npy')         
            y3[i]=np.load(self.label_file+str(idx)+'.npy')
            
        return X, keras.utils.to_categorical(y3, num_classes=self.num_classes3)
  

inputs=Input(shape=[img_rows,img_cols,img_deep])

#layer1=Conv2D(filters=16, kernel_size=(14,6),strides=(1,2),kernel_constraint=norm, bias_constraint=norm)(inputs)
#layer1a = Activation('relu')(layer1)
#layer1b=BatchNormalization()(layer1a)
#layer1p=MaxPooling2D(pool_size=(1,2))(layer1b)

#layer2=Conv2D(filters=64, kernel_size=(1,3),kernel_constraint=norm, bias_constraint=norm)(layer1p)
#layer2a = Activation('relu')(layer2)
#layer2b=BatchNormalization()(layer2a)
#layer2p=MaxPooling2D(pool_size=(1,2))(layer2b)
##layer2d=Dropout(0.25)(layer2p)

#layer3=Conv2D(filters=16, kernel_size=(1,3),kernel_constraint=norm, bias_constraint=norm)(layer1p)
#layer3a = Activation('relu')(layer3)
#layer3b=BatchNormalization()(layer3a)
#layer3p=MaxPooling2D(pool_size=(1,2))(layer3b)

#layer4=Conv2D(filters=32, kernel_size=(1,3),kernel_constraint=norm, bias_constraint=norm)(layer3p)
#layer4a = Activation('relu')(layer4)
#layer4b=BatchNormalization()(layer4a)
#layer4p=MaxPooling2D(pool_size=(1,2))(layer4b)
#layer4d=Dropout(0.25)(layer4p)

layer5f=Flatten()(inputs)

layer1dense=Dense(4096,kernel_constraint=norm, bias_constraint=norm)(layer5f)
layer1a = Activation('relu')(layer1dense)
layer1b=BatchNormalization()(layer1a)

layer2dense=Dense(1984,kernel_constraint=norm, bias_constraint=norm)(layer1b)
layer2a = Activation('relu')(layer2dense)
layer2b=BatchNormalization()(layer2a)

layer5dense=Dense(128,kernel_constraint=norm, bias_constraint=norm)(layer2a)
layer5a = Activation('relu')(layer5dense)
layer5b=BatchNormalization()(layer5a)
#layer5d=Dropout(0.5)(layer5b)

#layer_FC_out_3=Dense(10,kernel_constraint=norm, bias_constraint=norm)(layer5d)
#layer_FC_out_3a = Activation('relu')(layer_FC_out_3)

output3=Dense(3, activation='softmax',name='out_3')(layer5b)
model=Model(inputs=inputs, outputs=output3)

for layer in model.layers:
    print(layer.output_shape)
  
model.compile(loss=losses.categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model3=Model(inputs=inputs, outputs=output3)

training_generator = DataGenerator(
    data_file_name, 
    label_file_name,
    idx_file_name)

validation_generator = DataGenerator(
    val_data_file_name, 
    val_label_file_name,
    val_idx_file_name)


model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    workers=12
                    )

model3.save('/home/dsi/shvarta3/models/model_GEVD_18_separate_only_csd_spectrum_multi_channel_%s_%s.h5'%(time1,time2))

#confution matrix
size = np.load(val_idx_file_name)

x_test=np.zeros((size,img_rows, img_cols,img_deep),dtype=float)
y=np.zeros(size)
for index in range(size):
    x_test[index,:,:]=np.squeeze(np.load(val_data_file_name+str(index)+'.npy'))
    y[index]=np.load(val_label_file_name+str(index)+'.npy' )
        
annot = True
cmap = 'Oranges'
fmt = '.2f'
lw = 0.5
cbar = False
show_null_values = 2
pred_val_axis = 'y'
fz = 9
figsize = [18,18]

y_pred3=model3.predict(x_test)
y_pred3_2=pd.DataFrame(y_pred3)
y_prob3=y_pred3_2.idxmax(axis=1)

num_classes=model3.layers[-1].output_shape[1]     
cm_plot_labels = ['Noise','One speacker','2 speackers']
plot_confusion_matrix_from_data(y, y_prob3,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


