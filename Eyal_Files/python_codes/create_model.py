
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:05:40 2019

@author: shvarta3
"""

from __future__ import print_function
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import gmtime, strftime
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
import ordinal_categorical_crossentropy18 as OCC
import ordinal_categorical_crossentropy3 as OCC3
from keras.constraints import max_norm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  

data_file_name='/home/dsi/shvarta3/data_sets/separate_files/feature_vector_'
label_file_name='/home/dsi/shvarta3/data_sets/separate_files/label_'
label2_file_name='/home/dsi/shvarta3/data_sets/separate_files/label2_'
idx_file_name='/home/dsi/shvarta3/data_sets/separate_files/idx.npy'

val_data_file_name='/home/dsi/shvarta3/val_data_sets/separate_files/feature_vector_'
val_label_file_name='/home/dsi/shvarta3/val_data_sets/separate_files/label_'
val_label2_file_name='/home/dsi/shvarta3/val_data_sets/separate_files/label2_'
val_idx_file_name='/home/dsi/shvarta3/val_data_sets/separate_files/idx.npy'

plt.close("all")                                                                                                           
batch_size = 64
num_classes3 = 3
num_classes12 = 20
epochs = int(4)
img_rows, img_cols = 1025,7
time1=strftime("%d",gmtime())
time2=strftime("%m",gmtime())
norm = max_norm(5.0)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_file, label_file,label_file2,idx_file, batch_size=batch_size,
                 dim=(img_rows,img_cols),
                 num_classes3=num_classes3,num_classes12=num_classes12):
        
        self.data_file=data_file
        self.label_file=label_file
        self.label_file2=label_file2
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes3 = num_classes3
        self.num_classes12 = num_classes12
        self.size = np.load(idx_file)
        self.on_epoch_end()

    def __len__(self):
        return int(self.size/self.batch_size)
    
    def __getitem__(self, index):
        self.index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y3,y12 = self.__data_generation(self.index)
        return X, [y3,y12]

    def on_epoch_end(self):
        self.indexes = np.arange(self.size)
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))
        y3 = np.empty((self.batch_size), dtype=int)
        y12 = np.empty((self.batch_size), dtype=int)
        
        for i, idx in enumerate(self.index):
            X[i,]=np.load(self.data_file+str(idx)+'.npy')
            
            y3[i]=np.load(self.label_file+str(idx)+'.npy' )
            y12[i]=np.load(self.label_file2+str(idx)+'.npy')
            
        return X, keras.utils.to_categorical(y3, num_classes=self.num_classes3),keras.utils.to_categorical(y12, num_classes=self.num_classes12)
  

inputs=Input(shape=[img_rows,img_cols])

layer1=Conv1D(filters=64, kernel_size=6,strides=2,kernel_constraint=norm, bias_constraint=norm)(inputs)
layer1a = Activation('relu')(layer1)
layer1b=BatchNormalization()(layer1a)
layer1p=MaxPooling1D(pool_size=2)(layer1b)

layer2=Conv1D(filters=64, kernel_size=3,kernel_constraint=norm, bias_constraint=norm)(layer1p)
layer2a = Activation('relu')(layer2)
layer2b=BatchNormalization()(layer2a)
layer2p=MaxPooling1D(pool_size=2)(layer2b)
#layer2d=Dropout(0.25)(layer2p)

layer3=Conv1D(filters=32, kernel_size=3,kernel_constraint=norm, bias_constraint=norm)(layer2p)
layer3a = Activation('relu')(layer3)
layer3b=BatchNormalization()(layer3a)
layer3p=MaxPooling1D(pool_size=2)(layer3b)

layer4=Conv1D(filters=32, kernel_size=3,kernel_constraint=norm, bias_constraint=norm)(layer3p)
layer4a = Activation('relu')(layer4)
layer4b=BatchNormalization()(layer4a)
layer4p=MaxPooling1D(pool_size=2)(layer4b)
layer4d=Dropout(0.25)(layer4p)

layer5f=Flatten()(layer4d)
layer5dense=Dense(128,kernel_constraint=norm, bias_constraint=norm)(layer5f)
layer5a = Activation('relu')(layer5dense)
layer5b=BatchNormalization()(layer5a)
layer5d=Dropout(0.25)(layer5b)

layer_FC_out_18=Dense(32,kernel_constraint=norm, bias_constraint=norm)(layer5d)
layer_FC_out_18a = Activation('relu')(layer_FC_out_18)
layer_FC_out_18d = Dropout(0.25)(layer_FC_out_18a)

layer_FC_out_3=Dense(10,kernel_constraint=norm, bias_constraint=norm)(layer5d)
layer_FC_out_3a = Activation('relu')(layer_FC_out_3)

output3=Dense(3, activation='softmax',name='out_3')(layer_FC_out_3a)
output18=Dense(20, activation='softmax',name='out_12')(layer_FC_out_18d)
model=Model(inputs=inputs, outputs=[output3,output18])

for layer in model.layers:
    print(layer.output_shape)

lossWeights = {"out_3": 1.0, "out_12": 2.0}   
model.compile(loss=[OCC3.loss33,OCC.loss1]  , loss_weights=lossWeights,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model3=Model(inputs=inputs, outputs=output3)
model18=Model(inputs=inputs, outputs=output18)

training_generator = DataGenerator(
    data_file_name, 
    label_file_name,
    label2_file_name,
    idx_file_name)

validation_generator = DataGenerator(
    val_data_file_name, 
    val_label_file_name,
    val_label2_file_name,
    val_idx_file_name)


# filepath="/mnt/dsi_vol1/users/ayal_shvarts/project/ratio/weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
#                    use_multiprocessing=True,
                    workers=12
                    # callbacks=callbacks_list
                    )

model3.save('/home/dsi/shvarta3/models/model_GEVD_18_separate_%s_%s.h5'%(time1,time2))
model18.save('/home/dsi/shvarta3/models/model2_GEVD_18_separate_%s_%s.h5'%(time1,time2))

#confution matrix
size = np.load(val_idx_file_name)

#from keras.models import load_model
#forlder_to_work = '/mnt/dsi_vol1/users/ayal_shvarts/project/'
#model3=load_model(forlder_to_work+'count_speakers/model_GEVD_18_separate_24_02.h5')
#model18=load_model(forlder_to_work+'count_speakers/model2_GEVD_18_separate_24_02.h5')

x_test=np.zeros((size,img_rows, img_cols),dtype=float)
y=np.zeros(size)
y2=np.zeros(size)
for index in range(size):
    x_test[index,:,:]=np.squeeze(np.load(val_data_file_name+str(index)+'.npy'))
    y[index]=np.load(val_label_file_name+str(index)+'.npy' )
    y2[index]=np.load(val_label2_file_name+str(index)+'.npy' )

        
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
cm_plot_labels = ['Noise','One speaker','2 speakers']
plot_confusion_matrix_from_data(y, y_prob3,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

y_pred12=model18.predict(x_test)
y_pred12_2=pd.DataFrame(y_pred12)
y_prob12=y_pred12_2.idxmax(axis=1)

y2_delete = np.copy(y2)
y_prob12_np = y_prob12.to_numpy()
y_prob12_np = np.delete(y_prob12_np, np.where(y2_delete==0)[0])
y2_delete = np.delete(y2_delete, np.where(y2_delete==0))
y_prob12_np = np.delete(y_prob12_np, np.where(y2_delete==19))
y2_delete = np.delete(y2_delete, np.where(y2_delete==19))

num_classes=model18.layers[-1].output_shape[1]-2    
cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
                  ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
plot_confusion_matrix_from_data(y2_delete-1, y_prob12_np-1,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)




