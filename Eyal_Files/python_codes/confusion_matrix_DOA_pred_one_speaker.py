#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:04:37 2020

@author: shvarta3
"""

import numpy as np
import math
from scipy.io import wavfile
from stft import stft
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
from numpy import linalg as LA   
import pandas as pd
from tensorflow.keras.models import load_model 

start=1
num_signals=11


plt.close("all")
# variens
wlen = 2048                                                                  
hop = wlen/4                                                                    
nfft = 2048                                       
win=np.hamming(wlen)
win_vad = np.hamming(21)

NUP = math.ceil((nfft+1)/2)
first_mic = 2
last_mic = 6
frame_before=8
frame_after = 5 
frame_threshold=8

idx=0
threshold_freq=0.3
threshold=40
pad=30
num_cores = multiprocessing.cpu_count()
mode='val'
indices = [0,2,4,6]
class_wieght=np.zeros(3)
scaler = StandardScaler()
num_diraction=18
labels_location = np.arange(5,185,10)

annot = True
cmap = 'Oranges'
fmt = '.2f'
lw = 0.5
cbar = False
show_null_values = 2
pred_val_axis = 'y'
fz = 9
figsize = [18,18]
num_classes=18

sum_of_no_erorr_nodel = np.zeros((6,3))
sum_of_no_erorr_srp = np.zeros((6,3))

sum_of_no_erorr_nodel[0,0] = 89
sum_of_no_erorr_nodel[1,0] = 88
sum_of_no_erorr_nodel[2,0] = 88
sum_of_no_erorr_nodel[3,0] = 87
sum_of_no_erorr_nodel[4,0] = 86
sum_of_no_erorr_nodel[5,0] = 85

sum_of_no_erorr_nodel[0,1] = 11
sum_of_no_erorr_nodel[1,1] = 12
sum_of_no_erorr_nodel[2,1] = 12
sum_of_no_erorr_nodel[3,1] = 12
sum_of_no_erorr_nodel[4,1] = 14
sum_of_no_erorr_nodel[5,1] = 14.2

sum_of_no_erorr_nodel[0,2] = 0
sum_of_no_erorr_nodel[1,2] = 0
sum_of_no_erorr_nodel[2,2] = 0
sum_of_no_erorr_nodel[3,2] = 0
sum_of_no_erorr_nodel[4,2] = 1
sum_of_no_erorr_nodel[5,2] = 1.8

sum_of_no_erorr_srp[0,0] = 60
sum_of_no_erorr_srp[1,0] = 50
sum_of_no_erorr_srp[2,0] = 45
sum_of_no_erorr_srp[3,0] = 43
sum_of_no_erorr_srp[4,0] = 40
sum_of_no_erorr_srp[5,0] = 33

sum_of_no_erorr_srp[0,1] = 38
sum_of_no_erorr_srp[1,1] = 44
sum_of_no_erorr_srp[2,1] = 46
sum_of_no_erorr_srp[3,1] = 48
sum_of_no_erorr_srp[4,1] = 50
sum_of_no_erorr_srp[5,1] = 51

sum_of_no_erorr_srp[0,2] = 2
sum_of_no_erorr_srp[1,2] = 6
sum_of_no_erorr_srp[2,2] = 9
sum_of_no_erorr_srp[3,2] = 9
sum_of_no_erorr_srp[4,2] = 10
sum_of_no_erorr_srp[5,2] = 16

bars = ('0.30', '0.35','0.4','0.45', '0.50','0.55')
x_pos = np.arange(len(bars))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 14})

# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
f = plt.figure(figsize=(12,6))
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
ax = f.add_subplot(111)
w = 0.1
hist1=ax.bar(x_pos-2*w, sum_of_no_erorr_nodel[:,0],width=0.2, color='royalblue',align='edge')
hist2=ax.bar(x_pos-2*w, sum_of_no_erorr_srp[:,0], width=0.2, color='skyblue', align='edge')
hist3=ax.bar(x_pos, sum_of_no_erorr_srp[:,1],width=0.2, color='mediumpurple',align='edge')
hist4=ax.bar(x_pos, sum_of_no_erorr_nodel[:,1], width=0.2, color='rebeccapurple', align='edge')
hist5=ax.bar(x_pos+2*w, sum_of_no_erorr_srp[:,2],width=0.2, color='pink',align='edge')
hist6=ax.bar(x_pos+2*w, sum_of_no_erorr_nodel[:,2], width=0.2, color='palevioletred', align='edge')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height*0.8])

ax.legend( (hist1[0], hist4[0],hist6[0], hist2[0],hist3[0], hist5[0]), 
          ('Successful prediction -DNN model',
           'Low error prediction - DNN model',
           'High error prediction - DNN model',
           'Successful prediction -SRP',
           'Low error prediction - SRP',
           'High error prediction - SRP'),
           fontsize=14,
           bbox_to_anchor=(0.06,1),ncol=2)

# title1 = ('DOA prediction T60 = %sms'%t60[k])
#plt.title(title1)
plt.ylabel('Percentage success rate',fontsize=14)
plt.xlabel('T60 (sec)',fontsize=14)
# ax.xaxis_date()
# ax.autoscale(tight=True)
plt.xticks(x_pos, bars)
plt.yticks(np.arange(0,101,20))     
# plt.show()
plt.savefig('doa results.png',dpi=300)
print()
# cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
#                   ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
# plot_confusion_matrix_from_data(L_2_one_total-1, label_doa_one_total-1,num_classes,cm_plot_labels,
#   annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


# cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
#                   ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
# plot_confusion_matrix_from_data(L_2_one_total-1, y2_prob_stat_one_speaker_total-1,num_classes,cm_plot_labels,
#   annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)



# label_doa_one[label_doa_one==19] = 0

# plt.figure()
# line_1, = plt.plot(L_2_one, label='True DOA label')
# line_2, = plt.plot(label_doa_one, label='Predicted DOA label - SRP')
# #line_3, = plt.plot(y2_prob_stat_one_speaker, label='Predicted DOA label - DNN model')
# plt.legend(handles=[line_1,line_2])
# plt.title('DOA predicted - SRP vs DNN model')
# plt.xlabel('time (CSD==1)')
# plt.ylabel('DOA label')
# plt.yticks(np.arange(1,19,1))
# plt.margins(x=0)
# plt.show()


# #plt.figure()
# #line_1, = plt.plot(mse_SRP, label='SRP MSE')
# #line_2, = plt.plot(mse_model, label='Model MSE')
# #plt.legend(handles=[line_1,line_2])
# #plt.title('DOA predicted - SRP vs DNN model - MSE')
# #plt.xlabel('SNR')
# #plt.ylabel('MSE')
# #xi = list(range(len(mse_SRP)))
# #plt.xticks(xi,np.arange(10,22.5,2.5))
# #plt.margins(x=0)
# #plt.show()

# #plt.figure()
# #line_1, = plt.plot(mse_SRP, label='SRP MSE')
# #line_2, = plt.plot(mse_model, label='Model MSE')
# #plt.legend(handles=[line_1,line_2])
# #plt.title('DOA predicted - SRP vs DNN model - MSE')
# #plt.xlabel('T60')
# #plt.ylabel('MSE')
# #xi = list(range(len(mse_SRP)))
# #plt.xticks(xi,np.arange(300,600,50))
# #plt.margins(x=0)
# #plt.show()

# plt.figure()
# line_1, = plt.plot(sum_of_no_erorr_srp[:,0], label='SRP - Successful prediction')
# line_2, = plt.plot(sum_of_no_erorr_srp[:,1], label='SRP - Low error prediction')
# #line_3, = plt.plot(sum_of_no_erorr_srp[:,2], label='SRP - High error prediction')
# line_4, = plt.plot(sum_of_no_erorr_nodel[:,0], label='DNN model - Successful prediction')
# line_5, = plt.plot(sum_of_no_erorr_nodel[:,1], label='DNN model - Low error prediction')
# #line_6, = plt.plot(sum_of_no_erorr_nodel[:,2], label='DNN model - High error prediction')

# plt.legend(handles=[line_1,line_2,line_4,line_5])
# plt.title('DOA predicted - SRP vs DNN model - Successful prediction+Low error prediction')
# plt.xlabel('T60')
# plt.ylabel('Percent')
# xi = list(range(len(mse_SRP)))
# plt.xticks(xi,np.arange(300,600,50))
# plt.margins(x=0)
# plt.show()    



