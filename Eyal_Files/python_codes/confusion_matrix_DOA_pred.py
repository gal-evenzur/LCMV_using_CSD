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
import scipy.io as sio
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data

plt.close("all")
# variens
wlen = 2048                                                                  
hop = wlen/4                                                                    
nfft = 2048                                       
win=np.hamming(wlen)
win_vad = np.hamming(21)
NUP = math.ceil((nfft+1)/2)
frame_before=8
frame_after = 5 
frame_threshold=8
start=1
idx=0
threshold_freq=0.3
threshold=40
pad=30
indices = [0,2,4,6]
num_diraction=18

 
def stft_z(get_receivers):
    return stft(get_receivers, win, hop, nfft)


i=1
for i in range(1,51):
    first_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/first_%d.wav'%i)
    second_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/second_%d.wav'%i) 
    together_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/together_%d.wav'%i) 
    label_first_location_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/label_location_first_%d.mat'%i)
    label_second_location_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/label_location_second_%d.mat'%i)
    label_doa_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/label_doa_%d.mat'%i)
    
    fs,receiver_first = wavfile.read(first_file)
    receiver_first = receiver_first[:,indices]
    fs,receiver_second = wavfile.read(second_file)
    receiver_second = receiver_second[:,indices]
    fs,receivers= wavfile.read(together_file)
    receivers = receivers[:,indices]
    
    receiver_first=receiver_first/(abs(receiver_first).max())
    receiver_second=receiver_second/(abs(receiver_second).max())
    receivers=receivers/(abs(receivers).max())
    
    mat_contents_first=sio.loadmat(label_first_location_file)
    mat_contents_second=sio.loadmat(label_second_location_file)
    vad1_location=np.transpose(mat_contents_first['vad_first_speaker'])
    vad2_location=np.transpose(mat_contents_second['vad_second_speaker'])
    
    mat_contents_doa=sio.loadmat(label_doa_file)
    label_doa=np.squeeze(np.transpose(mat_contents_doa['DOAs_out']))

    M=len(receiver_first[0,:])
    index=int(1+np.fix((len(receiver_first[:,1])-wlen)/hop))

    z_k_first=[]
    z_k_first = Parallel(n_jobs=1, verbose=0)(delayed(
    stft_z)(receiver_first[:,i])for i in range(M))
    z_k_first=np.asarray(z_k_first)
      
    vad1_temp=abs(z_k_first)
    vad1_temp = vad1_temp/(vad1_temp.std())
    vad1_temp = vad1_temp.mean(0)
    vad1_temp = vad1_temp > threshold_freq
    vad1_temp=vad1_temp.astype(np.int)
    vad1_temp_sum=vad1_temp.sum(axis=0)
    vad1_temp_sum=vad1_temp_sum.astype(np.int)
    vad1_temp_sum1 = vad1_temp_sum > threshold
    vad1=vad1_temp_sum1.astype(np.int)

    z_k_second=[]
    z_k_second = Parallel(n_jobs=1, verbose=0)(delayed(
    stft_z)(receiver_second[:,i])for i in range(M))
    z_k_second=np.asarray(z_k_second)  
    
    vad2_temp=abs(z_k_second)
    vad2_temp = vad2_temp/(vad2_temp.std())
    vad2_temp = vad2_temp.mean(0)
    vad2_temp = vad2_temp > threshold_freq
    vad2_temp=vad2_temp.astype(np.int)
    vad2_temp_sum=vad2_temp.sum(axis=0)
    vad2_temp_sum=vad2_temp_sum.astype(np.int)
    vad2_temp_sum1 = vad2_temp_sum > threshold
    vad2=vad2_temp_sum1.astype(np.int)


    check_vad1=np.zeros(index)
    check_vad2=np.zeros(index)

    for l in range(frame_before,index-frame_after): 
        check_vad1[l]=vad1[l-1:l+2].sum()
        check_vad2[l]=vad2[l-1:l+2].sum()
    
    for l in range(frame_before,index-frame_after): 
        if check_vad1[l]==3:
            vad1[l]=1
        else:
            vad1[l]=0
        if check_vad2[l]==3:
            vad2[l]=1
        else:
            vad2[l]=0

    vad1_location_update = np.squeeze(vad1_location.T*vad1.T)
    vad2_location_update = np.squeeze(vad2_location.T*vad2.T)
    
    L=vad1+vad2
      
    L= L[frame_before:index-frame_after]
    L2 = np.squeeze(vad1_location_update+vad2_location_update)   
    L2= L2[frame_before:index-frame_after]
    L2=np.where(L!=2, L2,19)
#    label_doa= label_doa[frame_before:index-frame_after]
    label_doa = label_doa+1
    label_doa_one=label_doa[np.where(L==1)]
    L_2_one=L2[np.where(L==1)]
    if i==start:
        label_doa_one_total=label_doa_one
        L_2_one_total=L_2_one
    else :
        label_doa_one_total=np.concatenate((label_doa_one_total,label_doa_one))
        L_2_one_total=np.concatenate((L_2_one_total,L_2_one))        
        
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
cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
                  ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
plot_confusion_matrix_from_data(L_2_one_total-1, label_doa_one_total-1,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

      
label_doa_one[label_doa_one==19] = 0
plt.figure()
line_1, = plt.plot(L_2_one, label='True DOA label')
line_2, = plt.plot(label_doa_one, label='Predicted DOA label')
plt.legend(handles=[line_1,line_2])
plt.title('DOA predicted')
plt.xlabel('time (CSD==1)')
plt.ylabel('DOA label')
plt.show()
