#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:55:01 2021

@author: shvarta3
"""

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
from numpy import linalg as LA
from joblib import Parallel, delayed
import multiprocessing
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

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
start=1
idx=0
threshold_freq=0.3
threshold=40
pad=30
num_cores = multiprocessing.cpu_count()
mode='data'
indices = [0,2,4,6]
class_wieght=np.zeros(3)
scaler = StandardScaler()
num_diraction=18

if mode =='data':
    data_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/separate_files_spectrum/feature_vector_'
    label_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/separate_files_spectrum/label_'
    idx_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/separate_files_spectrum/idx.npy'
    nom_data_sets=21
    lottery=26
             
if mode =='val':
    data_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/val_data_sets/separate_files_spectrum/feature_vector_'
    label_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/val_data_sets/separate_files_spectrum/label_'
    idx_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/val_data_sets/separate_files_spectrum/idx.npy'
    nom_data_sets=3
    lottery=26

    
def stft_z(get_receivers):
    return stft(get_receivers, win, hop, nfft)



for k in range(1,nom_data_sets):
    print(k)
    idx_start_epoch = idx
    for i in range(start,lottery):
        index_file=i+(lottery-1)*(k-1)
        if mode =='data':
            print(i)
            first_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/DB_dynamic/first_%d.wav'%index_file)
            second_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/DB_dynamic/second_%d.wav'%index_file) 
            together_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/DB_dynamic/together_%d.wav'%index_file) 
            label_first_location_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/DB_dynamic/label_location_first_%d.mat'%index_file)
            label_second_location_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/DB_dynamic/label_location_second_%d.mat'%index_file)
       
        if mode =='val':
            print(i)
            first_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/first_%d.wav'%index_file)
            second_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/second_%d.wav'%index_file) 
            together_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/together_%d.wav'%index_file) 
            label_first_location_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/label_location_first_%d.mat'%index_file)
            label_second_location_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/val_dynamic/label_location_second_%d.mat'%index_file)


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


        M=len(receiver_first[0,:])
        index=int(1+np.fix((len(receiver_first[:,1])-wlen)/hop))

        z_k_first=[]
        z_k_first = Parallel(n_jobs=1, verbose=0)(delayed(
        stft_z)(receiver_first[:,i])for i in range(M))
        z_k_first=np.asarray(z_k_first)
        
        z_k=[]
        z_k = Parallel(n_jobs=1, verbose=0)(delayed(
        stft_z)(receivers[:,i])for i in range(M))
        z_k=np.asarray(z_k)

        z_k_log=z_k.T
        z_k_log=np.log(abs(z_k_log))
        for l in range(M):
           z_k_log[:,:,l] = scaler.fit_transform(z_k_log[:,:,l])
        
        X_T = np.zeros((index-frame_before-frame_after,frame_before+frame_after+1,NUP,M))
        index_data = 0
        for l in range(frame_before,index-frame_after): 
            X_T[index_data,:,:] = z_k_log[l-frame_before:l+frame_after+1,:]
            index_data+=1
     
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

        L=vad1+vad2
        L= L[frame_before:index-frame_after]
        
        if (mode=='data') | (mode == 'val'):
        
            if i==start:
                X_total=X_T
                L_total=L
            else :
                X_total=np.concatenate((X_total,X_T))
                L_total=np.concatenate((L_total,L))
              
        
    if (mode=='data') | (mode=='val'):
        
        x_0=X_total[np.where(L_total==0)]
        x_1=X_total[np.where(L_total==1)]
        x_2=X_total[np.where(L_total==2)]


        l_0=L_total[np.where(L_total==0)]
        l_1=L_total[np.where(L_total==1)]
        l_2=L_total[np.where(L_total==2)]
      
        len_balance=min(len(l_0),len(l_1),len(l_2))

        x_0=x_0[0:len_balance,:,:]
        x_1=x_1[0:len_balance,:,:]
        x_2=x_2[0:len_balance,:,:]

        
        l_0=l_0[0:len_balance]
        l_1=l_1[0:len_balance]
        l_2=l_2[0:len_balance]


        
        X_total=np.concatenate((x_0,x_1,x_2))
        L_total=np.concatenate((l_0,l_1,l_2))

    for n in range(X_total.shape[0]):    
        np.save(data_file_name+str(idx)+'.npy' , X_total[n,...])
        np.save(label_file_name+str(idx)+'.npy' , L_total[n])
        idx=idx+1

np.save(idx_file_name,idx)
plt.figure();plt.plot(vad1)
plt.figure();plt.plot(vad2)
plt.figure();plt.plot(L)
plt.figure();plt.hist(L_total,3)
