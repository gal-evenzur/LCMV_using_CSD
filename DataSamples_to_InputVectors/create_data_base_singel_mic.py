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
from istft import istft
import matplotlib.pyplot as plt
import h5py
import random
from numpy import linalg as LA
from joblib import Parallel, delayed
import multiprocessing
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
#


plt.close("all")
# variens
wlen = 2048                                                                  
hop = wlen/4                                                                    
nfft = 2048                                       
win=np.hamming(wlen)
NUP = math.ceil((nfft+1)/2)
frame_pad=10
frame=7
frame_threshold=9
start=1
value=0.02
idx=0
threshold_freq=0.3
threshold=40
deep=12
pad=50
num_cores = multiprocessing.cpu_count()
mode='test_y2'
e=0.01
class_wieght=np.zeros(3)
scaler = StandardScaler()


if mode =='data_y2':
    data_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/dataset_singel_mic_RD'
    label_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/label_dataset_singel_mic_RD'
    label2_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/data_sets/label2_dataset_singel_mic_RD'
    nom_data_sets=11
    lottery=51
    f_data = h5py.File(data_file_name, 'w')
    f_label = h5py.File(label_file_name, 'w')
    f_label2 = h5py.File(label2_file_name, 'w')
             
if mode =='test_y2':
    data_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/test/dataset_singel_mic_RD'
    label_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/test/label_dataset_singel_mic_RD'
    label2_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/test/label2_dataset_singel_mic_RD'
    nom_data_sets=3
    lottery=51
    f_data = h5py.File(data_file_name, 'w')
    f_label = h5py.File(label_file_name, 'w')
    f_label2 = h5py.File(label2_file_name, 'w')
      
if mode =='teta':
    data_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/test_emb_audio/dataset'
    label_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/test_emb_audio/label' 
    label2_file_name='/mnt/dsi_vol1/users/ayal_shvarts/project/test_emb_audio/label2'
    nom_data_sets=2
    lottery=2
    f_data = h5py.File(data_file_name, 'w')
    f_label = h5py.File(label_file_name, 'w')
    f_label2 = h5py.File(label2_file_name, 'w')
        
if mode =='test_kmeans':
    nom_data_sets=5
    lottery=2
def stft_z(get_receivers):
    return stft(get_receivers, win, hop, nfft)



          

for k in range(start,nom_data_sets):
    print(k)
    for i in range(start,lottery):
        index_file=i+(lottery-1)*(k-1)
        if mode =='data_y2':
            print(i)
            first_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/data_base_room_diffuse/first_%d.wav'%index_file)
            second_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/data_base_room_diffuse/second_%d.wav'%index_file) 
            together_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/data_base_room_diffuse/together_%d.wav'%index_file) 
            rtf_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/data_base_room_diffuse_rtf/rtf_%d.mat'%index_file)
            mat_contents=sio.loadmat(rtf_file)
            label_location=np.transpose(mat_contents['label_location'])
            label_location=label_location[0][0][0][0]
            fs,receiver_first = wavfile.read(first_file)
            fs,receiver_second = wavfile.read(second_file)
            fs,receivers= wavfile.read(together_file)
            receiver_second=receiver_second/(abs(receiver_second).max())
            receivers=receivers/(abs(receivers).max())
        
        if mode =='test_y2':
            print(i)
            first_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_room_diffuse/first_%d.wav'%index_file)
            second_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_room_diffuse/second_%d.wav'%index_file)           
            rtf_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_room_diffuse_rtf/rtf_%d.mat'%index_file)
            together_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_room_diffuse/together_%d.wav'%index_file) 
            mat_contents=sio.loadmat(rtf_file)
            label_location=np.transpose(mat_contents['label_location'])
            label_location=label_location[0][0][0][0]
            fs,receiver_first = wavfile.read(first_file)
            fs,receiver_second = wavfile.read(second_file)
            receiver_second=receiver_second/(abs(receiver_second).max())
            fs,receivers= wavfile.read(together_file)
            receivers=receivers/(abs(receivers).max())

        if mode =='test_kmeans': 
            print(i)
            first_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test/first_%d.wav'%index_file)
            second_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test/second_%d.wav'%index_file)           
            rtf_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_rtf/rtf_%d.mat'%index_file)
            together_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test/together_%d.wav'%index_file) 
            mat_contents=sio.loadmat(rtf_file)
            rtf_data=np.transpose(mat_contents['rtf'],(2,0,1))
            label_location=np.transpose(mat_contents['label_location'])
            label_location=label_location[0][0][0][0]
            fs,receiver_first = wavfile.read(first_file)
            fs,receiver_second = wavfile.read(second_file)
            receiver_second=receiver_second/(abs(receiver_second).max())
            fs,receivers= wavfile.read(together_file)
            receivers=receivers/(abs(receivers).max())
            
            data_file_name=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_kmeans_location/data_sets/dataset_kmeans_%d'%index_file)
            label_file_name=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_kmeans_location/data_sets/label_kmeans_%d'%index_file)
            label2_file_name=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_kmeans_location/data_sets/label2_kmeans_%d'%index_file)
            f_data = h5py.File(data_file_name, 'w')
            f_label = h5py.File(label_file_name, 'w')
            f_label2 = h5py.File(label2_file_name, 'w')
            
            
        if mode =='teta':
            first_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_emb_audio/first.wav')
            rtf_file=('/mnt/dsi_vol1/users/ayal_shvarts/project/test_emb_audio/rtf_1.mat')
            mat_contents=sio.loadmat(rtf_file)
            rtf_data=np.transpose(mat_contents['rtf'],(2,0,1))
            label_location=np.transpose(mat_contents['label_location'])
            label_location=label_location[0][0][0][0]
            fs,receiver_first = wavfile.read(first_file)
         
        
        
        
        
        receiver_first = receiver_first/(abs(receiver_first).max())
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
        z_k=z_k[:,:,frame:index-frame]
        
        # z_k_0_re=z_k[0,:,:].T.real
        # z_k_0_re_standart = scaler.fit_transform(z_k_0_re)
        
        # z_k_0_im=z_k[0,:,:].T.imag
        # z_k_0_im_standart = scaler.fit_transform(z_k_0_im)
        
        # z_k_0_re_standart=np.reshape(z_k_0_re_standart, (z_k_0_re_standart.shape[0],NUP,1))
        # z_k_0_im_standart=np.reshape(z_k_0_im_standart, (z_k_0_im_standart.shape[0],NUP,1))
        # z_k_0=np.concatenate((z_k_0_re_standart,z_k_0_im_standart),axis=2)

        z_k_0=z_k[0,:,:].T
        z_k_0=np.log(abs(z_k_0))
        z_k_0_standart = scaler.fit_transform(z_k_0)
        z_k_0_standart=np.reshape(z_k_0_standart, (z_k_0_standart.shape[0],NUP,1))
        
        # for v in range(1,deep):
        #     X_T[:,:,v] = scaler.fit_transform(X_T[:,:,v])

        X_T=z_k_0_standart
        
        vad1_temp=abs(z_k_first)
        vad1_temp = vad1_temp/(vad1_temp.std())
        vad1_temp = vad1_temp.mean(0)
        vad1_temp = vad1_temp > threshold_freq
        vad1_temp=vad1_temp.astype(np.int)
        vad1_temp_sum=vad1_temp.sum(axis=0)
        vad1_temp_sum=vad1_temp_sum.astype(np.int)
        vad1_temp_sum1 = vad1_temp_sum > threshold
        vad1=vad1_temp_sum1.astype(np.int)

        if (mode=='test') | (mode == 'data') | (mode == 'data_small') | (mode == 'test_kmeans') | (mode == 'data_y2') | (mode == 'test_y2'):
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
        
            L=vad1+vad2

        
        

        L= L[frame:index-frame]
        
        if (mode=='test') | (mode == 'data') | (mode == 'data_small') | (mode == 'test_kmeans') | (mode == 'data_y2') | (mode == 'test_y2'):
            L2=vad1*label_location[0]+vad2*label_location[1]
        else:
            L2=vad1*label_location[0]
        
        L2= L2[frame:index-frame]
        L2=np.where(L!=2, L2,11)
        


        
        if (mode=='test') | (mode == 'data') | (mode == 'data_small') | (mode== 'data_y2') | (mode== 'test_y2'):
        
            if i==start:
                X_total=X_T
                L_total=L
                L2_total=L2
            else :
                X_total=np.concatenate((X_total,X_T))
                L_total=np.concatenate((L_total,L))
                L2_total=np.concatenate((L2_total,L2))
        
    #class_wieght=class_wieght+np.bincount(L_total)    
    # f_data.create_dataset('mydataset', data=X_total)
    # f_label.create_dataset('mydataset2', data=L_total)
    # f_label2.create_dataset('mydataset2', data=L2_total)
    if (mode=='test_kmeans') :
        X_total=X_T
        L_total=L
        L2_total=L2
        
    if (mode=='test') | (mode == 'data') | (mode == 'data_small'):
        
        x_0=X_total[np.where(L_total==0)]
        x_1=X_total[np.where(L_total==1)]
        x_2=X_total[np.where(L_total==2)]
        l_0=L_total[np.where(L_total==0)]
        l_1=L_total[np.where(L_total==1)]
        l_2=L_total[np.where(L_total==2)]
        l2_0=L2_total[np.where(L_total==0)]
        l2_1=L2_total[np.where(L_total==1)]
        l2_2=L2_total[np.where(L_total==2)]  
        len_balance=min(len(l_0),len(l_1),len(l_2))
        x_0=x_0[0:len_balance,:,:]
        x_1=x_1[0:len_balance,:,:]
        x_2=x_2[0:len_balance,:,:]
        l_0=l_0[0:len_balance]
        l_1=l_1[0:len_balance]
        l_2=l_2[0:len_balance]
        l2_0=l2_0[0:len_balance]
        l2_1=l2_1[0:len_balance]
        l2_2=l2_2[0:len_balance]
        X_total=np.concatenate((x_0,x_1,x_2))
        L_total=np.concatenate((l_0,l_1,l_2))
        L2_total=np.concatenate((l2_0,l2_1,l2_2))
        
    if (mode=='data_y2') | (mode=='test_y2'):
        
        x_0=X_total[np.where(L2_total==0)]
        x_1=X_total[np.where(L2_total==1)]
        x_2=X_total[np.where(L2_total==2)]
        x_3=X_total[np.where(L2_total==3)]
        x_4=X_total[np.where(L2_total==4)]
        x_5=X_total[np.where(L2_total==5)]
        x_6=X_total[np.where(L2_total==6)]
        x_7=X_total[np.where(L2_total==7)]
        x_8=X_total[np.where(L2_total==8)]
        x_9=X_total[np.where(L2_total==9)]
        x_10=X_total[np.where(L2_total==10)]
        x_11=X_total[np.where(L2_total==11)]
        
        l_0=L_total[np.where(L2_total==0)]
        l_1=L_total[np.where(L2_total==1)]
        l_2=L_total[np.where(L2_total==2)]
        l_3=L_total[np.where(L2_total==3)]
        l_4=L_total[np.where(L2_total==4)]
        l_5=L_total[np.where(L2_total==5)]
        l_6=L_total[np.where(L2_total==6)]
        l_7=L_total[np.where(L2_total==7)]
        l_8=L_total[np.where(L2_total==8)]
        l_9=L_total[np.where(L2_total==9)]
        l_10=L_total[np.where(L2_total==10)]
        l_11=L_total[np.where(L2_total==11)]
        
        l2_0=L2_total[np.where(L2_total==0)]
        l2_1=L2_total[np.where(L2_total==1)]
        l2_2=L2_total[np.where(L2_total==2)]
        l2_3=L2_total[np.where(L2_total==3)]
        l2_4=L2_total[np.where(L2_total==4)]
        l2_5=L2_total[np.where(L2_total==5)]
        l2_6=L2_total[np.where(L2_total==6)]
        l2_7=L2_total[np.where(L2_total==7)]
        l2_8=L2_total[np.where(L2_total==8)]
        l2_9=L2_total[np.where(L2_total==9)]
        l2_10=L2_total[np.where(L2_total==10)]
        l2_11=L2_total[np.where(L2_total==11)]
        
        
        len_balance=min(len(l2_1),len(l2_2),len(l2_3),len(l2_4),len(l2_5),len(l2_6),len(l2_7),len(l2_8),len(l2_9),len(l2_10))
        len_balance=min(len_balance,int(len(l2_0)/10),int(len(l2_11)/10))
        
        x_0=x_0[0:10*len_balance,:,:]
        x_1=x_1[0:len_balance,:,:]
        x_2=x_2[0:len_balance,:,:]
        x_3=x_3[0:len_balance,:,:]
        x_4=x_4[0:len_balance,:,:]
        x_5=x_5[0:len_balance,:,:]
        x_6=x_6[0:len_balance,:,:]
        x_7=x_7[0:len_balance,:,:]
        x_8=x_8[0:len_balance,:,:]
        x_9=x_9[0:len_balance,:,:]
        x_10=x_10[0:len_balance,:,:]
        x_11=x_11[0:10*len_balance,:,:]
        
        l_0=l_0[0:10*len_balance]
        l_1=l_1[0:len_balance]
        l_2=l_2[0:len_balance]
        l_3=l_3[0:len_balance]
        l_4=l_4[0:len_balance]
        l_5=l_5[0:len_balance]
        l_6=l_6[0:len_balance]
        l_7=l_7[0:len_balance]
        l_8=l_8[0:len_balance]
        l_9=l_9[0:len_balance]
        l_10=l_10[0:len_balance]
        l_11=l_11[0:10*len_balance]
        
        l2_0=l2_0[0:10*len_balance]
        l2_1=l2_1[0:len_balance]
        l2_2=l2_2[0:len_balance]
        l2_3=l2_3[0:len_balance]
        l2_4=l2_4[0:len_balance]
        l2_5=l2_5[0:len_balance]
        l2_6=l2_6[0:len_balance]
        l2_7=l2_7[0:len_balance]
        l2_8=l2_8[0:len_balance]
        l2_9=l2_9[0:len_balance]
        l2_10=l2_10[0:len_balance]
        l2_11=l2_11[0:10*len_balance]
        
        X_total=np.concatenate((x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11))
        L_total=np.concatenate((l_0,l_1,l_2,l_3,l_4,l_5,l_6,l_7,l_8,l_9,l_10,l_11))
        L2_total=np.concatenate((l2_0,l2_1,l2_2,l2_3,l2_4,l2_5,l2_6,l2_7,l2_8,l2_9,l2_10,l2_11))
        
    for n in range(X_total.shape[0]):        
        dset = f_data.create_dataset('mydataset/'+str(idx),data=X_total[n,...])
        dset2 = f_label.create_dataset('mydataset2/'+str(idx),data=L_total[n])
        dset3 = f_label2.create_dataset('mydataset2/'+str(idx),data=L2_total[n])
        idx=idx+1
        
    if (mode=='test_kmeans') :
        dset2 = f_label.create_dataset('mydataset2/size',data=idx)   
        idx=0
        f_data.close()
        f_label.close()     
        f_label2.close()              
        
  

if (mode=='test') | (mode == 'data') | (mode == 'data_small') | (mode == 'teta') | (mode== 'data_y2')| (mode== 'test_y2'):
    dset2 = f_label.create_dataset('mydataset2/size',data=idx)   
    f_data.close()
    f_label.close()     
    f_label2.close()      


plt.figure();plt.plot(vad1)
if (mode=='test') | (mode == 'data') | (mode == 'data_small') | (mode == 'test_kmeans') | (mode == 'data_y2') | (mode == 'test_y2'):
    plt.figure();plt.plot(vad2)
plt.figure();plt.plot(L)
plt.figure();plt.plot(L2)
plt.figure();plt.hist(L,3)
plt.figure();plt.hist(L_total,3)
plt.figure();plt.hist(L2_total,12)

plt.figure();
temp=np.log(abs(z_k[0,:,:]));
plt.title('Model -label prediction')
plt.imshow(temp[::-1],aspect='auto')
plt.plot(L)
plt.plot(L2)
plt.show()


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.imshow(temp[::-1],aspect='auto')
ax2.plot(L, 'b-')
ax2.plot(L2, 'g-')
