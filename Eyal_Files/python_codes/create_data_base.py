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

#root_dir = '/home/dsi/shvarta3/data_sets/'
#data_root_dir = '/mnt/dsi_vol1/users/ayal_shvarts/project/'

root_dir = 'C:/project/'
data_root_dir = 'C:/project/'

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
indices = [0,1,2,3]
class_wieght=np.zeros(3)
scaler = StandardScaler()
num_diraction=18

if mode =='data':
    data_file_name=root_dir+'data_sets/separate_files/feature_vector_'
    label_file_name=root_dir+'data_sets/separate_files/label_'
    label2_file_name=root_dir+'data_sets/separate_files/label2_'
    idx_file_name=root_dir+'data_sets/separate_files/idx.npy'
    nom_data_sets=2
    lottery=11
             
if mode =='val':
    data_file_name=root_dir+'val_data_sets/separate_files/feature_vector_'
    label_file_name=root_dir+'val_data_sets/separate_files/label_'
    label2_file_name=root_dir+'val_data_sets/separate_files/label2_'
    idx_file_name=root_dir+'val_data_sets/separate_files/idx.npy'
    nom_data_sets=3
    lottery=26

def stft_z(get_receivers):
    return stft(get_receivers, win, hop, nfft)

def create_cholesky_Qvv(z_k_start):
    temp=z_k_start@(z_k_start.conj().T)/pad
    return LA.cholesky(temp)

def create_X(l,j):
    chol_j=LA.inv(cholesky_Qvv[j,:,:])
    Zvv = 0
    sum_win_vad = 0
    for p in range(frame_before+frame_after+1):
        temp_zvv = chol_j@(win_vad[10-frame_before+p]*z_k[:,j,l-frame_before+p].reshape(M,1))
        Zvv = Zvv+temp_zvv@temp_zvv.conj().T
        sum_win_vad = sum_win_vad+win_vad[10-frame_before+p]
    Zvv=Zvv/sum_win_vad
    # a=chol_j@z_k[:,j,l-frame_before:(l+frame_after+1)]
    # Zvv= a@a.conj().T/(frame_before+frame_after+1)
    w,v = LA.eig(Zvv)
    fi=v[:,w.argmax()].reshape(M,1)
    denominator=cholesky_Qvv[j,0,:].reshape(1,M)@fi
    G_cw=np.squeeze(cholesky_Qvv[j,:,:])@fi/denominator
    return np.squeeze(G_cw)

def create_X_total(l):
    X_temp=[]
    X_temp = Parallel(n_jobs=num_cores, verbose=0)(delayed(
        create_X)(l,j)for j in range(NUP))            
    return np.asarray(X_temp)[:,1:M] 

for k in range(1,nom_data_sets):
    print(k)
    idx_start_epoch = idx
    for i in range(start,lottery):
        index_file=i+(lottery-1)*(k-1)
        if mode =='data':
            print(i)
            first_file=(data_root_dir+'DB_dynamic/first_%d.wav'%index_file)
            second_file=(data_root_dir+'DB_dynamic/second_%d.wav'%index_file) 
            together_file=(data_root_dir+'DB_dynamic/together_%d.wav'%index_file) 
            label_first_location_file=(data_root_dir+'DB_dynamic/label_location_first_%d.mat'%index_file)
            label_second_location_file=(data_root_dir+'DB_dynamic/label_location_second_%d.mat'%index_file)
       
        if mode =='val':
            print(i)
            first_file=(data_root_dir+'val_dynamic/first_%d.wav'%index_file)
            second_file=(data_root_dir+'val_dynamic/second_%d.wav'%index_file) 
            together_file=(data_root_dir+'val_dynamic/together_%d.wav'%index_file) 
            label_first_location_file=(data_root_dir+'val_dynamic/label_location_first_%d.mat'%index_file)
            label_second_location_file=(data_root_dir+'val_dynamic/label_location_second_%d.mat'%index_file)


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

        cholesky_Qvv=[]
        cholesky_Qvv = Parallel(n_jobs=num_cores, verbose=0)(delayed(
        create_cholesky_Qvv)(z_k[:,i,0:pad])for i in range(0,NUP))
        cholesky_Qvv=np.asarray(cholesky_Qvv) 

        X_temp_total=[]
        X_temp_total = Parallel(n_jobs=num_cores, verbose=0)(delayed(
        create_X_total)(l)for l in range(frame_before,index-frame_after))
        X=np.asarray(X_temp_total)
        
        X_T=np.concatenate((X.real,X.imag),axis=2)
        for b in range(len(X_T)):
            X_T[b,:,:] = scaler.fit_transform(X_T[b,:,:])
        
        
        z_k_0=z_k[0,:,frame_before:index-frame_after].T
        z_k_0=np.log(abs(z_k_0))
        z_k_0_standart = scaler.fit_transform(z_k_0)
        z_k_0_standart=np.reshape(z_k_0_standart, (z_k_0_standart.shape[0],NUP,1))
        

        X_T=np.concatenate((X_T,z_k_0_standart),axis=2)
        
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
        
        if (mode=='data') | (mode == 'val'):
        
            if i==start:
                X_total=X_T
                L_total=L
                L2_total=L2
            else :
                X_total=np.concatenate((X_total,X_T))
                L_total=np.concatenate((L_total,L))
                L2_total=np.concatenate((L2_total,L2))
        
    if (mode=='test') :
        X_total=X_T
        L_total=L
        L2_total=L2
        
        
    if (mode=='data') | (mode=='val'):
        
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
        x_12=X_total[np.where(L2_total==12)]
        x_13=X_total[np.where(L2_total==13)]
        x_14=X_total[np.where(L2_total==14)]
        x_15=X_total[np.where(L2_total==15)]
        x_16=X_total[np.where(L2_total==16)]
        x_17=X_total[np.where(L2_total==17)]
        x_18=X_total[np.where(L2_total==18)]
        x_19=X_total[np.where(L2_total==19)]

        
        
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
        l_12=L_total[np.where(L2_total==12)]
        l_13=L_total[np.where(L2_total==13)]
        l_14=L_total[np.where(L2_total==14)]
        l_15=L_total[np.where(L2_total==15)]
        l_16=L_total[np.where(L2_total==16)]
        l_17=L_total[np.where(L2_total==17)]
        l_18=L_total[np.where(L2_total==18)]
        l_19=L_total[np.where(L2_total==19)]
        
        
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
        l2_12=L2_total[np.where(L2_total==12)]
        l2_13=L2_total[np.where(L2_total==13)]
        l2_14=L2_total[np.where(L2_total==14)]
        l2_15=L2_total[np.where(L2_total==15)]
        l2_16=L2_total[np.where(L2_total==16)]
        l2_17=L2_total[np.where(L2_total==17)]
        l2_18=L2_total[np.where(L2_total==18)]
        l2_19=L2_total[np.where(L2_total==19)]
        
        
        len_balance=min(len(l2_1),len(l2_2),len(l2_3),len(l2_4),len(l2_5),len(l2_6),len(l2_7),len(l2_8)
                        ,len(l2_9),len(l2_10),len(l2_11),len(l2_12),len(l2_13),len(l2_14),len(l2_15),
                        len(l2_16),len(l2_17),len(l2_18))
        len_balance=min(len_balance,int(len(l2_0)/num_diraction),int(len(l2_19)/num_diraction))
        
        x_0=x_0[0:num_diraction*len_balance,:,:]
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
        x_11=x_11[0:len_balance,:,:]
        x_12=x_12[0:len_balance,:,:]
        x_13=x_13[0:len_balance,:,:]
        x_14=x_14[0:len_balance,:,:]
        x_15=x_15[0:len_balance,:,:]
        x_16=x_16[0:len_balance,:,:]
        x_17=x_17[0:len_balance,:,:]
        x_18=x_18[0:len_balance,:,:]
        x_19=x_19[0:num_diraction*len_balance,:,:]
        
        l_0=l_0[0:num_diraction*len_balance]
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
        l_11=l_11[0:len_balance]
        l_12=l_12[0:len_balance]
        l_13=l_13[0:len_balance]
        l_14=l_14[0:len_balance]
        l_15=l_15[0:len_balance]
        l_16=l_16[0:len_balance]
        l_17=l_17[0:len_balance]
        l_18=l_18[0:len_balance]
        l_19=l_19[0:num_diraction*len_balance]
        
        l2_0=l2_0[0:num_diraction*len_balance]
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
        l2_11=l2_11[0:len_balance]
        l2_12=l2_12[0:len_balance]
        l2_13=l2_13[0:len_balance]
        l2_14=l2_14[0:len_balance]
        l2_15=l2_15[0:len_balance]
        l2_16=l2_16[0:len_balance]
        l2_17=l2_17[0:len_balance]
        l2_18=l2_18[0:len_balance]
        l2_19=l2_19[0:num_diraction*len_balance]
        
        X_total=np.concatenate((x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,x_18,x_19))
        L_total=np.concatenate((l_0,l_1,l_2,l_3,l_4,l_5,l_6,l_7,l_8,l_9,l_10,l_11,l_12,l_13,l_14,l_15,l_16,l_17,l_18,l_19))
        L2_total=np.concatenate((l2_0,l2_1,l2_2,l2_3,l2_4,l2_5,l2_6,l2_7,l2_8,l2_9,
                                  l2_10,l2_11,l2_12,l2_13,l2_14,l2_15,l2_16,l2_17,l2_18,l2_19))
        
    for n in range(X_total.shape[0]):    
        np.save(data_file_name+str(idx)+'.npy' , X_total[n,...])
        np.save(label_file_name+str(idx)+'.npy' , L_total[n])
        np.save(label2_file_name+str(idx)+'.npy' , L2_total[n])
        idx=idx+1
        
np.save(idx_file_name,idx)
plt.figure();plt.plot(vad1)
plt.figure();plt.plot(vad2)
plt.figure();plt.plot(L)
plt.figure();plt.plot(L2)
plt.figure();plt.hist(L,3)
plt.figure();plt.hist(L2,20)
plt.figure();plt.hist(L_total,3)
plt.figure();plt.hist(L2_total,20)

plt.figure();
temp=z_k_0.T;
plt.title('Model -label prediction')
plt.imshow(temp[::-1],aspect='auto')
plt.show()


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.imshow(temp[::-1],aspect='auto')
ax2.plot(L, 'b-')
ax2.plot(L2, 'g-')
