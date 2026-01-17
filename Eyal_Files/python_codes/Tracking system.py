
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Thu Apr 22 14:40:28 2021

@author: shvarta3
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.io import wavfile
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
from stft import stft
from istft import istft
import multiprocessing
from numpy import linalg as LA
from scipy.io.wavfile import write
import scipy.io as sio
#import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
import logging
from pystoi import stoi
from pesq import pesq
import mir_eval
from scipy import ndimage



plt.close("all")
forlder_to_work1 = 'C:/project/'
RTF_mode = 'GEVD'

who = 'second'

scaler = StandardScaler()
nfft=2048
wlen = 2048                                                                     
hop = wlen/4 
num_classes = 3
NUP=1025                                          
win=np.hamming(wlen)
pad=90
epsilon=0.01
threshold_freq=0.3
threshold=40
frame_before=8
frame_after = 5
win_vad = np.hamming(21)

annot = True
cmap = 'Oranges'
fmt = '.2f'
lw = 0.5
cbar = False
show_null_values = 2
pred_val_axis = 'y'
fz = 9
figsize = [18,18]

plt.close("all")
model3=load_model(forlder_to_work1+'models/model_GEVD_18_separate_17_02.h5')
model18=load_model(forlder_to_work1+'models/model2_GEVD_18_separate_17_02.h5')

y_total=[]
y2_total=[]

y_prob_total_stat=[]
y2_prob_total_stat=[]

folder_to_save = 'C:/project/dynamic_signals/two_speakers_real_recording/results/'
forlder_to_work2 = 'C:/project/dynamic_signals/two_speakers_real_recording/'

signal_file=(forlder_to_work2+'dynamic_signal_1.wav')
signal_first_file=(forlder_to_work2+'dynamic_signal_'+who+'_1.wav')
fs,receivers= wavfile.read(signal_file)
fs,receiver_first = wavfile.read(signal_first_file)
receivers=receivers/(abs(receivers).max())
receiver_first=receiver_first/(abs(receiver_first).max())
M=len(receiver_first[0,:])
index=int(1+np.fix((len(receiver_first[:,1])-wlen)/hop))



angle_location_first=(forlder_to_work2+'angle_location_'+who+'_1.mat')


mat_locations=sio.loadmat(angle_location_first)
location_first=np.transpose(mat_locations['label_'+who])




y2_first = np.zeros((index,1))

for l in range(0,index):
    locations_temp_first = location_first[int(l*hop):int(wlen+l*hop)]
    angle_frame_first = np.bincount(np.squeeze(locations_temp_first)).argmax()
    y2_first[l] = angle_frame_first

z_k = np.zeros((M,NUP,index),dtype=complex)
for i in range(M):
    z_k[i,:,:] = stft(receivers[:,i], win, hop, nfft)

z_k_first = np.zeros((M,NUP,index),dtype=complex)
for i in range(M):
    z_k_first[i,:,:] = stft(receiver_first[:,i], win, hop, nfft)

cholesky_Qvv = np.zeros((NUP,M,M),dtype=complex)
for i in range(0,NUP):
    PSD_tmp =z_k[:,i,450:500]@(z_k[:,i,450:500].conj().T)/50
    cholesky_Qvv[i,:,:] = LA.cholesky(PSD_tmp)

X = np.zeros((index-frame_before-frame_after,NUP,M),dtype=complex)
x_index = 0
for l in range(frame_before,index-frame_after):
    for j in range(NUP):
        chol_j=LA.inv(cholesky_Qvv[j,:,:]+epsilon*np.eye(M)*(LA.norm(cholesky_Qvv[j,:,:])))
        Zvv = 0
        sum_win_vad = 0
        for p in range(frame_before+frame_after+1):
            temp_zvv = chol_j@(win_vad[10-frame_before+p]*z_k[:,j,l-frame_before+p].reshape(M,1))
            Zvv = Zvv+temp_zvv@temp_zvv.conj().T
            sum_win_vad = sum_win_vad+win_vad[10-frame_before+p]
        Zvv=Zvv/sum_win_vad
        w,v = LA.eig(Zvv)
        fi=v[:,w.argmax()].reshape(M,1)
        denominator=cholesky_Qvv[j,0,:].reshape(1,M)@fi
        G_cw=np.squeeze(cholesky_Qvv[j,:,:])@fi/denominator
        X[x_index,j,:] = np.squeeze(G_cw)
    x_index+=1

X_T=np.concatenate((X[:,:,1:M].real,X[:,:,1:M].imag),axis=2)
for b in range(len(X_T)):
    X_T[b,:,:] = scaler.fit_transform(X_T[b,:,:])


z_k_0=z_k[0,:,frame_before:index-frame_after].T
z_k_0=np.log(abs(z_k_0))
z_k_0_standart = scaler.fit_transform(z_k_0)
z_k_0_standart=np.reshape(z_k_0_standart, (z_k_0_standart.shape[0],NUP,1))


x_test=np.concatenate((X_T,z_k_0_standart),axis=2)

vad1_temp=abs(z_k_first)
vad1_temp = vad1_temp/(vad1_temp.std())
vad1_temp = vad1_temp.mean(0)
vad1_temp = vad1_temp > threshold_freq
vad1_temp=vad1_temp.astype(np.int)
vad1_temp_sum=vad1_temp.sum(axis=0)
vad1_temp_sum=vad1_temp_sum.astype(np.int)
vad1_temp_sum1 = vad1_temp_sum > threshold
vad1=vad1_temp_sum1.astype(np.int)

check_vad1=np.zeros(index)

for l in range(frame_before,index-frame_after): 
    check_vad1[l]=vad1[l-1:l+2].sum()

for l in range(frame_before,index-frame_after): 
    if check_vad1[l]==3:
        vad1[l]=1
    else:
        vad1[l]=0

vad1_location_update = np.squeeze(y2_first.T*vad1.T)
L=vad1

y= L[frame_before:index-frame_after]
y2 = np.squeeze(vad1_location_update)   
y2= y2[frame_before:index-frame_after]
y2=np.where(y!=2, y2,19)

z_k= np.transpose(z_k, (2,1,0))
z_k= z_k[frame_before:index-frame_after,:,:]

y_pred=model3.predict(x_test)
y_pred2=pd.DataFrame(y_pred)
y_prob=y_pred2.idxmax(axis=1)
y_prob_stat=y_prob.to_numpy()

y2_pred=model18.predict(x_test)
y2_pred2=pd.DataFrame(y2_pred)
y2_prob=y2_pred2.idxmax(axis=1)
y2_prob_stat=y2_prob.to_numpy()

y_total=np.append(y_total,y)
y2_total=np.append(y2_total,y2)

y_prob_total_stat=np.append(y_prob_total_stat,y_prob_stat)
y2_prob_total_stat=np.append(y2_prob_total_stat,y2_prob_stat)

y2_prob_total_stat_plot=np.delete(y2_prob_total_stat, np.where(y2_total==0)[0])
y2_total_plot=np.delete(y2_total, np.where(y2_total==0))
y2_prob_total_stat_plot=np.delete(y2_prob_total_stat_plot, np.where(y2_total_plot==19))
y2_total_plot=np.delete(y2_total_plot, np.where(y2_total_plot==19))

num_classes=model3.layers[-1].output_shape[1]     
cm_plot_labels = ['Noise','One speacker','2 speackers']
plot_confusion_matrix_from_data(y_total, y_prob_total_stat,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

num_classes=model18.layers[-1].output_shape[1]-2    
cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
                  ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
plot_confusion_matrix_from_data(y2_total_plot-1,y2_prob_total_stat_plot-1,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)



print()
    



