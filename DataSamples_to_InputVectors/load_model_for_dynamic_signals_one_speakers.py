# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:57:44 2019

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
from joblib import Parallel, delayed
import multiprocessing
from numpy import linalg as LA
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  

plt.close("all")
forlder_to_work = 'C:\project'
RTF_mode = 'GEVD'

nom_data_sets=2             ################ amount of signals to test ###################
beta = 0.9
scaler = StandardScaler()
nfft=2048
wlen = 2048                                                                     
hop = wlen/4 
NUP=1025                                        
win=np.hamming(wlen)
num_cores = multiprocessing.cpu_count()
pad=30
e=0.01
epsilon=0.0001
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
labels_location = np.arange(5,185,10)
indices = [0,2,4,6]


y_total=[]
y2_total=[]

y_prob_total=[]
y2_prob_total=[]


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

#'/home/dsi/shvarta3/models/model_GEVD_18_separate_14_03.h5'
plt.close("all")
model3=load_model('/home/dsi/shvarta3/models/model_GEVD_18_separate_22_04.h5')
model18=load_model('/home/dsi/shvarta3/models/model2_GEVD_18_separate_22_04.h5')
accurercy_list=np.zeros(nom_data_sets-1)



k=1
for k in range(1,nom_data_sets):

    print(k)
    signal_file=(forlder_to_work+'dynamic_signals/one_speaker/dynamic_signal_%d.wav'%k)
    signal_clean_file=(forlder_to_work+'dynamic_signals/one_speaker/dynamic_signal_clean_%d.wav'%k)
    angle_location=(forlder_to_work+'dynamic_signals/one_speaker/angle_location_%d.mat'%k)
    sample_position=(forlder_to_work+'dynamic_signals/one_speaker/locations_%d.mat'%k)
    sample_position_small=(forlder_to_work+'dynamic_signals/one_speaker/locations_small_%d.mat'%k)
    mic_array_file=(forlder_to_work+'dynamic_signals/one_speaker/mic_array_%d.mat'%k)
############################### create input to the model ##############################################
    
    sp_path=sio.loadmat(sample_position)
    sp_path=np.transpose(sp_path['sp_path'])

    sp_path_small=sio.loadmat(sample_position_small)
    sp_path_small=np.transpose(sp_path_small['sp_path_small'])
    
    mat_locations=sio.loadmat(angle_location)
    locations=np.transpose(mat_locations['locations'])

    mic_array=sio.loadmat(mic_array_file)
    mic_array=np.transpose(mic_array['r'])
    mic_array = mic_array[:,indices]
    
 
    fs,receivers= wavfile.read(signal_file)
    receivers = receivers[:,indices]
    receivers=receivers/(abs(receivers).max())

    fs,receivers_clean= wavfile.read(signal_clean_file)
    receivers_clean = receivers_clean[:,indices]
    receivers_clean=receivers_clean/(abs(receivers_clean).max())

    M=len(receivers[0,:])
    index=int(1+np.fix((len(receivers[:,1])-wlen)/hop))

    xlen = len(locations)
    wlen = len(win) 
    y2 = np.zeros((index,1))
    sp_path_z = np.zeros((index,3))
    sp_path_small_z = np.zeros((index,3))
    for l in range(0,index):
        locations_temp = locations[int(l*hop):int(wlen+l*hop)]      
        angle_frame = np.bincount(np.squeeze(locations_temp)).argmax()
        y2[l] = np.argmin(abs(labels_location-angle_frame))+1
        
        sp_path_temp = sp_path[:,int(l*hop):int(wlen+l*hop)]
        sp_path_z[l,:] = np.mean(sp_path_temp,axis=1)
        sp_path_small_temp = sp_path_small[:,int(l*hop):int(wlen+l*hop)]
        sp_path_small_z[l,:] = np.mean(sp_path_small_temp,axis=1)

    z_k=[]
    z_k = Parallel(n_jobs=1, verbose=0)(delayed(
    stft_z)(receivers[:,i])for i in range(M))
    z_k=np.asarray(z_k)

    z_k_clean=[]
    z_k_clean = Parallel(n_jobs=1, verbose=0)(delayed(
    stft_z)(receivers_clean[:,i])for i in range(M))
    z_k_clean=np.asarray(z_k_clean)

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
    

    x_test=np.concatenate((X_T,z_k_0_standart),axis=2)

    
############################### source separation #####################################

    vad1_temp=abs(z_k_clean)
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
            
    vad1= vad1[frame_before:index-frame_after]



############################### create actual vad #####################################

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
 
    y2 = y2[frame_before:index-frame_after]
    
################################### save prediction to confusion matrix ######################


    y2_total=np.append(y2_total,y2)
    
    y_prob_total=np.append(y_prob_total,y_prob_stat)
    y2_prob_total=np.append(y2_prob_total,y2_prob_stat)
    
############################ scatter plot #############################################################    
    

sp_path_z = sp_path_z[frame_before:index-frame_after]
sp_path_small_z = sp_path_small_z[frame_before:index-frame_after]

y2_prob_stat=np.where(y_prob_stat!=0, np.squeeze(y2_prob_stat),np.nan)
y2_prob_stat=np.where(y_prob_stat!=2, np.squeeze(y2_prob_stat),np.nan)

y2_with_vad1=np.where(vad1!=0, np.squeeze(y2),np.nan)

plt.figure()
plt.plot(y2_with_vad1,'ro',markersize=4)
plt.plot(y2_prob_stat,'bo',markersize=2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('DOA prediction of dynamic speaker')
scatter1 = ax.scatter3D(sp_path_small_z[:,0], sp_path_small_z[:,1], sp_path_small_z[:,2]
                        ,c=y2_with_vad1, cmap='hsv',s=5,label = 'Actual DOA activity')
ax.scatter3D(sp_path_z[:,0], sp_path_z[:,1], sp_path_z[:,2],c=y2_prob_stat, cmap='hsv',s=30,label = 'Predicted DOA activity')
cbar = fig.colorbar(scatter1, ax=ax)
cbar.set_label('time(sec)', labelpad=-17, y=1.05, rotation=0)
cbar.set_ticks(list())
for index, label in enumerate(["- 1", "- 2", "- 3", "- 4","- 5","- 6", "- 7", "- 8","- 9","- 10"]):
    x = 18.0
    y = (12 * index + 11) / 7
    cbar.ax.text(x, y, label)

ax.scatter3D(mic_array[0,:], mic_array[1,:],mic_array[2,:],'red',marker='X',label = "Microphones Array")
plt.legend(loc="upper right")
############################ confusion matrix static ########################################### 


y2_total=np.delete(y2_total, np.where(y_prob_total==0))
y2_prob_total=np.delete(y2_prob_total, np.where(y_prob_total==0))
y_prob_total=np.delete(y_prob_total, np.where(y_prob_total==0))

y2_total=np.delete(y2_total, np.where(y_prob_total==2))
y2_prob_total=np.delete(y2_prob_total, np.where(y_prob_total==2))

############################ confusion matrix dynamic ########################################### 

num_classes=model18.layers[-1].output_shape[1]-2    
cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
                  ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
plot_confusion_matrix_from_data(y2_total-1,y2_prob_total-1,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

