
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
import torch
import pysepm 

A = -17.49
B = 9.69
def mapping_stoi(d,a=A,b=B):
        return 100/(1+np.exp(a*d+b))

def si_sdr_torchaudio_calc(estimate, reference, epsilon=1e-8):
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
        mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
        scale = mix_pow / (reference_pow + epsilon)
        reference = scale * reference
        error = estimate - reference
        reference_pow = reference.pow(2)
        error_pow = error.pow(2)
        reference_pow = reference_pow.mean(axis=1)
        error_pow = error_pow.mean(axis=1)
        sisdr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
        return torch.mean(sisdr)



plt.close("all")
forlder_to_work1 = 'C:/project/'
RTF_mode = 'GEVD'

nom_data_sets=11          ################ amount of signals to test ###################
scaler = StandardScaler()
nfft=2048
wlen = 2048                                                                     
hop = wlen/4 
num_classes = 3
new = 1
img_rows, img_cols = 1025,7
NUP=1025                                          
win=np.hamming(wlen)
num_cores = multiprocessing.cpu_count()
pad=60
e=0.01
epsilon=0.01
num_speech=2
frame_threshold=9
threshold_freq=0.3
threshold=40
alfa_Qvv = 0.99
last_update_first = 0
last_update_second = 0
alfa_G = 1
threshold_chage_location = 5
frame_before=8
frame_after = 5
win_vad = np.hamming(21)
indices = [0,1,2,3]
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

time_second =0
time_first =0
first_speaker_active = 0
second_speaker_active = 0

plt.close("all")
model3=load_model(forlder_to_work1+'models/model_GEVD_18_separate_17_02.h5')
model18=load_model(forlder_to_work1+'models/model2_GEVD_18_separate_17_02.h5')
accurercy_list=np.zeros(nom_data_sets-1)

y_total=[]
y2_total=[]

y_prob_total_stat=[]
y2_prob_total_stat=[]

y_prob_total_dync=[]
y2_prob_total_dync=[]

separation_total = np.array([])
noise_separation_total = np.array([])

stoi_total = np.array([])
pesq_total = np.array([])

stoi_total_noisy = np.array([])
pesq_total_noisy = np.array([])
type_r = 'SNR'
#folder_names = ['10'] 
folder_names = ['20','17.5','15','12.5','10']
#folder_names = ['300','400','500']
#folder_names = ['20']
folder_to_save = 'C:/project/dynamic_signals/two_speakers_WSJ/SNR/'

flag_start_lcmv = 1

if new:
    stoi_alg = np.zeros((len(folder_names),nom_data_sets-1))
    stoi_noisy = np.zeros((len(folder_names),nom_data_sets-1))
  
    pesq_alg = np.zeros((len(folder_names),nom_data_sets-1))
    pesq_noisy = np.zeros((len(folder_names),nom_data_sets-1))

    sir_noisy = np.zeros((len(folder_names),nom_data_sets-1))
    sir_alg = np.zeros((len(folder_names),nom_data_sets-1))
    
    sdr_noisy = np.zeros((len(folder_names),nom_data_sets-1))
    sdr_alg = np.zeros((len(folder_names),nom_data_sets-1))

    sisdr_alg = np.zeros((len(folder_names),nom_data_sets-1))
    sisdr_noisy = np.zeros((len(folder_names),nom_data_sets-1))
    
else:
    stoi_alg=np.load(folder_to_save+'stoi_alg.npy')
    stoi_noisy=np.load(folder_to_save+'stoi_noisy.npy')
    pesq_alg=np.load(folder_to_save+'pesq_alg.npy')
    pesq_noisy=np.load(folder_to_save+'pesq_noisy.npy')
    sir_alg=np.load(folder_to_save+'sir_alg.npy')
    sir_noisy=np.load(folder_to_save+'sir_noisy.npy')
    sdr_alg=np.load(folder_to_save+'sdr_alg.npy')
    sdr_noisy=np.load(folder_to_save+'sdr_noisy.npy')
    sisdr_alg=np.load(folder_to_save+'sisdr_alg.npy')
    sisdr_noisy=np.load(folder_to_save+'sisdr_noisy.npy')

flag_start_lcmv = 1
for d in range(0,len(folder_names)):
    forlder_to_work2 = ('C:/project/dynamic_signals/two_speakers_WSJ/SNR/SNR_%s_T60_300_SIR_0/'%folder_names[d])
    for k in range(1,nom_data_sets):
        flag_start_lcmv = 1
            
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=forlder_to_work2+'/results/log_files/info_%d.log'%k,level=logging.DEBUG)    
        
        print(k)
        signal_file=(forlder_to_work2+'dynamic_signal_%d.wav'%k)
    
        signal_first_file=(forlder_to_work2+'dynamic_signal_first_%d.wav'%k)
        signal_second_file=(forlder_to_work2+'dynamic_signal_second_%d.wav'%k)
        
        angle_location_first=(forlder_to_work2+'angle_location_first_%d.mat'%k)
        angle_location_Second=(forlder_to_work2+'angle_location_second_%d.mat'%k)
        
        signal_noise_file=(forlder_to_work2+'dynamic_signal_noise_%d.wav'%k)
    

        ############################### create input to the model ##############################################
        
        mat_locations=sio.loadmat(angle_location_first)
        location_first=np.transpose(mat_locations['location_first'])
        mat_locations=sio.loadmat(angle_location_Second)
        location_second=np.transpose(mat_locations['location_second'])
    
    
    
        fs,receiver_first = wavfile.read(signal_first_file)
        receiver_first = receiver_first[:,indices]
        fs,receiver_second = wavfile.read(signal_second_file)
        receiver_second = receiver_second[:,indices]        
        fs,receivers= wavfile.read(signal_file)
        receivers = receivers[:,indices]
    
        
        receiver_first=receiver_first/(abs(receiver_first).max())
        receiver_second=receiver_second/(abs(receiver_second).max())
        receivers=receivers/(abs(receivers).max())
    
        fs,noise = wavfile.read(signal_noise_file)
        noise = noise[:,indices]
        noise = noise/(abs(noise).max())
    
    
        M=len(receiver_first[0,:])
        index=int(1+np.fix((len(receiver_first[:,1])-wlen)/hop))
    
    ############################## create y2 from location file #############################
    
        y2_first = np.zeros((index,1))
        y2_second = np.zeros((index,1))
        
        for l in range(0,index):
            locations_temp_first = location_first[int(l*hop):int(wlen+l*hop)]
            locations_temp_second = location_second[int(l*hop):int(wlen+l*hop)]
            
            angle_frame_first = np.bincount(np.squeeze(locations_temp_first)).argmax()
            angle_frame_second = np.bincount(np.squeeze(locations_temp_second)).argmax()
            
            y2_first[l] = np.argmin(abs(labels_location-angle_frame_first))+1
            y2_second[l] = np.argmin(abs(labels_location-angle_frame_second))+1
    
        z_k = np.zeros((M,NUP,index),dtype=complex)
        for i in range(M):
            z_k[i,:,:] = stft(receivers[:,i], win, hop, nfft)
        
        z_k_noise = np.zeros((M,NUP,index),dtype=complex)
        for i in range(M):
            z_k_noise[i,:,:] = stft(noise[:,i], win, hop, nfft)

        z_k_first = np.zeros((M,NUP,index),dtype=complex)
        for i in range(M):
            z_k_first[i,:,:] = stft(receiver_first[:,i], win, hop, nfft)
            
        z_k_second = np.zeros((M,NUP,index),dtype=complex)
        for i in range(M):
            z_k_second[i,:,:] = stft(receiver_second[:,i], win, hop, nfft)
     
        cholesky_Qvv = np.zeros((NUP,M,M),dtype=complex)
        for i in range(0,NUP):
            PSD_tmp =z_k[:,i,0:pad]@(z_k[:,i,0:pad].conj().T)/pad
            cholesky_Qvv[i,:,:] = LA.cholesky(PSD_tmp)
        
        X = np.zeros((index-frame_before-frame_after,NUP,M),dtype=complex)
        x_index = 0
        for l in range(frame_before,index-frame_after):
            for j in range(NUP):
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
        vad1_temp=(vad1_temp).astype(np.int)
        vad1_temp_sum=vad1_temp.sum(axis=0)
        vad1_temp_sum=(vad1_temp_sum).astype(np.int)
        vad1_temp_sum1 = vad1_temp_sum > threshold
        vad1=(vad1_temp_sum1).astype(np.int)
    
       
        vad2_temp=abs(z_k_second)
        vad2_temp = vad2_temp/(vad2_temp.std())
        vad2_temp = vad2_temp.mean(0)
        vad2_temp = vad2_temp > threshold_freq
        vad2_temp=(vad2_temp).astype(np.int)
        vad2_temp_sum=vad2_temp.sum(axis=0)
        vad2_temp_sum=(vad2_temp_sum).astype(np.int)
        vad2_temp_sum1 = vad2_temp_sum > threshold
        vad2=(vad2_temp_sum1).astype(np.int)
    
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
        
        ############# fix direction ####################

        y2_second_roll = np.roll(y2_second, 30)
        vad1_location_update = np.squeeze(y2_first.T*vad1.T)
        vad2_location_update = np.squeeze(y2_second_roll.T*vad2.T)
        L=vad1+vad2
        y= L[frame_before:index-frame_after]
        y2 = np.squeeze(vad1_location_update+vad2_location_update)   
        y2= y2[frame_before:index-frame_after]
        y2=np.where(y!=2, y2,19)
        
    ############################### source separation #####################################
    
        z_k= np.transpose(z_k, (2,1,0))
        z_k= z_k[frame_before:index-frame_after,:,:]
        
        
#        z_k_ilrma_first= z_k_ilrma_first[frame_before:index-frame_after]
#        z_k_ilrma_second= z_k_ilrma_second[frame_before:index-frame_after]
        
        z_k_first= z_k_first[:,:,frame_before:index-frame_after]
        z_k_second= z_k_second[:,:,frame_before:index-frame_after]
        z_k_noise= z_k_noise[:,:,frame_before:index-frame_after]
           
        flag_first_noise = 0
        Frame_classification_system = np.zeros((2,3))
        G = np.ones((NUP,M,2),dtype=complex)
#        G_stand = np.ones((NUP,M)) + np.ones((NUP,M)) * 1j
        
        y_prob_dynamic = np.zeros(len(x_test))
        y2_prob_dynamic = np.zeros(len(x_test))    
        Qvv_temp=np.zeros((NUP,M,M),complex)
        
        
        y_pred=model3.predict(x_test)
        y_pred2=pd.DataFrame(y_pred)
        y_prob=y_pred2.idxmax(axis=1)
        y_prob_stat=y_prob.to_numpy()
        
        y2_pred=model18.predict(x_test)
        y2_pred2=pd.DataFrame(y2_pred)
        y2_prob=y2_pred2.idxmax(axis=1)
        y2_prob_stat=y2_prob.to_numpy()
        
        flag_last_y_prob=0
        stand_z = []
        
        Qvv = np.zeros((NUP,M,M))
        for j in range(NUP):
            Qvv[j,:,:] = np.eye(M)
#        Qyy_stand = np.zeros((NUP,M,M),complex)
#        Qyy_first = np.zeros((NUP,M,M),complex)
#        Qyy_second = np.zeros((NUP,M,M),complex)
            
        W=np.ones((len(x_test),NUP,M,2),complex)
    
        PSD_matrix_per_DOA = np.zeros((18,NUP,M,M),complex)
        total_frame_per_DOA = np.zeros((18))
        

        save_last_frames_first = np.zeros((32,1025,4),dtype=complex)
        save_last_frames_doa_first = np.zeros(32)
        save_last_frames_second = np.zeros((32,1025,4),dtype=complex)
        save_last_frames_doa_second = np.zeros(32)


        for l in range(len(x_test)): 
              
            y_pred=model3.predict(x_test[l].reshape(1,NUP,img_cols))
            y_pred2=pd.DataFrame(y_pred)
            y_prob=y_pred2.idxmax(axis=1)
            y_prob=y_prob.to_numpy()
            
            
            logging.info('frame number %d CSD association: %d'%(l,y_prob[0]))
            
            if y_prob[0]==0:
    #            if flag_last_y_prob==0:
                    current_y2_label = 0
                    current_y_label = 0
                    time_second +=1
                    time_first +=1
                    
                    for j in range(0,NUP):
                        Qvv_temp[j,:,:] = z_k[l,j,:].reshape(M,1)@(z_k[l,j,:].reshape(M,1).conj().T)
                        
                    if flag_first_noise==0:
                        flag_first_noise = 1
                        Qvv = Qvv_temp
                        sum_Qvv =1
                    else:
                        sum_Qvv +=1
                        alfa_Qvv = 0.05
                        Qvv = (1-alfa_Qvv)*Qvv+(alfa_Qvv)*Qvv_temp

            if y_prob[0]==1:

                    y2_pred=model18.predict(x_test[l].reshape(1,NUP,img_cols))
                    y2_pred2=pd.DataFrame(y2_pred)
                    y2_prob=y2_pred2.idxmax(axis=1)
                    y2_prob=y2_prob.to_numpy()
                    y2_prob = y2_prob[0]

                    if abs(y2_prob - Frame_classification_system[0,0])<2:
                        if Frame_classification_system[0,0]!=0:
                            save_last_frames_first = np.roll(save_last_frames_first,1,axis=0)
                            save_last_frames_first[0] = z_k[l]
                            save_last_frames_doa_first = np.roll(save_last_frames_doa_first,1)
                            save_last_frames_doa_first[0] = y2_prob

                    elif abs(y2_prob - Frame_classification_system[0,1])<2:
                        if Frame_classification_system[0,0]!=0:
                            save_last_frames_second = np.roll(save_last_frames_second,1,axis=0)
                            save_last_frames_second[0] = z_k[l]
                            save_last_frames_doa_second = np.roll(save_last_frames_doa_second,1)
                            save_last_frames_doa_second[0] = y2_prob

                    if y2_prob == Frame_classification_system[0,0]:
                        
                        current_save_last_frames_first = save_last_frames_first[np.where(save_last_frames_doa_first>0)]
                        current_save_last_frames_doa_first = save_last_frames_doa_first[np.where(save_last_frames_doa_first>0)]

                        current_save_last_frames_first = current_save_last_frames_first[np.where(abs(current_save_last_frames_doa_first-y2_prob)!=0)]
                        current_save_last_frames_doa_first = save_last_frames_doa_first[np.where(abs(current_save_last_frames_doa_first-y2_prob)!=0)]

                        current_save_last_frames_first = current_save_last_frames_first[np.where(abs(current_save_last_frames_doa_first-y2_prob)<2)]
                        
                        Frame_classification_system[1,0]=Frame_classification_system[1,0]+1
                        Frame_classification_system[1,2]=0
                        Frame_classification_system[0,2]=0
                        time_first = 0
                        time_second += 1

                        total_frame_per_DOA[y2_prob-1] += 1
                        alfa_G = (1+len(current_save_last_frames_first))/(total_frame_per_DOA[y2_prob-1]+len(current_save_last_frames_first))
                        
                        for j in range(NUP):
                            if len(current_save_last_frames_first):
                                current_frames = np.concatenate((z_k[l,j,:].reshape(1,M).T,current_save_last_frames_first[:,j,:].reshape(len(current_save_last_frames_first),M).T),axis = 1)
                            else:
                                current_frames = z_k[l,j,:].reshape(1,M).T

                            if RTF_mode=='GEVD':
                                cholesky_Qvv=LA.cholesky(Qvv[j,:,:])
                                chol_j=LA.inv(cholesky_Qvv+epsilon*np.eye(M)*(LA.norm(cholesky_Qvv)))
                                a=chol_j@current_frames
                                Zvv_temp = a@a.conj().T
                                PSD_matrix_per_DOA[y2_prob-1,j,:,:] = (1-alfa_G)*PSD_matrix_per_DOA[y2_prob-1,j,:,:]+(alfa_G)*Zvv_temp
                                w,v = LA.eig(PSD_matrix_per_DOA[y2_prob-1,j,:,:])
                                phi=v[:,w.argmax()].reshape(M,1)
                                denominator=cholesky_Qvv[0,:].reshape(1,M)@phi
                                G[j,:,0]=np.squeeze(cholesky_Qvv@phi/denominator)        
                        current_y2_label = y2_prob

                    elif y2_prob == Frame_classification_system[0,1]:
                        Frame_classification_system[1,1]=Frame_classification_system[1,1]+1
                        Frame_classification_system[1,2]=0
                        Frame_classification_system[0,2]=0
                        time_second = 0
                        time_first +=1
                        
                        current_save_last_frames_second = save_last_frames_second[np.where(save_last_frames_doa_second>0)]
                        current_save_last_frames_doa_second = save_last_frames_doa_second[np.where(save_last_frames_doa_second>0)]

                        current_save_last_frames_second = current_save_last_frames_second[np.where(abs(current_save_last_frames_doa_second-y2_prob)!=0)]
                        current_save_last_frames_doa_second = save_last_frames_doa_second[np.where(abs(current_save_last_frames_doa_second-y2_prob)!=0)]

                        current_save_last_frames_second = current_save_last_frames_second[np.where(abs(current_save_last_frames_doa_second-y2_prob)<3)]
                        

                        total_frame_per_DOA[y2_prob-1] += 1
                        alfa_G = (1+len(current_save_last_frames_second))/(total_frame_per_DOA[y2_prob-1]+len(current_save_last_frames_second))
                        
                        for j in range(NUP):
                            if len(current_save_last_frames_second):
                                current_frames = np.concatenate((z_k[l,j,:].reshape(1,M).T,current_save_last_frames_second[:,j,:].reshape(len(current_save_last_frames_second),M).T),axis = 1)
                            else:
                                current_frames = z_k[l,j,:].reshape(1,M).T
                            if RTF_mode=='GEVD':
                                cholesky_Qvv=LA.cholesky(Qvv[j,:,:])
                                chol_j=LA.inv(cholesky_Qvv+epsilon*np.eye(M)*(LA.norm(cholesky_Qvv)))
                                a=chol_j@current_frames
                                Zvv_temp = a@a.conj().T
                                PSD_matrix_per_DOA[y2_prob-1,j,:,:] = (1-alfa_G)*PSD_matrix_per_DOA[y2_prob-1,j,:,:]+(alfa_G)*Zvv_temp
                                
                                w,v = LA.eig(PSD_matrix_per_DOA[y2_prob-1,j,:,:])
                                phi=v[:,w.argmax()].reshape(M,1)
                                denominator=cholesky_Qvv[0,:].reshape(1,M)@phi
                                G[j,:,1]=np.squeeze(cholesky_Qvv@phi/denominator)
                                
                        current_y2_label = y2_prob

                    elif y2_prob == Frame_classification_system[0,2]:
                        stand_z = np.concatenate((stand_z,z_k[l,:,:].reshape(1,NUP,M)))
                        Frame_classification_system[1,2]=Frame_classification_system[1,2]+1 
                        time_second +=1
                        time_first +=1
                        
                        if Frame_classification_system[1,2]>(threshold_chage_location-1):
                            current_y2_label = y2_prob
                            for j in range(NUP):
                                if RTF_mode=='GEVD':
                                    cholesky_Qvv=LA.cholesky(Qvv[j,:,:])
                                    chol_j=LA.inv(cholesky_Qvv+epsilon*np.eye(M)*(LA.norm(cholesky_Qvv)))
                                    a=chol_j@stand_z[:,j,:].T
                                    Zvv_temp = a@a.conj().T/threshold_chage_location
                                    temp_alfa = total_frame_per_DOA[y2_prob-1]+threshold_chage_location
                                    PSD_matrix_per_DOA[y2_prob-1,j,:,:] = total_frame_per_DOA[y2_prob-1]/temp_alfa*PSD_matrix_per_DOA[y2_prob-1,j,:,:]+threshold_chage_location/temp_alfa*Zvv_temp
                            if Frame_classification_system[1,0] == 0:
                                Frame_classification_system[0,0] = y2_prob
                                Frame_classification_system[1,0] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
                            elif (Frame_classification_system[1,1] == 0) & (abs(Frame_classification_system[0,0]-y2_prob)<4):
                                Frame_classification_system[0,0] = y2_prob
                                Frame_classification_system[1,0] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
                            elif Frame_classification_system[1,1] == 0: 
                                Frame_classification_system[0,1] = y2_prob
                                Frame_classification_system[1,1] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
                            else:     
                                to_change = np.argmin(np.abs(np.array((y2_prob,y2_prob)) - Frame_classification_system[0,0:2]))
                                min1,min2 = np.abs(np.array((y2_prob,y2_prob)) - Frame_classification_system[0,0:2])

                                if (min1>(threshold_chage_location-2)) & (min2>(threshold_chage_location-2)):
                                    if (time_first-min1*30)>(time_second-min2*30):
                                        first_speaker_active = 1
                                        second_speaker_active = 1
                                        to_change = 0
                                    else: 
                                        first_speaker_active = 1
                                        second_speaker_active = 1
                                        to_change = 1
                                    
                                        
                                Frame_classification_system[0,to_change] = y2_prob
                                Frame_classification_system[1,to_change] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
                            total_frame_per_DOA[y2_prob-1] += threshold_chage_location
                            
                    else:
                        stand_z = z_k[l,:,:].reshape(1,NUP,M)
                        Frame_classification_system[0,2] = y2_prob
                        Frame_classification_system[1,2] = 1
                        time_second +=1
                        time_first +=1

                    logging.info('Frame DOAc association : %d'%y2_prob)
                    logging.info('Frame classification system:')
                    logging.info(Frame_classification_system[0,:])
                    logging.info(Frame_classification_system[1,:])

            if y_prob==2:
                    first_speaker_active = 1
                    second_speaker_active = 1
                    time_second +=1
                    time_first +=1
            
    
    ################### source separation ###################################################
        
            
            if (Frame_classification_system[0,0]==0) & (Frame_classification_system[0,1]==0):
                s_hat=z_k[l,:,0]
                s_hat = np.concatenate((s_hat.reshape(1,NUP),1e-10*np.ones((1,NUP))))
            elif (Frame_classification_system[0,0]!=0) & (Frame_classification_system[0,1]==0):
                s_hat = np.zeros(NUP,complex)
                for j in range(NUP):
                    g=G[j,:,0]
                    inv_Qvv=LA.inv(Qvv[j,:,:]+e*LA.norm(Qvv[j,:,:])*np.eye(M))
                    c=inv_Qvv@g
                    inv_temp=g.conj().T@c+epsilon
                    W[l,j,:,0]=c/inv_temp
                    W[l,j,:,1]=c/inv_temp
                    s_hat[j]=W[l,j,:,0].conj().T@z_k[l,j,:]
                    
                s_hat = np.concatenate((s_hat.reshape(1,NUP),s_hat.reshape(1,NUP)))
            elif (Frame_classification_system[0,0]!=0) & (Frame_classification_system[0,1]!=0):
                if flag_start_lcmv:
                    start_lcmv = l
                    flag_start_lcmv = 0
                    
                s_hat = np.zeros((2,NUP),complex)
                for j in range(NUP):
                    g=G[j,:,:]
                    inv_b=LA.inv(Qvv[j,:,:]+e*LA.norm(Qvv[j,:,:])*np.eye(M))
                    c=inv_b@g
                    inv_temp=LA.inv(g.conj().T@c+e*LA.norm(g.conj().T@c)*np.eye(num_speech))
                    W[l,j,:,:]=c@inv_temp
                    s_hat[:,j]=W[l,j,:,:].conj().T@z_k[l,j,:]
                    
                s_hat[0,:] = s_hat[0,:]*first_speaker_active
                s_hat[1,:] = s_hat[1,:]*second_speaker_active
                
            if l==0:
                s_hat_total = s_hat.T.reshape(1,NUP,num_speech)
            else:
                s_hat_total = np.concatenate((s_hat_total,s_hat.T.reshape(1,NUP,num_speech)),axis=0)
        
    ################################### bss eval ################################
        start_overlap = np.nonzero(y == 2)[0][0]
        finish_overlap = np.nonzero(y == 2)[0][-1]

        s_hat_total_for_test = s_hat_total[start_overlap:finish_overlap]
        z_k_first_one_speaker = z_k_first.T[start_overlap:finish_overlap]
        z_k_second_one_speaker = z_k_second.T[start_overlap:finish_overlap]
        z_k_noise_one_speaker = z_k_noise.T[start_overlap:finish_overlap]
        z_k_noisy = z_k[start_overlap:finish_overlap]
        
        s_hat_first_one_speaker_time,_=istft(s_hat_total_for_test[:,:,0].T, win, win, hop, nfft, fs)
        s_hat_second_one_speaker_time,_=istft(s_hat_total_for_test[:,:,1].T, win, win, hop, nfft, fs)

        z_k_first_one_speaker_time,_=istft(z_k_first_one_speaker[:,:,0].T, win, win, hop, nfft, fs)
        z_k_second_one_speaker_time,_=istft(z_k_second_one_speaker[:,:,0].T, win, win, hop, nfft, fs)

        z_k_noise_time,_=istft(z_k_noisy[:,:,0].T, win, win, hop, nfft, fs)

        noisy_sources = np.concatenate((z_k_noise_time.reshape(z_k_noise_time.shape[0],1),z_k_noise_time.reshape(z_k_noise_time.shape[0],1)),axis=1)

        reference_sources = np.concatenate((z_k_first_one_speaker_time.reshape(z_k_first_one_speaker_time.shape[0],1),z_k_second_one_speaker_time.reshape(z_k_second_one_speaker_time.shape[0],1)),axis=1)
        estimated_sources = np.concatenate((s_hat_first_one_speaker_time.reshape(s_hat_first_one_speaker_time.shape[0],1),s_hat_second_one_speaker_time.reshape(s_hat_second_one_speaker_time.shape[0],1)),axis=1)

        (sdr_noise,sir_noise,sar_noise,perm)  = mir_eval.separation.bss_eval_sources(reference_sources.T+10**(-9), noisy_sources.T, compute_permutation=True)

        sir_noisy[d,k-1] = sir_noise.mean()
        sdr_noisy[d,k-1] = sdr_noise.mean()
        
    ###################################### alg results ##################################################3    
        
        (sdr,sir,sar,perm)  = mir_eval.separation.bss_eval_sources(reference_sources.T+10**(-9), estimated_sources.T, compute_permutation=True)

        sir_alg[d,k-1] = sir.mean()
        sdr_alg[d,k-1] = sdr.mean()
        
        first_stoi = max(stoi(z_k_first_one_speaker_time,s_hat_first_one_speaker_time,fs),stoi(z_k_first_one_speaker_time,s_hat_second_one_speaker_time,fs))
        second_stoi = max(stoi(z_k_second_one_speaker_time,s_hat_first_one_speaker_time,fs),stoi(z_k_second_one_speaker_time,s_hat_second_one_speaker_time,fs))

        first_stoi_noisy = stoi(z_k_first_one_speaker_time,z_k_noise_time,fs)
        second_stoi_noisy = stoi(z_k_second_one_speaker_time,z_k_noise_time,fs)

        first_pesq = max(pysepm.pesq(z_k_first_one_speaker_time,s_hat_first_one_speaker_time,fs)[1],pysepm.pesq(z_k_first_one_speaker_time,s_hat_second_one_speaker_time,fs)[1])
        second_pesq = max(pysepm.pesq(z_k_second_one_speaker_time,s_hat_first_one_speaker_time,fs)[1],pysepm.pesq(z_k_second_one_speaker_time,s_hat_second_one_speaker_time,fs)[1])

        first_pesq_noisy = pysepm.pesq(z_k_first_one_speaker_time,z_k_noise_time,fs)[1]
        second_pesq_noisy = pysepm.pesq(z_k_second_one_speaker_time,z_k_noise_time,fs)[1]

        stoi_alg[d,k-1] = (first_stoi+second_stoi)/2
        pesq_alg[d,k-1] = (first_pesq+second_pesq)/2   
        stoi_noisy[d,k-1] = (first_stoi_noisy+second_stoi_noisy)/2
        pesq_noisy[d,k-1]  = (first_pesq_noisy+second_pesq_noisy)/2


        sisdr_noisy[d,k-1] = (si_sdr_torchaudio_calc(z_k_noise_time, z_k_first_one_speaker_time)+si_sdr_torchaudio_calc(z_k_noise_time, z_k_second_one_speaker_time))/2
        si_sdr_pred_first = torch.max(si_sdr_torchaudio_calc(s_hat_first_one_speaker_time, z_k_first_one_speaker_time),si_sdr_torchaudio_calc(s_hat_second_one_speaker_time, z_k_first_one_speaker_time))
        si_sdr_pred_second = torch.max(si_sdr_torchaudio_calc(s_hat_first_one_speaker_time, z_k_second_one_speaker_time),si_sdr_torchaudio_calc(s_hat_second_one_speaker_time, z_k_second_one_speaker_time))
        sisdr_alg[d,k-1] = (si_sdr_pred_second+si_sdr_pred_first)/2
    ################################### save prediction to confusion matrix ######################
    
        y_total=np.append(y_total,y)
        y2_total=np.append(y2_total,y2)
        
        y_prob_total_stat=np.append(y_prob_total_stat,y_prob_stat)
        y2_prob_total_stat=np.append(y2_prob_total_stat,y2_prob_stat)
        
        y_prob_total_dync=np.append(y_prob_total_dync,y_prob_dynamic)
        y2_prob_total_dync=np.append(y2_prob_total_dync,y2_prob_dynamic)
        
        np.save(folder_to_save+'stoi_alg.npy', stoi_alg)
        np.save(folder_to_save+'stoi_noisy.npy', stoi_noisy)
        np.save(folder_to_save+'pesq_alg.npy', pesq_alg)
        np.save(folder_to_save+'pesq_noisy.npy', pesq_noisy)
        np.save(folder_to_save+'sir_alg.npy', sir_alg)
        np.save(folder_to_save+'sir_noisy.npy', sir_noisy)
        np.save(folder_to_save+'sdr_alg.npy', sdr_alg)
        np.save(folder_to_save+'sdr_noisy.npy', sdr_noisy)   
        np.save(folder_to_save+'sisdr_alg.npy', sisdr_alg)
        np.save(folder_to_save+'sisdr_noisy.npy', sisdr_noisy)   


#        s_hat_total_for_test = s_hat_total[np.where(y==2)]
#        z_k_first_one_speaker = z_k_first.T[np.where(y==2)]
#        z_k_second_one_speaker = z_k_second.T[np.where(y==2)]
#        z_k_noise_one_speaker = z_k_noise.T[np.where(y==2)]
#        z_k_noisy = z_k[np.where(y==2)]
        
        for p in range(0,num_speech):
            speech,t1=istft(s_hat_total[:,:,p].T, win, win, hop, nfft, fs)  
            # scaled = speech/np.max(speech)
            file_temp=(forlder_to_work2+'results/separating_speaker_number_{}_{}.wav'.format(p,k))
            write(file_temp, fs,speech)
    
        speech,t1=istft(z_k[:,:,0].T, win, win, hop, nfft, fs)  
        file_temp=(forlder_to_work2+'results/noisy_signal_%d.wav'%k)
        write(file_temp, fs,speech)
        
        speech,t1=istft(z_k_first.T[:,:,0].T, win, win, hop, nfft, fs)  
        file_temp=(forlder_to_work2+'results/first_clean_%d.wav'%k)
        write(file_temp, fs,speech)
        
        speech,t1=istft(z_k_second.T[:,:,0].T, win, win, hop, nfft, fs)  
        file_temp=(forlder_to_work2+'results/second_clean_%d.wav'%k)
        write(file_temp, fs,speech)
        
        speech,t1=istft(z_k_noise.T[:,:,0].T, win, win, hop, nfft, fs)  
        file_temp=(forlder_to_work2+'results/only_noise_%d.wav'%k)
        write(file_temp, fs,speech)  
        

        
############# overlap area only #####################################        

        file_temp=(forlder_to_work2+'results_overlap/separating_speaker_number_0_{}.wav'.format(k))
        write(file_temp, fs,s_hat_first_one_speaker_time)
        file_temp=(forlder_to_work2+'results_overlap/separating_speaker_number_1_{}.wav'.format(k))
        write(file_temp, fs,s_hat_second_one_speaker_time)
        file_temp=(forlder_to_work2+'results_overlap/noisy_signal_%d.wav'%k)
        write(file_temp, fs,z_k_noise_time)
        file_temp=(forlder_to_work2+'results_overlap/first_clean_%d.wav'%k)
        write(file_temp, fs,z_k_first_one_speaker_time)
        file_temp=(forlder_to_work2+'results_overlap/second_clean_%d.wav'%k)
        write(file_temp, fs,z_k_second_one_speaker_time) 
############# plot z_k and VAD for first and second channel (projection is difficult to recover) ######################
epsilon = 1e-6
logging.shutdown()
temp=20*np.log10(abs(z_k[:,:,0].T)+epsilon);

temp_1=20*np.log10(abs(z_k_first[0,:,:])+epsilon);
temp_2=20*np.log10(abs(z_k_second[0,:,:])+epsilon);

temp1=20*np.log10(abs(s_hat_total[:,:,0].T)+epsilon);
temp2=20*np.log10(abs(s_hat_total[:,:,1].T)+epsilon);


x=nfft/4
#fig, axs = plt.subplots(2)
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#plt.rcParams.update({'font.size': 12})
#fig.suptitle('real scenario with 2 speakers. first speaker #1 is active then speaker #2 and then both') #BF output for extracting speaker #1 with the CSD
#im = axs[0].imshow(temp[::-1],vmin=-2,vmax=2,aspect='auto')
#
#cb_ax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
#cbar = fig.colorbar(im, cax=cb_ax)
#line_up =axs[1].scatter(np.arange(len(y)),342*y,2*np.ones(len(y)),c='b',alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
#line_down =axs[1].scatter(np.arange(len(y)),342*y_prob_dynamic,2*np.ones(len(y)),c='r',alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')
#plt.legend(handles=[line_up,line_down],loc=(-37,0.3))
#axs[1].set_xlabel('Time(sec)')
#axs[0].set_ylabel('Frequency(KHz)')
#axs[1].set_ylabel('CSD(l)')
#axs[1].set_yticks(np.arange(0,nfft/2,342))
#axs[1].set_yticklabels(['Noise','One speaker','Many speakers'])
#axs[0].set_yticks(np.arange(0,nfft/2,257))
#axs[0].set_yticklabels(['8','6','4','2'])
#axs[1].set_xticks(np.arange(0,len(y),31.25))
#axs[1].set_xticklabels(np.arange(0,int(len(y)/31.25),1))
#axs[1].margins(x=0) 
#plt.subplots_adjust(wspace=0, hspace=0)
#axs[0].set_xticks([])
#plt.show()
#plt.savefig(r'C:\Users\user\Desktop\csd_before.jpg', dpi=300)
#
y2_for_plot = y2.copy()
y2_for_plot[y2_for_plot==0]=0
y2_for_plot[y2_for_plot==19]=0

y2_prob_for_plot = y2_prob_stat.copy()
y2_prob_for_plot = y2_prob_for_plot.astype(np.float64)
y2_prob_for_plot[y2==0]=0
y2_prob_for_plot[y2==19]=0

y2_prob_for_plot = y2_prob_for_plot
y2_for_plot = y2_for_plot
#
#x=nfft/4
#fig, axs = plt.subplots(2)
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#fig.suptitle('BF output for extracting speaker #1 with the DOA')
#im = axs[0].imshow(temp2[::-1],vmin=-2,vmax=2,aspect='auto')
#cb_ax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
#cbar = fig.colorbar(im, cax=cb_ax)
#line_up =axs[1].scatter(np.arange(len(y)),57*(y2_for_plot-1),2*np.ones(len(y)),c='b',alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
#line_down =axs[1].scatter(np.arange(len(y)),57*(y2_prob_for_plot-1),2*np.ones(len(y)),c='r',alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')
#plt.legend(handles=[line_up,line_down],loc=(-37,0.4))
#axs[1].set_xlabel('Time(sec)')
#axs[0].set_ylabel('Frequency(KHz)')
#axs[1].set_ylabel('DOA(l)')
#axs[1].set_yticks(np.arange(0,nfft/2,54))
#axs[1].set_yticklabels(['Noise/Many speakers','0-10','11-20','21-30',
#                               '31-40','41-50','51-60',
#                               '61-70','71-80','81-90',
#                          '91-100','101-110','111-120',
#                         '121-130','131-140','141-150',
#                         '151-160','161-170','171-180'])
#axs[0].set_yticks(np.arange(0,nfft/2,257))
#axs[0].set_yticklabels(['8','6','4','2'])
#axs[1].set_xticks(np.arange(0,len(y),31.25))
#axs[1].set_xticklabels(np.arange(0,int(len(y)/31.25),1))
#axs[1].margins(x=0) 
#plt.subplots_adjust(wspace=0, hspace=0)
#axs[0].set_xticks([])
#plt.show()
#plt.savefig(r'C:\Users\user\Desktop\csd.jpg', dpi=300)

fig, axs = plt.subplots(7, 1, 
                       gridspec_kw={
                           'width_ratios': [1],
                           'height_ratios': [1,1,1,1,1,2,6]})
#fig, axs = plt.subplots(7,1,gridspec_kw={'width_ratios': [1,1,1,1,1,1,1]})
fig.set_figheight(6)
fig.set_figwidth(3)
fig.tight_layout(pad=0.25)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
im = axs[0].imshow(temp_1[::-1],vmin=-2,vmax=2,aspect='auto')
im = axs[1].imshow(temp_2[::-1],vmin=-2,vmax=2,aspect='auto')
im = axs[2].imshow(temp[::-1],vmin=-2,vmax=2,aspect='auto')
im = axs[3].imshow(temp1[::-1],vmin=-2,vmax=2,aspect='auto')
im = axs[4].imshow(temp2[::-1],vmin=-2,vmax=2,aspect='auto')
cb_ax = fig.add_axes([0.75, 0.335, 0.02, 0.55])
cbar = fig.colorbar(im, cax=cb_ax)

line_up =axs[5].scatter(np.arange(len(y)),342*y,2*np.ones(len(y)),c='b',alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
line_down =axs[5].scatter(np.arange(len(y)),342*y_prob_dynamic,2*np.ones(len(y)),c='r',alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')
plt.legend(handles=[line_up,line_down],loc=(-37,0.4))

colors1 = []
colors2 = []
for i,v in enumerate(y2_for_plot):
    if v==0.0:
       colors1.append('white')
       colors2.append('white')
    else:
       colors1.append('g')
       colors2.append('r')

line_up2 =axs[6].scatter(np.arange(len(y)),57*(y2_for_plot-1),2*np.ones(len(y)),c=colors1,alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
line_down2 =axs[6].scatter(np.arange(len(y)),57*(y2_prob_for_plot-1),2*np.ones(len(y)),c=colors2,alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')

axs[0].title.set_text('(a) First oracle speaker')
axs[1].title.set_text('(b) Second oracle speaker')
axs[2].title.set_text('(c) Input signal')
axs[3].title.set_text('(d) First output channel')
axs[4].title.set_text('(e) Second output channel')
axs[5].title.set_text('(f) CSD(l)')
axs[6].title.set_text('(g) DOA(l)')


axs[6].set_xlabel('Time(sec)')
axs[2].set_ylabel('Frequency(KHz)',fontsize=20)


axs[5].set_ylabel('Label')
axs[5].set_yticks(np.arange(0,nfft/2,342))
axs[5].set_yticklabels(['Noise','One speaker','Many speakers'])
    
axs[6].set_ylabel('Center label')
axs[6].set_yticks(np.arange(0,nfft/2,57))
axs[6].set_yticklabels(['5','15','25',
                               '35','45','55',
                               '65','75','85',
                          '95','105','115',
                         '125','135','145',
                         '155','165','175'])

axs[0].set_yticks(np.arange(0,nfft/2,257))
axs[0].set_yticklabels(['8','6','4','2'])

axs[1].set_yticks(np.arange(0,nfft/2,257))
axs[1].set_yticklabels(['8','6','4','2'])

axs[2].set_yticks(np.arange(0,nfft/2,257))
axs[2].set_yticklabels(['8','6','4','2'])

axs[3].set_yticks(np.arange(0,nfft/2,257))
axs[3].set_yticklabels(['8','6','4','2'])

axs[4].set_yticks(np.arange(0,nfft/2,257))
axs[4].set_yticklabels(['8','6','4','2'],fontsize=20)



axs[6].set_xticks(np.arange(0,len(y),31.25))
axs[6].set_xticklabels(np.arange(0,int((len(y)+30)/31.25),1))
axs[6].margins(x=0) 
axs[5].margins(x=0)
#plt.subplots_adjust(wspace=0, hspace=0)
axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xticks([])
axs[3].set_xticks([])
axs[4].set_xticks([])
axs[5].set_xticks([])
#axs[6].set_xticks([])

plt.show()

################### plot separation channel first ##############
#
#together_time,t1=istft(z_k_first[0,:,:], win, win, hop, nfft, fs)
#for k in range(0,num_speech):
#    speech,t1=istft(s_hat_first[:,:,k].T, win, win, hop, nfft, fs)
#    fig, ax1 = plt.subplots()
#    line_up, =ax1.plot(together_time,label='Before LCMV first')
#    line_down, =ax1.plot(speech,label='After LCMV first')
#    plt.legend(handles=[line_up,line_down])
#    ax1.set_xlabel('Time(sec)')
#    ax1.set_ylabel('Amplitude')
#    plt.xticks(np.arange(0,len(speech),fs))
#    plt.margins(x=0)
#    ticks_x = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z/fs))
#    ax1.xaxis.set_major_formatter(ticks_x)
#    axes = plt.gca()
#    axes.set_ylim([-1.5,1.5])
#    plt.show()
#
################### plot separation channel second ##############
#
#together_time,t1=istft(z_k_second[0,:,:], win, win, hop, nfft, fs)
#for k in range(0,num_speech):
#    speech,t1=istft(s_hat_second[:,:,k].T, win, win, hop, nfft, fs)
#    fig, ax1 = plt.subplots()
#    line_up, =ax1.plot(together_time,label='Before LCMV second')
#    line_down, =ax1.plot(speech,label='After LCMV second')
#    plt.legend(handles=[line_up,line_down])
#    ax1.set_xlabel('Time(sec)')
#    ax1.set_ylabel('Amplitude')
#    plt.xticks(np.arange(0,len(speech),fs))
#    plt.margins(x=0)
#    ticks_x = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z/fs))
#    ax1.xaxis.set_major_formatter(ticks_x)
#    axes = plt.gca()
#    axes.set_ylim([-1.5,1.5])
#    plt.show()
#
################ plot 2 speakers ####################################
#
#
#fig, ax1 = plt.subplots()
#line_up, =ax1.plot(receiver_first[:,0],label='first speaker')
#line_down, =ax1.plot(receiver_second[:,0],label='second speaker')
#plt.legend(handles=[line_up,line_down])
#ax1.set_xlabel('Time(sec)')
#ax1.set_ylabel('Amplitude')
#plt.xticks(np.arange(0,len(speech),fs))
#plt.margins(x=0)
#ticks_x = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z/fs))
#ax1.xaxis.set_major_formatter(ticks_x)
#axes = plt.gca()
#axes.set_ylim([-1.5,1.5])
#plt.show()
#
# 
##### save wav ##################################################
#
#together_time,t1=istft(z_k[:,:,0].T, win, win, hop, nfft, fs)
#scaled = together_time/np.max(together_time)
#file_temp=(forlder_to_work+'results/noisy_signal.wav')
#write(file_temp, fs,scaled)
#

############################ confusion matrix static ########################################### 


y2_prob_total_stat=np.delete(y2_prob_total_stat, np.where(y2_total==0)[0])
y2_prob_total_dync=np.delete(y2_prob_total_dync, np.where(y2_total==0)[0])
y2_total=np.delete(y2_total, np.where(y2_total==0))
y2_prob_total_stat=np.delete(y2_prob_total_stat, np.where(y2_total==19))
y2_prob_total_dync=np.delete(y2_prob_total_dync, np.where(y2_total==19))
y2_total=np.delete(y2_total, np.where(y2_total==19))


num_classes=model3.layers[-1].output_shape[1]     
cm_plot_labels = ['Noise','One speacker','2 speackers']
plot_confusion_matrix_from_data(y_total, y_prob_total_stat,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


num_classes=model18.layers[-1].output_shape[1]-2    
cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
                  ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
plot_confusion_matrix_from_data(y2_total-1,y2_prob_total_stat-1,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


# mir_eval_plot  ##############################################


if type_r == 'T60':
    range_x = np.arange(300,551,100)
if type_r == 'overlap':
    range_x = np.arange(0,76,25)
if type_r == 'SIR':
    range_x = np.arange(0,16,5)
if type_r == 'SNR':
    range_x = np.arange(10,21,2.5)

plt.figure()
line_1, = plt.plot(stoi_alg.mean(axis=1), label='Proposed')
line_2, = plt.plot(stoi_noisy.mean(axis=1), label='Noisy')
plt.legend(handles=[line_1,line_2])
plt.title('STOI test')
plt.xlabel(type_r)
plt.ylabel('Percent')
xi = list(range(len(stoi_alg)))
plt.xticks(xi,range_x)
plt.margins(x=0)
plt.show()    


plt.figure()
line_1, = plt.plot(pesq_alg.mean(axis=1), label='Proposed')
line_2, = plt.plot(pesq_noisy.mean(axis=1), label='Noisy')
plt.legend(handles=[line_1,line_2])
plt.xlabel(type_r)
plt.title('PESQ test')
plt.ylabel('Score')
xi = list(range(len(stoi_alg)))
plt.xticks(xi,range_x)
plt.margins(x=0)
plt.show()    

plt.figure()
line_1, = plt.plot(sir_alg.mean(axis=1), label='Proposed')
plt.legend(handles=[line_1])
plt.title('Speach Enhancement')
plt.xlabel(type_r)
plt.ylabel('dB')
xi = list(range(len(stoi_alg)))
plt.xticks(xi,range_x)
plt.margins(x=0)
plt.show()    
