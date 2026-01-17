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
from joblib import delayed
import multiprocessing
from numpy import linalg as LA
from scipy.io.wavfile import write
import scipy.io as sio
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
import logging
from pystoi import stoi
from pesq import pesq
import mir_eval

plt.close("all")
forlder_to_work1 = 'C:/project/'
RTF_mode = 'GEVD'
new=0
nom_data_sets=11           ################ amount of signals to test ###################
scaler = StandardScaler()
nfft=2048
wlen = 2048                                                                     
hop = wlen/4 
num_classes = 3
img_rows, img_cols = 1025,7
NUP=1025                                          
win=np.hamming(wlen)
num_cores = multiprocessing.cpu_count()
pad=30
e=0.01
epsilon=0.0001
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

#def stft_z(get_receivers):
#    return stft(get_receivers, win, hop, nfft)

#def create_cholesky_Qvv(z_k_start):
#    temp=z_k_start@(z_k_start.conj().T)/pad
#    return LA.cholesky(temp)
#
#def create_X(l,j):
#    chol_j=LA.inv(cholesky_Qvv[j,:,:])
#    Zvv = 0
#    sum_win_vad = 0
#    for p in range(frame_before+frame_after+1):
#        temp_zvv = chol_j@(win_vad[10-frame_before+p]*z_k[:,j,l-frame_before+p].reshape(M,1))
#        Zvv = Zvv+temp_zvv@temp_zvv.conj().T
#        sum_win_vad = sum_win_vad+win_vad[10-frame_before+p]
#    Zvv=Zvv/sum_win_vad
#    w,v = LA.eig(Zvv)
#    fi=v[:,w.argmax()].reshape(M,1)
#    denominator=cholesky_Qvv[j,0,:].reshape(1,M)@fi
#    G_cw=np.squeeze(cholesky_Qvv[j,:,:])@fi/denominator
#    return np.squeeze(G_cw)
#
#def create_X_total(l):
#    X_temp=[]
#    X_temp = Parallel(n_jobs=num_cores, verbose=0)(delayed(
#        create_X)(l,j)for j in range(NUP))            
#    return np.asarray(X_temp)[:,1:M] 

plt.close("all")
model3=load_model(forlder_to_work1+'models/model_GEVD_18_separate_17_02.h5',compile=False)
model18=load_model(forlder_to_work1+'models/model2_GEVD_18_separate_17_02.h5',compile=False)
accurercy_list=np.zeros(nom_data_sets-1)

y_total=[]
y2_total=[]

y_prob_total_stat=[]
y2_prob_total_stat=[]

y_prob_total_dync=[]
y2_prob_total_dync=[]

type_r = 'SNR'
folder_names = ['10','12.5','15','17.5','20'] #folder_names = ['300','350','400','450','500','550'] #['10','12.5','15','17.5','20']

folder_to_save = 'C:/project/static_signals/two_speakers/SNR/'

if new:
    stoi_alg = np.zeros((len(folder_names),nom_data_sets-1))
    stoi_noisy = np.zeros((len(folder_names),nom_data_sets-1))
    stoi_ilerma = np.zeros((len(folder_names),nom_data_sets-1))
    
    pesq_alg = np.zeros((len(folder_names),nom_data_sets-1))
    pesq_noisy = np.zeros((len(folder_names),nom_data_sets-1))
    pesq_ilerma = np.zeros((len(folder_names),nom_data_sets-1))
    
    snr_noisy = np.zeros((len(folder_names),nom_data_sets-1))
    snr_alg = np.zeros((len(folder_names),nom_data_sets-1))
    snr_ilerma = np.zeros((len(folder_names),nom_data_sets-1))
    
    sdr_noisy = np.zeros((len(folder_names),nom_data_sets-1))
    sdr_alg = np.zeros((len(folder_names),nom_data_sets-1))
    sdr_ilerma = np.zeros((len(folder_names),nom_data_sets-1))

else:
    stoi_alg=np.load(folder_to_save+'stoi_alg.npy')
    stoi_noisy=np.load(folder_to_save+'stoi_noisy.npy')
    stoi_ilerma=np.load(folder_to_save+'stoi_ilerma.npy')
    pesq_alg=np.load(folder_to_save+'pesq_alg.npy')
    pesq_noisy=np.load(folder_to_save+'pesq_noisy.npy')
    pesq_ilerma=np.load(folder_to_save+'pesq_ilerma.npy')
    snr_alg=np.load(folder_to_save+'sir_alg.npy')
    snr_noisy=np.load(folder_to_save+'sir_noisy.npy')
    snr_ilerma=np.load(folder_to_save+'sir_ilerma.npy')
    sdr_alg=np.load(folder_to_save+'sdr_alg.npy')
    sdr_noisy=np.load(folder_to_save+'sdr_noisy.npy')
    sdr_ilerma=np.load(folder_to_save+'sdr_ilerma.npy')  


flag_start_lcmv = 1
for d in range(0,len(folder_names)):
    forlder_to_work2 = ('C:/project/static_signals/two_speakers/SNR/SNR_%s_T60_300_SIR_0/'%folder_names[d])
    for k in range(1,2):
        flag_start_lcmv = 1
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=forlder_to_work2+'/log_files/info_%d.log'%k,level=logging.DEBUG)    
        
        print(k)
        signal_file=(forlder_to_work2+'dynamic_signal_%d.wav'%k)
    
        signal_first_file=(forlder_to_work2+'dynamic_signal_first_%d.wav'%k)
        signal_second_file=(forlder_to_work2+'dynamic_signal_second_%d.wav'%k)
        
        angle_location_first=(forlder_to_work2+'angle_location_first_%d.mat'%k)
        angle_location_Second=(forlder_to_work2+'angle_location_second_%d.mat'%k)
        
        signal_noise_file=(forlder_to_work2+'dynamic_signal_noise_%d.wav'%k)
    
        ilrma_first_file=(forlder_to_work2+'estimated_first_channel_%d.wav'%k)
        ilrma_second_file=(forlder_to_work2+'estimated_second_channel_%d.wav'%k)
          
    ############################### create input to the model ##############################################
    
        mat_locations=sio.loadmat(angle_location_first)
        location_first=np.transpose(mat_locations['label_first'])
        location_first = location_first[0][0]
        mat_locations=sio.loadmat(angle_location_Second)
        location_second=np.transpose(mat_locations['label_second'])
        location_second = location_second[0][0]
    
    
    
        fs,receiver_first = wavfile.read(signal_first_file)
        receiver_first = receiver_first[:,indices]
        fs,receiver_second = wavfile.read(signal_second_file)
        receiver_second = receiver_second[:,indices]        
        fs,receivers= wavfile.read(signal_file)
        receivers = receivers[:,indices]
    
        fs,ilrma_first= wavfile.read(ilrma_first_file)
        fs,ilrma_second= wavfile.read(ilrma_second_file)
        ilrma_first=ilrma_first/(abs(ilrma_first).max())
        ilrma_second=ilrma_second/(abs(ilrma_second).max())
        
        receiver_first=receiver_first/(abs(receiver_first).max())
        receiver_second=receiver_second/(abs(receiver_second).max())
        receivers=receivers/(abs(receivers).max())
    
        fs,noise = wavfile.read(signal_noise_file)
        noise = noise[:,indices]
        noise = noise/(abs(noise).max())
    
    
        M=len(receiver_first[0,:])
        index=int(1+np.fix((len(receiver_first[:,1])-wlen)/hop))
    
    ############################## create y2 from location file #############################
    
        y2_first = np.ones((index,1))*location_first
        y2_second = np.ones((index,1))*location_second
                
        z_k_ilrma_first = stft(ilrma_first, win, hop, nfft).T
        z_k_ilrma_second = stft(ilrma_second, win, hop, nfft).T
        
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
        vad1_temp=vad1_temp.astype(np.int)
        vad1_temp_sum=vad1_temp.sum(axis=0)
        vad1_temp_sum=vad1_temp_sum.astype(np.int)
        vad1_temp_sum1 = vad1_temp_sum > threshold
        vad1=vad1_temp_sum1.astype(np.int)
    
       
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
    
        
        vad1_location_update = np.squeeze(y2_first.T*vad1.T)
        vad2_location_update = np.squeeze(y2_second.T*vad2.T)
        L=vad1+vad2
    
        y= L[frame_before:index-frame_after]
        y2 = np.squeeze(vad1_location_update+vad2_location_update)   
        y2= y2[frame_before:index-frame_after]
        y2=np.where(y!=2, y2,19)
        
    ############################### source separation #####################################
    
        z_k= np.transpose(z_k, (2,1,0))
        z_k= z_k[frame_before:index-frame_after,:,:]
        
        
        z_k_ilrma_first= z_k_ilrma_first[frame_before:index-frame_after]
        z_k_ilrma_second= z_k_ilrma_second[frame_before:index-frame_after]
        
        z_k_first= z_k_first[:,:,frame_before:index-frame_after]
        z_k_second= z_k_second[:,:,frame_before:index-frame_after]
        z_k_noise= z_k_noise[:,:,frame_before:index-frame_after]
           
        flag_first_noise = 0
        Frame_classification_system = np.zeros((2,3))
        G = np.ones((NUP,M,2),dtype=complex)
        G_stand = np.ones((NUP,M)) + np.ones((NUP,M)) * 1j
        
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
        
        Qyy_stand = np.zeros((NUP,M,M),complex)
        Qyy_first = np.zeros((NUP,M,M),complex)
        Qyy_second = np.zeros((NUP,M,M),complex)
            
        W=np.ones((len(x_test),NUP,M,2),complex)
    
        PSD_matrix_per_DOA = np.zeros((18,NUP,M,M),complex)
        total_frame_per_DOA = np.zeros((18))
        
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
                        alfa_Qvv = 1/sum_Qvv
                        Qvv = (1-alfa_Qvv)*Qvv+(alfa_Qvv)*Qvv_temp
    #            else:
    #                flag_last_y_prob=0
    #                
            if y_prob[0]==1:
    #            if flag_last_y_prob==1:
    #                
                    current_y_label = 1
                    y2_pred=model18.predict(x_test[l].reshape(1,NUP,img_cols))
                    y2_pred2=pd.DataFrame(y2_pred)
                    y2_prob=y2_pred2.idxmax(axis=1)
                    y2_prob=y2_prob.to_numpy()
                          
                    if y2_prob[0] == Frame_classification_system[0,0]:
                        Frame_classification_system[1,0]=Frame_classification_system[1,0]+1
                        Frame_classification_system[1,2]=0
                        Frame_classification_system[0,2]=0
                        time_first = 0
                        time_second +=1
                        
                        total_frame_per_DOA[y2_prob[0]-1] += 1
                        alfa_G = 1/(total_frame_per_DOA[y2_prob[0]-1])
    
                        for j in range(NUP):
                            if RTF_mode=='spectral subtraction':
                                Qyy_first[j,:,:] = alfa_G*Qyy_first[j,:,:]+(1-alfa_G)*(z_k[l,j,:].reshape((M,1))@z_k[l,j,:].reshape((M,1)).conj().T)
                                A= Qyy_first[j,:,:]-Qvv[j,:,:]
                                G[j,:,0] = (A)@A[0,:].conj().T/(A[0,:].T.conj()@A[0,:]+e*sum(A[0,:]).real)
                            if RTF_mode=='GEVD':
                                cholesky_Qvv=LA.cholesky(Qvv[j,:,:]+e*abs(np.matrix.trace(Qvv[j,:,:]))*np.eye(M))
                                chol_j=LA.inv(cholesky_Qvv)
                                a=chol_j@z_k[l,j,:].reshape(1,M).T
                                Zvv_temp = a@a.conj().T
                                PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:] = (1-alfa_G)*PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:]+(alfa_G)*Zvv_temp
                                
                                w,v = LA.eig(PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:])
                                phi=v[:,w.argmax()].reshape(M,1)
                                denominator=cholesky_Qvv[0,:].reshape(1,M)@phi
                                G[j,:,0]=np.squeeze(cholesky_Qvv@phi/denominator)        
                        current_y2_label = y2_prob[0]
        
                    elif y2_prob[0] == Frame_classification_system[0,1]:
                        Frame_classification_system[1,1]=Frame_classification_system[1,1]+1
                        Frame_classification_system[1,2]=0
                        Frame_classification_system[0,2]=0
                        time_second = 0
                        time_first +=1
                        
                        total_frame_per_DOA[y2_prob[0]-1] += 1
                        alfa_G = 1/(total_frame_per_DOA[y2_prob[0]-1])                    
        
                        for j in range(NUP):
                            if RTF_mode=='spectral subtraction':
                                Qyy_second[j,:,:] = alfa_G*Qyy_second[j,:,:]+(1-alfa_G)*(z_k[l,j,:].reshape((M,1))@z_k[l,j,:].reshape((M,1)).conj().T)
                                A= Qyy_second[j,:,:]-Qvv[j,:,:]
                                G[j,:,1] = (A)@A[0,:].conj().T/(A[0,:].T.conj()@A[0,:]+e*sum(A[0,:]).real)
                            if RTF_mode=='GEVD':
                                
                                cholesky_Qvv=LA.cholesky(Qvv[j,:,:]+e*abs(np.matrix.trace(Qvv[j,:,:]))*np.eye(M))
                                chol_j=LA.inv(cholesky_Qvv)
                                a=chol_j@z_k[l,j,:].reshape(1,M).T
                                Zvv_temp = a@a.conj().T
                                PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:] = (1-alfa_G)*PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:]+(alfa_G)*Zvv_temp
                                
                                w,v = LA.eig(PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:])
                                phi=v[:,w.argmax()].reshape(M,1)
                                denominator=cholesky_Qvv[0,:].reshape(1,M)@phi
                                G[j,:,1]=np.squeeze(cholesky_Qvv@phi/denominator)
                                
                        current_y2_label = y2_prob[0]
                        
                    elif y2_prob[0] == Frame_classification_system[0,2]:
                        stand_z = np.concatenate((stand_z,z_k[l,:,:].reshape(1,NUP,M)))
                        Frame_classification_system[1,2]=Frame_classification_system[1,2]+1 
                        time_second +=1
                        time_first +=1
                        
                        if Frame_classification_system[1,2]>(threshold_chage_location-1):
                            current_y2_label = y2_prob[0]
                            for j in range(NUP):
                                if RTF_mode=='GEVD':
                                    cholesky_Qvv=LA.cholesky(Qvv[j,:,:]+e*abs(np.matrix.trace(Qvv[j,:,:]))*np.eye(M))
                                    chol_j=LA.inv(cholesky_Qvv)
                                    a=chol_j@stand_z[:,j,:].T
                                    Zvv_temp = a@a.conj().T/threshold_chage_location
                                    temp_alfa = total_frame_per_DOA[y2_prob[0]-1]+threshold_chage_location
                                    PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:] = total_frame_per_DOA[y2_prob[0]-1]/temp_alfa*PSD_matrix_per_DOA[y2_prob[0]-1,j,:,:]+threshold_chage_location/temp_alfa*Zvv_temp
                                if RTF_mode=='spectral subtraction':
                                    Qyy_stand[j,:,:] = stand_z[:,j,:].T@stand_z[:,j,:].conj()/threshold_chage_location
                                    A= Qyy_stand[j,:,:]-Qvv[j,:,:]
                                    G_stand[j,:] = (A)@A[0,:].conj().T/(A[0,:].T.conj()@A[0,:]+e*sum(A[0,:]).real)                           
                            
                            if Frame_classification_system[1,0] == 0:
                                Frame_classification_system[0,0] = y2_prob[0]
                                Frame_classification_system[1,0] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
                                Qyy_first = Qyy_stand
                            elif (Frame_classification_system[1,1] == 0) & (abs(Frame_classification_system[0,0]-y2_prob[0])<3):
                                Frame_classification_system[0,0] = y2_prob[0]
                                Frame_classification_system[1,0] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
                                Qyy_first = Qyy_stand
                            elif Frame_classification_system[1,1] == 0: 
                                Frame_classification_system[0,1] = y2_prob[0]
                                Frame_classification_system[1,1] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
                                Qyy_second = Qyy_stand
                            else:     
                                to_change = np.argmin(np.abs(np.array((y2_prob[0],y2_prob[0])) - Frame_classification_system[0,0:2]))
                                min1,min2 = np.abs(np.array((y2_prob[0],y2_prob[0])) - Frame_classification_system[0,0:2])
    
                                if (min1>(threshold_chage_location-2)) & (min2>(threshold_chage_location-2)):
                                    if (time_first-min1*30)>(time_second-min2*30):
                                        to_change = 0
                                    else: 
                                        to_change = 1
                                    
                                        
                                Frame_classification_system[0,to_change] = y2_prob[0]
                                Frame_classification_system[1,to_change] = Frame_classification_system[1,2] 
                                Frame_classification_system[0,2] = 0
                                Frame_classification_system[1,2] = 0
    
                                if to_change==0:
                                    Qyy_first = Qyy_stand
                                if to_change==1:
                                    Qyy_second = Qyy_stand
                                    
                            total_frame_per_DOA[y2_prob[0]-1] += threshold_chage_location
                            
                    else:
                        stand_z = z_k[l,:,:].reshape(1,NUP,M)
                        Frame_classification_system[0,2] = y2_prob[0]
                        Frame_classification_system[1,2] = 1
                        time_second +=1
                        time_first +=1
    
                    logging.info('Frame DOAc association : %d'%y2_prob[0])
                    logging.info('Frame classification system:')
                    logging.info(Frame_classification_system[0,:])
                    logging.info(Frame_classification_system[1,:])
    #            else:
    #                flag_last_y_prob=1
            if y_prob[0]==2:
    #            if flag_last_y_prob==2:
                    current_y_label = 2
                    current_y2_label = 19
                    time_second +=1
                    time_first +=1
    #            else:
    #               flag_last_y_prob=2 
            
            
    ################### source separation ###################################################
            
           
            if (Frame_classification_system[0,0]==0) & (Frame_classification_system[0,1]==0):
                s_hat=z_k[l,:,0]
                s_hat = np.concatenate((s_hat.reshape(1,NUP),s_hat.reshape(1,NUP)))
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
                    inv_temp=LA.inv(g.conj().T@c+e*abs(np.matrix.trace(g.conj().T@c))*np.eye(num_speech))
                    W[l,j,:,:]=c@inv_temp
                    s_hat[:,j]=W[l,j,:,:].conj().T@z_k[l,j,:]
            
            if l==0:
                s_hat_total = s_hat.T.reshape(1,NUP,num_speech)
            else:
                s_hat_total = np.concatenate((s_hat_total,s_hat.T.reshape(1,NUP,num_speech)),axis=0)
            
            y_prob_dynamic[l] = current_y_label
            y2_prob_dynamic[l] = current_y2_label
    
    ################################### bss eval ################################
        s_hat_total_for_test = s_hat_total[np.where(y==2)]
        z_k_first_one_speaker = z_k_first.T[np.where(y==2)]
        z_k_second_one_speaker = z_k_second.T[np.where(y==2)]
        z_k_noisy = z_k[np.where(y==2)]    
        
        s_hat_total_for_test_ilrma = np.concatenate((z_k_ilrma_first.reshape(z_k_ilrma_first.shape[0],NUP,1),z_k_ilrma_second.reshape(z_k_ilrma_first.shape[0],NUP,1)),axis=2)
        s_hat_total_for_test_ilrma=s_hat_total_for_test_ilrma[np.where(y==2)]


        s_hat_first_one_speaker_ilerma_time,_=istft(s_hat_total_for_test_ilrma[:,:,0].T, win, win, hop, nfft, fs)
        s_hat_second_one_speaker_ilerma_time,_=istft(s_hat_total_for_test_ilrma[:,:,1].T, win, win, hop, nfft, fs)

        s_hat_first_one_speaker_time,_=istft(s_hat_total_for_test[:,:,0].T, win, win, hop, nfft, fs)
        s_hat_second_one_speaker_time,_=istft(s_hat_total_for_test[:,:,1].T, win, win, hop, nfft, fs)

        z_k_first_one_speaker_time,_=istft(z_k_first_one_speaker[:,:,0].T, win, win, hop, nfft, fs)
        z_k_second_one_speaker_time,_=istft(z_k_second_one_speaker[:,:,0].T, win, win, hop, nfft, fs)
        
        z_k_noise_time,_=istft(z_k_noisy[:,:,0].T, win, win, hop, nfft, fs)

        noisy_sources = np.concatenate((z_k_noise_time.reshape(z_k_noise_time.shape[0],1),z_k_noise_time.reshape(z_k_noise_time.shape[0],1)),axis=1)
        
        reference_sources = np.concatenate((z_k_first_one_speaker_time.reshape(z_k_first_one_speaker_time.shape[0],1),z_k_second_one_speaker_time.reshape(z_k_second_one_speaker_time.shape[0],1)),axis=1)
        estimated_sources = np.concatenate((s_hat_first_one_speaker_time.reshape(s_hat_first_one_speaker_time.shape[0],1),s_hat_second_one_speaker_time.reshape(s_hat_second_one_speaker_time.shape[0],1)),axis=1)
        estimated_sources_ilerma = np.concatenate((s_hat_first_one_speaker_ilerma_time.reshape(s_hat_first_one_speaker_ilerma_time.shape[0],1),s_hat_second_one_speaker_ilerma_time.reshape(s_hat_second_one_speaker_ilerma_time.shape[0],1)),axis=1)
        
        (sdr_noise,sir_noise,sar_noise,perm)  = mir_eval.separation.bss_eval_sources(reference_sources.T+10**(-9), noisy_sources.T, compute_permutation=True)
        
        snr_noisy[d,k-1] = sir_noise.mean()
        sdr_noisy[d,k-1] = sdr_noise.mean()
        
    ###################################### alg results ##################################################3    
#        first_separation = max(z_k_first_noisy_time.std()/s_hat_first_one_speaker_time0.std(),z_k_first_noisy_time.std()/s_hat_first_one_speaker_time1.std())
#        first_separation = 10*np.log10((first_separation)**2)
#        
#        second_separation = max(z_k_second_noisy_time.std()/s_hat_second_one_speaker_time0.std(),z_k_second_noisy_time.std()/s_hat_second_one_speaker_time1.std())
#        second_separation = 10*np.log10((second_separation)**2)
        
        (sdr,sir,sar,perm)  = mir_eval.separation.bss_eval_sources(reference_sources.T+10**(-9), estimated_sources.T, compute_permutation=True)

        snr_alg[d,k-1] = sir.mean()
        sdr_alg[d,k-1] = sdr.mean()
        
        first_stoi = max(stoi(z_k_first_one_speaker_time,s_hat_first_one_speaker_time,fs),stoi(z_k_first_one_speaker_time,s_hat_second_one_speaker_time,fs))
        second_stoi = max(stoi(z_k_second_one_speaker_time,s_hat_first_one_speaker_time,fs),stoi(z_k_second_one_speaker_time,s_hat_second_one_speaker_time,fs))
    
        first_stoi_noisy = stoi(z_k_first_one_speaker_time,z_k_noise_time,fs)
        second_stoi_noisy = stoi(z_k_second_one_speaker_time,z_k_noise_time,fs)
    
    
        first_pesq = max(pesq(fs,z_k_first_one_speaker_time,s_hat_first_one_speaker_time,'wb'),pesq(fs,z_k_first_one_speaker_time,s_hat_second_one_speaker_time,'wb'))
        second_pesq = max(pesq(fs,z_k_second_one_speaker_time,s_hat_first_one_speaker_time,'wb'),pesq(fs,z_k_second_one_speaker_time,s_hat_second_one_speaker_time,'wb'))
    
        first_pesq_noisy = pesq(fs,z_k_first_one_speaker_time,z_k_noise_time,'wb')
        second_pesq_noisy = pesq(fs,z_k_second_one_speaker_time,z_k_noise_time,'wb') 
        
#        separation = (first_separation+second_separation)/2
#        separation_total = np.append(separation_total,separation)
#    
        stoi_alg[d,k-1] = (first_stoi+second_stoi)/2
        pesq_alg[d,k-1] = (first_pesq+second_pesq)/2   
        stoi_noisy[d,k-1] = (first_stoi_noisy+second_stoi_noisy)/2
        pesq_noisy[d,k-1]  = (first_pesq_noisy+second_pesq_noisy)/2

        
        
    ###################################### ilrma results ##################################################3    
#        first_separation_ilerma = max(z_k_first_noisy_time.std()/s_hat_first_one_speaker_ilerma_time0.std(),z_k_first_noisy_time.std()/s_hat_first_one_speaker_ilerma_time1.std())
#        first_separation_ilerma = 10*np.log10((first_separation_ilerma)**2)
#        
#        second_separation_ilerma = max(z_k_second_noisy_time.std()/s_hat_second_one_speaker_ilerma_time0.std(),z_k_second_noisy_time.std()/s_hat_second_one_speaker_ilerma_time1.std())
#        second_separation_ilerma = 10*np.log10((second_separation_ilerma)**2)
        (sdr_ilrma,sir_ilrma,sar_ilrma,perm_ilrma)  = mir_eval.separation.bss_eval_sources(reference_sources.T+10**(-9), estimated_sources_ilerma.T, compute_permutation=True)

        snr_ilerma[d,k-1] = sir_ilrma.mean()
        sdr_ilerma[d,k-1] = sdr_ilrma.mean()
            
        first_stoi_ilerma = max(stoi(z_k_first_one_speaker_time,s_hat_first_one_speaker_ilerma_time,fs),stoi(z_k_first_one_speaker_time,s_hat_second_one_speaker_ilerma_time,fs))
        second_stoi_ilerma = max(stoi(z_k_second_one_speaker_time,s_hat_first_one_speaker_ilerma_time,fs),stoi(z_k_second_one_speaker_time,s_hat_second_one_speaker_ilerma_time,fs))
    
        first_pesq_ilerma = max(pesq(fs,z_k_first_one_speaker_time,s_hat_first_one_speaker_ilerma_time,'wb'),pesq(fs,z_k_first_one_speaker_time,s_hat_second_one_speaker_ilerma_time,'wb'))
        second_pesq_ilerma = max(pesq(fs,z_k_second_one_speaker_time,s_hat_first_one_speaker_ilerma_time,'wb'),pesq(fs,z_k_second_one_speaker_time,s_hat_second_one_speaker_ilerma_time,'wb'))
    
#        separation_ilerma = (first_separation_ilerma+second_separation_ilerma)/2
#        separation_ilerma_total = np.append(separation_ilerma_total,separation_ilerma)
    
        stoi_ilerma[d,k-1] = (first_stoi_ilerma+second_stoi_ilerma)/2
        pesq_ilerma[d,k-1] = (first_pesq_ilerma+second_pesq_ilerma)/2
 
        
        
    ################################### save prediction to confusion matrix ######################
    
        y_total=np.append(y_total,y)
        y2_total=np.append(y2_total,y2)
        
        y_prob_total_stat=np.append(y_prob_total_stat,y_prob_stat)
        y2_prob_total_stat=np.append(y2_prob_total_stat,y2_prob_stat)
        
        y_prob_total_dync=np.append(y_prob_total_dync,y_prob_dynamic)
        y2_prob_total_dync=np.append(y2_prob_total_dync,y2_prob_dynamic)
        
#        np.save(folder_to_save+'stoi_alg.npy', stoi_alg)
#        np.save(folder_to_save+'stoi_noisy.npy', stoi_noisy)
#        np.save(folder_to_save+'stoi_ilerma.npy', stoi_ilerma)
#        np.save(folder_to_save+'pesq_alg.npy', pesq_alg)
#        np.save(folder_to_save+'pesq_noisy.npy', pesq_noisy)
#        np.save(folder_to_save+'pesq_ilerma.npy', pesq_ilerma)
#        np.save(folder_to_save+'snr_alg.npy', snr_alg)
#        np.save(folder_to_save+'snr_noisy.npy', snr_noisy)
#        np.save(folder_to_save+'snr_ilerma.npy', snr_ilerma)
#        np.save(folder_to_save+'sdr_alg.npy', sdr_alg)
#        np.save(folder_to_save+'sdr_noisy.npy', sdr_noisy)
#        np.save(folder_to_save+'sdr_ilerma.npy', sdr_ilerma)       
############# plot z_k and VAD for first and second channel (projection is difficult to recover) ######################

logging.shutdown()
temp=np.log(abs(z_k[:,:,0].T));

temp2=np.log(abs(s_hat_total[:,:,0].T));

x=nfft/4
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.imshow(temp[::-1],aspect='auto')
line_up, =ax2.plot(y, 'b-',label='Actual label - Speakers Detector')
line_down, =ax2.plot(y_prob_dynamic, 'r-',label='Predicted label - Speakers Detector')
plt.legend(handles=[line_up,line_down])
ax1.set_xlabel('Time(sec)')
ax1.set_ylabel('Frequency(Hz)')
ax2.set_yticklabels(['','Noise','','','','One speaker','','','','Many speakers'])
plt.xticks(np.arange(0,len(y),31.25))
plt.margins(x=0)
ticks_x = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z*x/fs))
ax2.xaxis.set_major_formatter(ticks_x)
plt.show()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.imshow(temp[::-1],aspect='auto')
line_up, =ax2.plot(y2, 'b-',label='Actual label - Speakers Detector and Localization')
line_down, =ax2.plot(y2_prob_dynamic, 'r-',label='Predicted label - Speakers Detector and Localization')
plt.legend(handles=[line_up,line_down])
ax1.set_xlabel('Time(sec)')
ax1.set_ylabel('Frequency(Hz)')
ax2.set_yticks(np.arange(0,20,1))
ax2.set_yticklabels(['Noise','0-10 degrees','11-20 degrees','21-30 degrees',
                               '31-40 degrees','41-50 degrees','51-60 degrees',
                               '61-70 degrees','71-80 degrees','81-90 degrees',
                          '91-100 degrees','101-110 degrees','111-120 degrees',
                         '121-130 degrees','131-140 degrees','141-150 degrees',
                         '151-160 degrees','161-170 degrees','171-180 degrees'
                         ,'Many speakers'])
plt.xticks(np.arange(0,len(y),31.25))
plt.margins(x=0)
ticks_x = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z*x/fs))
ax2.xaxis.set_major_formatter(ticks_x)
plt.show()


################### plot separation channel ##############
#
together_time,t1=istft(z_k[start_lcmv:,:,0].T, win, win, hop, nfft, fs)
speech1,t1=istft(s_hat_total[start_lcmv:,:,0].T, win, win, hop, nfft, fs)
speech2,t1=istft(s_hat_total[start_lcmv:,:,1].T, win, win, hop, nfft, fs)

fig, ax1 = plt.subplots()
line_1, =ax1.plot(together_time,alpha=1,label='Before LCMV',color='r')
line_2, =ax1.plot(speech1,alpha=0.5,label='After LCMV first channel',color='g')
plt.legend(handles=[line_1,line_2])
ax1.set_xlabel('Time(sec)')
ax1.set_ylabel('Amplitude')
plt.xticks(np.arange(0,len(speech1),fs))
plt.margins(x=0)
ticks_x = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z/fs))
ax1.xaxis.set_major_formatter(ticks_x)
axes = plt.gca()
axes.set_ylim([-1.5,1.5])
plt.show()


fig, ax1 = plt.subplots()
line_1, =ax1.plot(together_time,alpha=1,label='Before LCMV',color='r')
line_2, =ax1.plot(speech2,alpha=0.5,label='After LCMV second channel',color='g')
plt.legend(handles=[line_1,line_2])
ax1.set_xlabel('Time(sec)')
ax1.set_ylabel('Amplitude')
plt.xticks(np.arange(0,len(speech1),fs))
plt.margins(x=0)
ticks_x = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z/fs))
ax1.xaxis.set_major_formatter(ticks_x)
axes = plt.gca()
axes.set_ylim([-1.5,1.5])
plt.show()



############### plot 2 speakers ####################################


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

 
#### save wav ##################################################

#together_time,t1=istft(z_k[:,:,0].T, win, win, hop, nfft, fs)
#scaled = together_time/np.max(together_time)
#file_temp=(forlder_to_work2+'results/noisy_signal.wav')
#write(file_temp, fs,scaled)
#
################### save separation ##############
#
for k in range(0,num_speech):
    speech,t1=istft(s_hat_total[:,:,k].T, win, win, hop, nfft, fs)  
    scaled = speech/np.max(speech)
    file_temp=(forlder_to_work2+'results/separating_speaker_number_%d.wav'%k)
    write(file_temp, fs,scaled)

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
    range_x = np.arange(300,551,50)
if type_r == 'overlap':
    range_x = np.arange(0,76,25)
if type_r == 'SIR':
    range_x = np.arange(0,16,5)
if type_r == 'SNR':
    range_x = np.arange(10,21,2.5)
    
plt.figure()
line_1, = plt.plot(stoi_alg.mean(axis=1), label='stoi alg')
line_2, = plt.plot(stoi_noisy.mean(axis=1), label='stoi noisy')
line_3, = plt.plot(stoi_ilerma.mean(axis=1), label='stoi_ilerma')

plt.legend(handles=[line_1,line_2,line_3])
plt.title('stoi test')
plt.xlabel('SNR')
plt.ylabel('Percent')
xi = list(range(len(stoi_alg)))
plt.xticks(xi,range_x)
plt.margins(x=0)
plt.show()    


plt.figure()
line_1, = plt.plot(pesq_alg.mean(axis=1), label='pesq alg')
line_2, = plt.plot(pesq_noisy.mean(axis=1), label='pesq noisy')
line_3, = plt.plot(pesq_ilerma.mean(axis=1), label='pesq ilerma')

plt.legend(handles=[line_1,line_2,line_3])
plt.title('pesq test')
plt.xlabel('SNR')
plt.ylabel('Percent')
xi = list(range(len(stoi_alg)))
plt.xticks(xi,range_x)
plt.margins(x=0)
plt.show()    

plt.figure()
line_1, = plt.plot(snr_alg.mean(axis=1), label='SNR alg')
line_3, = plt.plot(snr_ilerma.mean(axis=1), label='SNR ilerma')

plt.legend(handles=[line_1,line_3])
plt.title('SNR')
plt.xlabel('SNR')
plt.ylabel('dB')
xi = list(range(len(stoi_alg)))
plt.xticks(xi,range_x)
plt.margins(x=0)
plt.show()    




