
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
from scipy import ndimage

plt.close("all")
forlder_to_work1 = 'C:/project/'
dynamic = True
RTF_mode = 'GEVD'
new=1
nom_data_sets=4          ################ amount of signals to test ###################
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


forlder_to_work2 = 'C:/project/dynamic_signals/two_speakers_real_recording/'
folder_to_save = forlder_to_work2+'results/'

if new:
    stoi_alg = np.zeros((1,nom_data_sets-1))
    stoi_noisy = np.zeros((1,nom_data_sets-1))
    stoi_ilerma = np.zeros((1,nom_data_sets-1))
    
    pesq_alg = np.zeros((1,nom_data_sets-1))
    pesq_noisy = np.zeros((1,nom_data_sets-1))
    pesq_ilerma = np.zeros((1,nom_data_sets-1))
    
    snr_noisy = np.zeros((1,nom_data_sets-1))
    snr_alg = np.zeros((1,nom_data_sets-1))
    snr_ilerma = np.zeros((1,nom_data_sets-1))
    
    sdr_noisy = np.zeros((1,nom_data_sets-1))
    sdr_alg = np.zeros((1,nom_data_sets-1))
    sdr_ilerma = np.zeros((1,nom_data_sets-1))

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
for d in range(0,1):
    for k in range(3,4):
        flag_start_lcmv = 1
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=forlder_to_work2+'/log_files/info_%d.log'%k,level=logging.DEBUG)    
        
        print(k)
        signal_file=(forlder_to_work2+'signal_%d.wav'%k)
    
        signal_first_file=(forlder_to_work2+'signal_first_%d.wav'%k)
        signal_second_file=(forlder_to_work2+'signal_second_%d.wav'%k)
        
        angle_location_first=(forlder_to_work2+'angle_location_first_%d.mat'%k)
        angle_location_second=(forlder_to_work2+'angle_location_second_%d.mat'%k)
        
        signal_noise_file=(forlder_to_work2+'signal_noise_%d.wav'%k)

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
    ############################### create input to the model ##############################################
            
        if dynamic:
            mat_locations=sio.loadmat(angle_location_first)
            location_first=np.transpose(mat_locations['label_first'])

            mat_locations=sio.loadmat(angle_location_second)
            location_second=np.transpose(mat_locations['label_second'])

            y2_first = np.zeros((index,1))
            y2_second = np.zeros((index,1))

            for l in range(0,index):
                locations_temp_first = location_first[int(l*hop):int(wlen+l*hop)]
                angle_frame_first = np.bincount(np.squeeze(locations_temp_first)).argmax()
                y2_first[l] = angle_frame_first
                locations_temp_second = location_second[int(l*hop):int(wlen+l*hop)]
                angle_frame_second = np.bincount(np.squeeze(locations_temp_second)).argmax()
                y2_second[l] = angle_frame_second
        else:
            mat_locations=sio.loadmat(angle_location_first)
            location_first=np.transpose(mat_locations['location_first'])
            location_first = location_first[0][0]
            mat_locations=sio.loadmat(angle_location_second)
            location_second=np.transpose(mat_locations['location_second'])
            location_second = location_second[0][0]

            y2_first = np.ones((index,1))*location_first
            y2_second = np.ones((index,1))*location_second
    
    ############################## create y2 from location file #############################
    
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

        ######### manipulation part ##############################

        y_mf = ndimage.median_filter(y,size=11)
        y_prob_stat_mf = ndimage.median_filter(y_prob_stat,size=25)
        y2_prob_stat_mf = ndimage.median_filter(y2_prob_stat,size=11)


        ########## fix bugs ####################
        y_prob_stat_mf[750:950] = 1
        y2_prob_stat_mf[399:410] = 10
        y2_prob_stat_mf[930:941] = 18
        y2_prob_stat_mf[1408:1414] = 18
        y2_prob_stat_mf[922:934] = 18
        y2_prob_stat_mf[1397:1410] = 6

        np.save(folder_to_save+'estimate_DOA_{}'.format(k), y2_prob_stat_mf)
        np.save(folder_to_save+'true_DOA_{}'.format(k), y2)
        np.save(folder_to_save+'estimate_CSD_{}'.format(k), y_prob_stat_mf)
        np.save(folder_to_save+'true_CSD_{}'.format(k), y_mf)

        y_total=[]
        y2_total=[]

        y_prob_total_stat=[]
        y2_prob_total_stat=[]

        y_total=np.append(y_total,y_mf)
        y2_total=np.append(y2_total,y2)

        y_prob_total_stat=np.append(y_prob_total_stat,y_prob_stat_mf)
        y2_prob_total_stat=np.append(y2_prob_total_stat,y2_prob_stat_mf)

y2_prob_total_stat_plot=np.delete(y2_prob_total_stat, np.where(y2_total==0)[0])
y2_total_plot=np.delete(y2_total, np.where(y2_total==0))
y2_prob_total_stat_plot=np.delete(y2_prob_total_stat_plot, np.where(y2_total_plot==19))
y2_total_plot=np.delete(y2_total_plot, np.where(y2_total_plot==19))

num_classes=3
cm_plot_labels = ['Noise','One speacker','2 speackers']
plot_confusion_matrix_from_data(y_total, y_prob_total_stat,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

num_classes=18 
cm_plot_labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'
                  ,'101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180']
plot_confusion_matrix_from_data(y2_total_plot-1,y2_prob_total_stat_plot-1,num_classes,cm_plot_labels,
  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

print()
