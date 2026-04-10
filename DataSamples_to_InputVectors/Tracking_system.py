
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tracking front-end for CSD/DOA estimation from multichannel STFT features.

This script builds GEVD-based spatial features and runs two pretrained models:
1) CSD classifier (noise / one speaker / overlap)
2) DOA classifier (18 angular bins)

Theory-to-code map (GEVD reminder for separation pipeline):
1) Estimate noise covariance Qvv per frequency from a noise-dominant segment.
2) Whiten each frame with L^{-1}, where L is Cholesky(Qvv).
3) Build local whitened PSD and take its dominant eigenvector phi.
4) Recover an RTF-like steering vector g = L phi / (reference normalization).
5) Feed g-derived features to tracking models; these tracked labels are later used
    by separation code to update GEVD/RTF estimates and form MVDR/LCMV filters.
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
import os


plt.close("all")
pydir = os.path.dirname(os.path.realpath(__file__))

# --------------------------------------------------------------------------------------
# Paths and experiment selection
# --------------------------------------------------------------------------------------
# Where to save the results of the tracking system:
folder_to_save = pydir + '/tracking_results/' 

# Where to take the data to work with:
# ! Need to change the path to use real data, the path is for example only.
forlder_to_work2 = os.path.dirname(pydir) + '/createAudio' + '/data' + '/val/'


RTF_mode = 'GEVD'

who = 'second' # Idx of speaker
idx = 1 # Idx of the file to work with, for example if idx=1, the file name is 'together_1.wav' and 'angle_location_second_1.mat'

# --------------------------------------------------------------------------------------
# Core configuration
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Model loading (trained offline)
# --------------------------------------------------------------------------------------
model3_path = os.path.join(pydir, 'models', 'model3_GEVD_30_3.h5')
model3=load_model(model3_path)
model18_path = os.path.join(pydir, 'models', 'model18_GEVD_30_3.h5')
model18=load_model(model18_path)

y_total=[]
y2_total=[]

y_prob_total_stat=[]
y2_prob_total_stat=[]

# --------------------------------------------------------------------------------------
# Input loading and normalization
# --------------------------------------------------------------------------------------
signal_file=(forlder_to_work2+'together_'+str(idx)+'.wav')
signal_first_file=(forlder_to_work2 + who+'_'+str(idx)+'.wav')
fs,receivers= wavfile.read(signal_file)
fs,receiver_first = wavfile.read(signal_first_file)
receivers=receivers/(abs(receivers).max())
receiver_first=receiver_first/(abs(receiver_first).max())
M=len(receiver_first[0,:])
index=int(1+np.fix((len(receiver_first[:,1])-wlen)/hop))



angle_location_first=(forlder_to_work2+'label_location_'+who+'_'+str(idx)+'.npy')

# Load location labels (stored as a pickled dict-like structure in this dataset).
mat_locations=np.load(angle_location_first, allow_pickle=True)
location_first=np.transpose(mat_locations['label_'+who])




y2_first = np.zeros((index,1))

# Framewise ground-truth DOA label by majority vote inside each STFT window.
for l in range(0,index):
    locations_temp_first = location_first[int(l*hop):int(wlen+l*hop)]
    angle_frame_first = np.bincount(np.squeeze(locations_temp_first)).argmax()
    y2_first[l] = angle_frame_first

# --------------------------------------------------------------------------------------
# STFT construction
# --------------------------------------------------------------------------------------
z_k = np.zeros((M,NUP,index),dtype=complex)
for i in range(M):
    z_k[i,:,:] = stft(receivers[:,i], win, hop, nfft)

z_k_first = np.zeros((M,NUP,index),dtype=complex)
for i in range(M):
    z_k_first[i,:,:] = stft(receiver_first[:,i], win, hop, nfft)

# --------------------------------------------------------------------------------------
# GEVD feature extraction
# --------------------------------------------------------------------------------------
# Step 1: estimate Qvv from a fixed noise-dominant segment and factorize Qvv = L L^H.
cholesky_Qvv = np.zeros((NUP,M,M),dtype=complex)
for i in range(0,NUP):
    PSD_tmp =z_k[:,i,450:500]@(z_k[:,i,450:500].conj().T)/50
    cholesky_Qvv[i,:,:] = LA.cholesky(PSD_tmp)

X = np.zeros((index-frame_before-frame_after,NUP,M),dtype=complex)
x_index = 0
for l in range(frame_before,index-frame_after):
    for j in range(NUP):
        # Step 2: whitening operator L^{-1} (with small regularization).
        chol_j=LA.inv(cholesky_Qvv[j,:,:]+epsilon*np.eye(M)*(LA.norm(cholesky_Qvv[j,:,:])))
        Zvv = 0
        sum_win_vad = 0
        for p in range(frame_before+frame_after+1):
            # Local context window around frame l to stabilize RTF estimation.
            temp_zvv = chol_j@(win_vad[10-frame_before+p]*z_k[:,j,l-frame_before+p].reshape(M,1))
            Zvv = Zvv+temp_zvv@temp_zvv.conj().T
            sum_win_vad = sum_win_vad+win_vad[10-frame_before+p]
        Zvv=Zvv/sum_win_vad

        # Step 3: dominant eigenvector of whitened PSD (GEVD interpretation).
        w,v = LA.eig(Zvv)
        fi=v[:,w.argmax()].reshape(M,1)

        # Step 4: recolor/normalize to an RTF-like steering vector.
        # The first microphone is used as reference normalization.
        # This g-style vector is later used by separation as:
        #   w_mvdr = Qvv^{-1} g / (g^H Qvv^{-1} g)
        #   W_lcmv = Qvv^{-1} G (G^H Qvv^{-1} G)^{-1}
        denominator=cholesky_Qvv[j,0,:].reshape(1,M)@fi
        G_cw=np.squeeze(cholesky_Qvv[j,:,:])@fi/denominator
        X[x_index,j,:] = np.squeeze(G_cw)
    x_index+=1

# The models were trained on real/imag parts of non-reference channels +
# log-magnitude of reference channel.
X_T=np.concatenate((X[:,:,1:M].real,X[:,:,1:M].imag),axis=2)
for b in range(len(X_T)):
    X_T[b,:,:] = scaler.fit_transform(X_T[b,:,:])


z_k_0=z_k[0,:,frame_before:index-frame_after].T
z_k_0=np.log(abs(z_k_0))
z_k_0_standart = scaler.fit_transform(z_k_0)
z_k_0_standart=np.reshape(z_k_0_standart, (z_k_0_standart.shape[0],NUP,1))


x_test=np.concatenate((X_T,z_k_0_standart),axis=2)

# --------------------------------------------------------------------------------------
# VAD-derived labels for evaluation
# --------------------------------------------------------------------------------------
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
# Label 19 is reserved as a sentinel for overlap/non-single-speaker frames.
y2=np.where(y!=2, y2,19)

z_k= np.transpose(z_k, (2,1,0))
z_k= z_k[frame_before:index-frame_after,:,:]

# --------------------------------------------------------------------------------------
# Model inference: CSD + DOA tracking outputs
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Confusion-matrix reporting
# --------------------------------------------------------------------------------------
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
    



