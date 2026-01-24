# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:31:27 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.io import wavfile
from stft import stft
# from pytohn_utills.stft import stft

plt.close("all")
main_dir = 'C:/project/dynamic_signals/two_speakers_real_recording/results/'

folder_to_work = main_dir
number_of_signal = 3
# CSD=load_model(main_dir+'models/CSD.h5')
# DOA=load_model(main_dir+'models/DOA.h5')

########## STFT variables

nfft=2048
wlen = 2048                                                                     
hop = wlen/4 
new = 0
NUP=1025                                          
win=np.hamming(wlen)
noise_pad_frames=30
epsilon=0.0001
big_epsilon = 0.01
frame_before = 8
frame_after = 5


fs,temp1wav = wavfile.read(main_dir+'separating_speaker_number_0_{}.wav'.format(number_of_signal))
fs,temp2wav = wavfile.read(main_dir+'separating_speaker_number_1_{}.wav'.format(number_of_signal))

temp1_temp = stft(temp1wav, win, hop, nfft)
temp2_temp = stft(temp2wav, win, hop, nfft)

temp1temp=20*np.log10(abs(temp1_temp)+epsilon);
temp2temp=20*np.log10(abs(temp2_temp)+epsilon);

signal_first_file=(folder_to_work+'first_clean_{}.wav'.format(number_of_signal))
signal_second_file=(folder_to_work+'second_clean_{}.wav'.format(number_of_signal))
signal_file=(folder_to_work+'noisy_signal_{}.wav'.format(number_of_signal))

fs,clean1wav = wavfile.read(signal_first_file)
fs,clean2wav = wavfile.read(signal_second_file)
fs,cleanwav = wavfile.read(signal_file)


clean1spec = stft(clean1wav, win, hop, nfft)
clean2spec = stft(clean2wav, win, hop, nfft)
cleanspec = stft(cleanwav, win, hop, nfft)

clean1logspec=20*np.log10(abs(clean1spec)+epsilon);
clean2logspec=20*np.log10(abs(clean2spec)+epsilon);
cleanlogspec=20*np.log10(abs(cleanspec)+epsilon);

clean1logspec = clean1logspec[:,frame_before:clean1logspec.shape[1]-frame_after]
clean2logspec = clean2logspec[:,frame_before:clean2logspec.shape[1]-frame_after]
cleanlogspec = cleanlogspec[:,frame_before:cleanlogspec.shape[1]-frame_after]


y2_prob_stat_mf = np.load(folder_to_work+'estimate_DOA_{}.npy'.format(number_of_signal))
y2 = np.load(folder_to_work+'true_DOA_{}.npy'.format(number_of_signal))
y_prob_stat_mf = np.load(folder_to_work+'estimate_CSD_{}.npy'.format(number_of_signal))
y_mf = np.load(folder_to_work+'true_CSD_{}.npy'.format(number_of_signal))



y_prob_dynamic = y_prob_stat_mf
y = y_mf
y2_for_plot = y2
y2_prob_for_plot = y2_prob_stat_mf


cleanlogspec = cleanlogspec[...,:1500]
y = y[:1500]
y_prob_dynamic = y_prob_dynamic[:1500]
y2_for_plot = y2_for_plot[:1500]
y2_prob_for_plot = y2_prob_for_plot[:1500]


y_to_multi_y2 = y>0
y_to_multi_y2 = y_to_multi_y2.astype(np.int32)
y2_for_plot = y2_for_plot*y_to_multi_y2

y2_prob_for_plot[np.where(y2_for_plot==0)[0]]=0
y2_for_plot[np.where(y2_for_plot==0)]=0
y2_prob_for_plot[np.where(y2_for_plot==19)]=0
y2_for_plot[np.where(y2_for_plot==19)]=0


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 10})

c_bar_loc = [0.95, 0.655, 0.01, 0.25]
loc_legend = (0.2,0.05)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

# Get the screen size (you may need to install the `screeninfo` library)
import subprocess
subprocess.check_call(["pip", "install", 'screeninfo'])
from screeninfo import get_monitors

# Determine the screen dimensions
screen = get_monitors()[0]  # Assuming you have only one monitor
screen_width = screen.width
screen_height = screen.height



fig, axs = plt.subplots(3,figsize=(screen_width / 300, screen_height / 100))

fig.suptitle('    ', fontsize=12)
#fig, axs = plt.subplots(7,1,gridspec_kw={'width_ratios': [1,1,1,1,1,1,1]})

# fig.tight_layout(pad=0.25)
im = axs[0].imshow(cleanlogspec[::-1],aspect='auto',vmin=-40,vmax=40)
# cb_ax = fig.add_axes(c_bar_loc)
# cbar = fig.colorbar(im, cax=cb_ax)

line_up =axs[1].scatter(np.arange(len(y)),342*y,2*np.ones(len(y)),c='b',alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
line_down =axs[1].scatter(np.arange(len(y)),342*y_prob_dynamic,2*np.ones(len(y)),c='r',alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')
axs[2].legend(handles=[line_up,line_down],fontsize = '14',loc = loc_legend)

colors1 = []
colors2 = []
for i,v in enumerate(y2_for_plot):
    if v==0.0:
       colors1.append('white')
       colors2.append('white')
    else:
       colors1.append('b')
       colors2.append('r')

line_up2 =axs[2].scatter(np.arange(len(y)),57*(y2_for_plot-1),2*np.ones(len(y)),c=colors1,alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
line_down2 =axs[2].scatter(np.arange(len(y)),57*(y2_prob_for_plot-1),2*np.ones(len(y)),c=colors2,alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')

axs[0].set_title('(a) Input signal',fontsize=14)
axs[1].set_title('(b) CSD',fontsize=14)
axs[2].set_title('(c) DOA',fontsize=14)

axs[2].set_xlabel('Time(sec)',fontsize=14)
axs[0].set_ylabel('Frequency(KHz) \n',fontsize=14)

axs[1].set_ylabel('Label',fontsize=14)
axs[1].set_yticks(np.arange(0,nfft/2,342))
axs[1].set_yticklabels(['1','2','3'],fontsize=14)
    
axs[2].set_ylabel('Center label',fontsize=14)
axs[2].set_yticks(np.arange(0,nfft/2,57*2))
axs[2].set_yticklabels(['5','25','45','65','85','105','125','145','165'],fontsize=14)

axs[0].set_yticks(np.arange(0,nfft/2,257))
axs[0].set_yticklabels(['8','6','4','2'],fontsize=14)

axs[2].set_xticks(np.arange(0,len(y),31.25*5))
axs[2].set_xticklabels(np.arange(0,int((len(y)+30)/31.25),5),fontsize=14)
axs[2].margins(x=0) 
axs[1].margins(x=0)
#plt.subplots_adjust(wspace=0, hspace=0)
axs[0].set_xticks([])
axs[1].set_xticks([])
#axs[6].set_xticks([])
plt.tight_layout()
plt.savefig('separation_results.png',dpi=300)
plt.show()

