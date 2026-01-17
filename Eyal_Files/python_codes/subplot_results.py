# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:31:27 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.io import wavfile
from pytohn_utills.stft import stft



plt.close("all")
main_dir = 'C:/Users/user/Desktop/thesis codes/'

folder_to_work = main_dir+'signals/SNR_15_T60_300_SIR_0/'
number_of_signal = 3
CSD=load_model(main_dir+'models/CSD.h5')
DOA=load_model(main_dir+'models/DOA.h5')


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


fs,temp1wav = wavfile.read(main_dir+'results/separating_speaker_number_0.wav')
fs,temp2wav = wavfile.read(main_dir+'results/separating_speaker_number_1.wav')

temp1_temp = stft(temp1wav, win, hop, nfft)
temp2_temp = stft(temp2wav, win, hop, nfft)

temp1temp=20*np.log10(abs(temp1_temp)+epsilon);
temp2temp=20*np.log10(abs(temp2_temp)+epsilon);


signal_first_file=(folder_to_work+'dynamic_signal_first_{}.wav'.format(number_of_signal))
signal_second_file=(folder_to_work+'dynamic_signal_second_{}.wav'.format(number_of_signal))
signal_file=(folder_to_work+'dynamic_signal_{}.wav'.format(number_of_signal))



fs,clean1wav = wavfile.read(signal_first_file)
fs,clean2wav = wavfile.read(signal_second_file)
fs,cleanwav = wavfile.read(signal_file)

clean1wav = clean1wav/max(clean1wav[:,0])
clean2wav = clean2wav/max(clean2wav[:,0])
cleanwav = cleanwav/max(cleanwav[:,0])

clean1spec = stft(clean1wav[:,0], win, hop, nfft)
clean2spec = stft(clean2wav[:,0], win, hop, nfft)
cleanspec = stft(cleanwav[:,0], win, hop, nfft)

clean1logspec=20*np.log10(abs(clean1spec)+epsilon);
clean2logspec=20*np.log10(abs(clean2spec)+epsilon);
cleanlogspec=20*np.log10(abs(cleanspec)+epsilon);

clean1logspec = clean1logspec[:,frame_before:clean1logspec.shape[1]-frame_after]
clean2logspec = clean2logspec[:,frame_before:clean2logspec.shape[1]-frame_after]
cleanlogspec = cleanlogspec[:,frame_before:cleanlogspec.shape[1]-frame_after]



clean1logspec=20*np.log10(abs(z_k[:,:,0].T)+epsilon);
clean2logspec=20*np.log10(abs(z_k[:,:,0].T)+epsilon);
cleanlogspec=20*np.log10(abs(z_k[:,:,0].T)+epsilon);
temp1temp=20*np.log10(abs(z_k[:,:,0].T)+epsilon);
temp2temp=20*np.log10(abs(z_k[:,:,0].T)+epsilon);

y_prob_dynamic = y_prob_total_stat
y = y_total
y2_for_plot = y2_total
y2_prob_for_plot = y2_prob_stat


fig, axs = plt.subplots(7, 2, 
                       gridspec_kw={
                           'width_ratios': [1,1],
                           'height_ratios': [1,1,1,1,1,2,6]})
fig.suptitle('    ', fontsize=6)
#fig, axs = plt.subplots(7,1,gridspec_kw={'width_ratios': [1,1,1,1,1,1,1]})
fig.set_figheight(6)
fig.set_figwidth(3)
fig.tight_layout(pad=0.25)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
im = axs[0,0].imshow(clean1logspec[::-1],aspect='auto',vmin=-20,vmax=20)
im = axs[1,0].imshow(clean2logspec[::-1],aspect='auto',vmin=-20,vmax=20)
im = axs[2,0].imshow(cleanlogspec[::-1],aspect='auto',vmin=-20,vmax=20)
im = axs[3,0].imshow(temp1temp[::-1],aspect='auto',vmin=-20,vmax=20)
im = axs[4,0].imshow(temp2temp[::-1],aspect='auto',vmin=-20,vmax=20)
cb_ax = fig.add_axes([0.47, 0.505, 0.01, 0.46])
cbar = fig.colorbar(im, cax=cb_ax)

line_up =axs[5,0].scatter(np.arange(len(y)),342*y,2*np.ones(len(y)),c='b',alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
line_down =axs[5,0].scatter(np.arange(len(y)),342*y_prob_dynamic,2*np.ones(len(y)),c='r',alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')
plt.legend(handles=[line_up,line_down],loc=(-20,-0.7))

colors1 = []
colors2 = []
for i,v in enumerate(y2_for_plot):
    if v==0.0:
       colors1.append('white')
       colors2.append('white')
    else:
       colors1.append('g')
       colors2.append('r')

line_up2 =axs[6,0].scatter(np.arange(len(y)),57*(y2_for_plot-1),2*np.ones(len(y)),c=colors1,alpha=0.5,linewidth=5,marker='o',label='Actual label - Speakers Detector')
line_down2 =axs[6,0].scatter(np.arange(len(y)),57*(y2_prob_for_plot-1),2*np.ones(len(y)),c=colors2,alpha=0.75,linewidth=2,marker='.',label='Predicted label - Speakers Detector')

axs[0,0].title.set_text('(a) First oracle speaker')
axs[1,0].title.set_text('(b) Second oracle speaker')
axs[2,0].title.set_text('(c) Input signal')
axs[3,0].title.set_text('(d) First output channel')
axs[4,0].title.set_text('(e) Second output channel')
axs[5,0].title.set_text('(f) CSD')
axs[6,0].title.set_text('(g) DOA')

axs[6,0].set_xlabel('Time(sec)')
axs[2,0].set_ylabel('Frequency(KHz) \n',fontsize=16)


axs[5,0].set_ylabel('Label',fontsize=14)
axs[5,0].set_yticks(np.arange(0,nfft/2,342))
axs[5,0].set_yticklabels(['Noise','One speaker','Many speakers'])
    
axs[6,0].set_ylabel('Center label',fontsize=14)
axs[6,0].set_yticks(np.arange(0,nfft/2,57))
axs[6,0].set_yticklabels(['5','15','25',
                               '35','45','55',
                               '65','75','85',
                          '95','105','115',
                         '125','135','145',
                         '155','165','175'],fontsize=10)

axs[0,0].set_yticks(np.arange(0,nfft/2,257))
axs[0,0].set_yticklabels(['8','6','4','2'],fontsize=9)

axs[1,0].set_yticks(np.arange(0,nfft/2,257))
axs[1,0].set_yticklabels(['8','6','4','2'],fontsize=9)

axs[2,0].set_yticks(np.arange(0,nfft/2,257))
axs[2,0].set_yticklabels(['8','6','4','2'],fontsize=9)

axs[3,0].set_yticks(np.arange(0,nfft/2,257))
axs[3,0].set_yticklabels(['8','6','4','2'],fontsize=9)

axs[4,0].set_yticks(np.arange(0,nfft/2,257))
axs[4,0].set_yticklabels(['8','6','4','2'],fontsize=9)

axs[6,0].set_xticks(np.arange(0,len(y),31.25))
axs[6,0].set_xticklabels(np.arange(0,int((len(y)+30)/31.25),1))
axs[6,0].margins(x=0) 
axs[5,0].margins(x=0)
#plt.subplots_adjust(wspace=0, hspace=0)
axs[0,0].set_xticks([])
axs[1,0].set_xticks([])
axs[2,0].set_xticks([])
axs[3,0].set_xticks([])
axs[4,0].set_xticks([])
axs[5,0].set_xticks([])
#axs[6].set_xticks([])

plt.show()
