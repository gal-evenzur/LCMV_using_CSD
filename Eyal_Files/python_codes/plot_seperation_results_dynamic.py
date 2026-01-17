# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 08:42:24 2021

@author: user
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
plt.close("all")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.autolayout"] = True

type_r = 'SNR'
type_stat = 'static_signals'
folder_to_save = 'C:/project/'+type_stat+'/two_speakers_WSJ/'+type_r+'/'

# stoi_alg=np.load(folder_to_save+'stoi_alg.npy')
# stoi_noisy=np.load(folder_to_save+'stoi_noisy.npy')
# pesq_alg=np.load(folder_to_save+'pesq_alg.npy')
# pesq_noisy=np.load(folder_to_save+'pesq_noisy.npy')
# sir_alg=np.load(folder_to_save+'sir_alg.npy')
# sir_noisy=np.load(folder_to_save+'sir_noisy.npy')
# sdr_alg=np.load(folder_to_save+'sdr_alg.npy')
# sdr_noisy=np.load(folder_to_save+'sdr_noisy.npy')

stoi_alg_mapping=np.array([94,95,96,98,98.5])
stoi_noisy_mapping=np.array([65,67,67,68,71])
sisdr_alg=np.array([7.8,8.3,8.5,8.6,8.8])
sir_alg=np.array([12,13,14,14.3,14.5])

# def mapping_stoi(d,a,b):
#     return 100/(1+np.exp(a*d+b))

# a = -17.49
# b = 9.69

# stoi_alg_mapping = mapping_stoi(stoi_alg,a,b)
# stoi_noisy_mapping = mapping_stoi(stoi_noisy,a,b)

# mir_eval_plot  ##############################################

if type_r == 'T60':
    range_x = np.arange(300,551,100)
if type_r == 'overlap':
    range_x = np.arange(0,76,25)
if type_r == 'SIR':
    range_x = -np.arange(0,11,5)
if type_r == 'SNR':
    range_x = np.arange(10,21,2.5)




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



fig, axs = plt.subplots(3,3,figsize=(screen_width / 100, screen_height / 100))
# fig.suptitle('LCMV results  - dynamic signals')

line_1, = axs[0,0].plot(stoi_alg_mapping,'-bo', label='Proposed')
line_2, = axs[0,0].plot(stoi_noisy_mapping, '-y^',label='Noisy')
axs[0,0].legend(handles=[line_1,line_2])
axs[0,0].set_ylim([0,100])
axs[0,0].title.set_text('STOI test')
axs[0,0].set_xlabel(type_r)
axs[0,0].set_ylabel('Percent')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[0,0])
axs[0,0].set_xticks(xi)
axs[0,0].set_xticklabels(range_x)
# plt.xticks(xi,range_x)

line_1, = axs[0,1].plot(sisdr_alg, '-bo',label='Proposed')
# line_2, = axs[0,1].plot(sisdr_noisy, '-y^',label='Noisy')
axs[0,1].legend(handles=[line_1])
axs[0,1].set_ylim([5,15])
axs[0,1].set_xlabel(type_r)
axs[0,1].title.set_text('Delta SI-SDR')
axs[0,1].set_ylabel('dB')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[0,1])
axs[0,1].set_xticks(xi)
axs[0,1].set_xticklabels(range_x)
# plt.xticks(xi,range_x)

if type_r=="SIR":
    line_1, = axs[0,2].plot((sir_alg), label='Proposed')
else:
    line_1, = axs[0,2].plot((sir_alg),'-bo', label='Proposed')
axs[0,2].legend(handles=[line_1])
axs[0,2].set_ylim([5, 20])
axs[0,2].set_xlabel(type_r)
axs[0,2].set_ylabel('dB')
axs[0,2].title.set_text('Delta SIR')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[0,2])
axs[0,2].set_xticks(xi)
axs[0,2].set_xticklabels(range_x)
# plt.xticks(xi,range_x)

type_r = 'T60 (sec)'
# folder_to_save = 'C:/project/'+type_stat+'/two_speakers_WSJ/'+type_r+'/'

# stoi_alg=np.load(folder_to_save+'stoi_alg.npy')
# stoi_noisy=np.load(folder_to_save+'stoi_noisy.npy')
# pesq_alg=np.load(folder_to_save+'pesq_alg.npy')
# pesq_noisy=np.load(folder_to_save+'pesq_noisy.npy')
# sir_alg=np.load(folder_to_save+'sir_alg.npy')
# sir_noisy=np.load(folder_to_save+'sir_noisy.npy')
# sdr_alg=np.load(folder_to_save+'sdr_alg.npy')
# sdr_noisy=np.load(folder_to_save+'sdr_noisy.npy')

stoi_alg_mapping=np.array([98.5,95,86])
stoi_noisy_mapping=np.array([71,67,60])
sisdr_alg=np.array([8.8,7,6.5])
sir_alg=np.array([14.5,11,9])

if type_r == 'T60 (sec)':
    range_x = np.arange(0.3,0.551,0.1)
if type_r == 'overlap':
    range_x = np.arange(0,76,25)
if type_r == 'SIR':
    range_x = -np.arange(0,11,5)
if type_r == 'SNR':
    range_x = np.arange(10,21,2.5)

# stoi_alg_mapping = mapping_stoi(stoi_alg,a,b)
# stoi_noisy_mapping = mapping_stoi(stoi_noisy,a,b)

line_1, = axs[1,0].plot(stoi_alg_mapping, '-bo',label='Proposed')
line_2, = axs[1,0].plot(stoi_noisy_mapping, '-y^',label='Noisy')
axs[1,0].legend(handles=[line_1,line_2])
axs[1,0].set_ylim([0,100])
axs[1,0].set_xlabel(type_r)
axs[1,0].set_ylabel('Percent')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[1,0])
axs[1,0].set_xticks(xi)
axs[1,0].set_xticklabels(range_x)
# plt.xticks(xi,range_x)


line_1, = axs[1,1].plot(sisdr_alg, '-bo',label='Proposed')
axs[1,1].legend(handles=[line_1])
axs[1,1].set_ylim([5,15])
axs[1,1].set_xlabel(type_r)
axs[1,1].set_ylabel('dB')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[1,1])
axs[1,1].set_xticks(xi)
axs[1,1].set_xticklabels(range_x)
# plt.xticks(xi,range_x)


if type_r=="SIR":
    line_1, = axs[1,2].plot((sir_alg), label='Proposed')
else:
    line_1, = axs[1,2].plot((sir_alg),'-bo', label='Proposed')
axs[1,2].legend(handles=[line_1])
axs[1,2].set_ylim([5, 20])
axs[1,2].set_xlabel(type_r)
axs[1,2].set_ylabel('dB')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[1,2])
axs[1,2].set_xticks(xi)
axs[1,2].set_xticklabels(range_x)
# plt.xticks(xi,range_x)
  

type_r = 'SIR'
folder_to_save = 'C:/project/'+type_stat+'/two_speakers_WSJ/'+type_r+'/'


stoi_alg_mapping=np.array([98.5,88,50])
stoi_noisy_mapping=np.array([71,25,10])
sisdr_alg=np.array([8.8,7.5,7.2])
sir_alg=np.array([14.5,14,13.8])

# stoi_alg=np.load(folder_to_save+'stoi_alg.npy')
# stoi_noisy=np.load(folder_to_save+'stoi_noisy.npy')
# pesq_alg=np.load(folder_to_save+'pesq_alg.npy')
# pesq_noisy=np.load(folder_to_save+'pesq_noisy.npy')
# sir_alg=np.load(folder_to_save+'sir_alg.npy')
# sir_noisy=np.load(folder_to_save+'sir_noisy.npy')
# sdr_alg=np.load(folder_to_save+'sdr_alg.npy')
# sdr_noisy=np.load(folder_to_save+'sdr_noisy.npy')


# stoi_alg_mapping = mapping_stoi(stoi_alg,a,b)
# stoi_noisy_mapping = mapping_stoi(stoi_noisy,a,b)

if type_r == 'T60':
    range_x = np.arange(300,551,100)
if type_r == 'overlap':
    range_x = np.arange(0,76,25)
if type_r == 'SIR':
    range_x = -np.arange(0,11,5)
if type_r == 'SNR':
    range_x = np.arange(10,21,2.5)


line_1, = axs[2,0].plot(stoi_alg_mapping,'-bo', label='Proposed')
line_2, = axs[2,0].plot(stoi_noisy_mapping, '-y^',label='Noisy')
axs[2,0].legend(handles=[line_1,line_2])
axs[2,0].set_ylim([0,100])
axs[2,0].set_xlabel('SIR in')
axs[2,0].set_ylabel('Percent')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[2,0])
axs[2,0].set_xticks(xi)
axs[2,0].set_xticklabels(range_x)
# plt.xticks(xi,range_x)


line_1, = axs[2,1].plot(sisdr_alg,'-bo', label='Proposed')
axs[2,1].legend(handles=[line_1])
axs[2,1].set_ylim([5,15])
axs[2,1].set_xlabel('SIR in')
axs[2,1].set_ylabel('dB')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[2,1])
axs[2,1].set_xticks(xi)
axs[2,1].set_xticklabels(range_x)
# plt.xticks(xi,range_x)

if type_r=="SIR":
    line_1, = axs[2,2].plot(sir_alg,'-bo', label='Proposed')
else:
    line_1, = axs[2,2].plot(sir_alg, label='Proposed')
axs[2,2].legend(handles=[line_1])
axs[2,2].set_ylim([5, 20])
axs[2,2].set_xlabel('SIR in')
axs[2,2].set_ylabel('dB')
xi = list(range(len(stoi_alg_mapping)))
plt.sca(axs[2,2])
axs[2,2].set_xticks(xi)
axs[2,2].set_xticklabels(range_x)
# plt.xticks(xi,range_x)
axs[0,0].grid()
axs[0,1].grid()
axs[0,2].grid()
axs[1,0].grid()
axs[1,1].grid()
axs[1,2].grid()
axs[2,0].grid()
axs[2,1].grid()
axs[2,2].grid()
fig.tight_layout(pad=-0.2)
# fig.set_size_inches(30,10)

plt.savefig('dynamic_results.png',dpi=300)

print()

#np.save(folder_to_save+'stoi_alg.npy', stoi_alg)
#np.save(folder_to_save+'stoi_noisy.npy', stoi_noisy)
#np.save(folder_to_save+'pesq_alg.npy', pesq_alg)
#np.save(folder_to_save+'pesq_noisy.npy', pesq_noisy)
#np.save(folder_to_save+'sir_alg.npy', sir_alg)
#np.save(folder_to_save+'sir_noisy.npy', sir_noisy)
#np.save(folder_to_save+'sdr_alg.npy', sdr_alg)
#np.save(folder_to_save+'sdr_noisy.npy', sdr_noisy) 
