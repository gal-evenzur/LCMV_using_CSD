# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:38:14 2019

@author: shvarta3
"""


from __future__ import print_function
import numpy as np
#import math
import numpy.matlib




def istft(stft, awin, swin, hop, nfft, fs=16000):
    L = len(stft[0,:])     
    wlen = len(swin)       
    xlen = int(wlen + (L-1)*hop)  
    x = np.zeros((1,xlen),dtype=complex)       
    hop=int(hop)
                 
    X = np.zeros((nfft,L),dtype=complex)
    temp1=np.flipud(stft[1:-1,:]).conj()
    X=np.concatenate((stft,temp1))

    
    for l in range(0,L):
        xw_temp = np.fft.ifft(X[:,l]);
        xw=xw_temp.real
        x[:,int(l*hop):int(wlen+l*hop)] = x[:,int(l*hop):int(wlen+l*hop)]+(xw*swin).T;           
    t = np.linspace(0,xlen,xlen)/fs;    
    return np.squeeze(x.real),t
    