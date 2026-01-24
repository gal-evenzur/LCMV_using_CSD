# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:46:09 2019

@author: shvarta3
"""

from __future__ import print_function
import numpy as np




def stft(x, win, hop, nfft):
 #   x = x(:); 
    xlen = len(x);
    wlen = len(win);
    NUP = int(np.ceil((nfft+1)/2));  
    L = int(1+np.fix((xlen-wlen)/hop));
    STFT = np.zeros((L,NUP),dtype=complex);
    for l in range(0,L):
        x_temp = x[int(l*hop):int(wlen+l*hop)]
        x_w = x_temp*win;
        X = np.fft.fft(x_w)
        STFT[l,:] = X[0:NUP].T
    #FIXME- check maybe we should add one, look if it's the same size as in matlab
    STFT=STFT.T;
    return STFT