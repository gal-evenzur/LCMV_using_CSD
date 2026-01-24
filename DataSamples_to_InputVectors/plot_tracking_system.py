
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Thu Apr 22 14:40:28 2021

@author: shvarta3
"""

import numpy as np
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
import matplotlib.pyplot as plt


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


forlder_to_work2 = 'C:/project/dynamic_signals/two_speakers_real_recording/'
folder_to_save = forlder_to_work2+'results/'

y2_prob_stat_mf = np.load(folder_to_save+'estimate_DOA_1.npy')
y2 = np.load(folder_to_save+'true_DOA_1.npy')
y_prob_stat_mf = np.load(folder_to_save+'estimate_CSD_1.npy')
y_mf = np.load(folder_to_save+'true_CSD_1.npy')

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


plt.figure()
plt.plot(y2_prob_total_stat_plot)
plt.plot(y2_total_plot)
