

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from stft import stft
from istft import istft
from numpy import linalg as LA
from scipy.io.wavfile import write
import logging
from pystoi import stoi
from pesq import pesq
import mir_eval
from scipy.io import wavfile
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


k=3
RTF_mode = 'GEVD'

nom_data_sets=11          ################ amount of signals to test ###################
nfft=2048
wlen = 2048                                                                     
hop = wlen/4 
num_classes = 3
new = 1
img_rows, img_cols = 1025,7
NUP=1025                                          
win=np.hamming(wlen)
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
threshold_chage_location = 8
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

forlder_to_work2 = 'C:/project/dynamic_signals/two_speakers_real_recording/'
folder_to_save = forlder_to_work2+'results/'

y2_prob_stat_mf = np.load(folder_to_save+'estimate_DOA_3.npy')
y2 = np.load(folder_to_save+'true_DOA_3.npy')
y_prob_stat_mf = np.load(folder_to_save+'estimate_CSD_3.npy')
y_mf = np.load(folder_to_save+'true_CSD_3.npy')

z_k = 0
M = 4

y_prob_dynamic = np.zeros(len(y2))
y2_prob_dynamic = np.zeros(len(y2))    
Qvv_temp=np.zeros((NUP,M,M),complex)
Frame_classification_system = np.zeros((2,3))
G = np.ones((NUP,M,2),dtype=complex)
Qvv = np.zeros((NUP,M,M))
for j in range(NUP):
    Qvv[j,:,:] = np.eye(M)

W=np.ones((len(y2),NUP,M,2),complex)

PSD_matrix_per_DOA = np.zeros((18,NUP,M,M),complex)
total_frame_per_DOA = np.zeros((18))
stand_z = []


signal_file=(forlder_to_work2+'signal_%d.wav'%k)

signal_first_file=(forlder_to_work2+'signal_first_%d.wav'%k)
signal_second_file=(forlder_to_work2+'signal_second_%d.wav'%k)

signal_noise_file=(forlder_to_work2+'signal_noise_%d.wav'%k)


############################### create input to the model ##############################################

flag_start_lcmv = 1

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

flag_first_noise = 0
M=len(receiver_first[0,:])
index=int(1+np.fix((len(receiver_first[:,1])-wlen)/hop))

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

z_k= np.transpose(z_k, (2,1,0))
z_k= z_k[frame_before:index-frame_after,:,:]

z_k_first= z_k_first[:,:,frame_before:index-frame_after]
z_k_second= z_k_second[:,:,frame_before:index-frame_after]
z_k_noise= z_k_noise[:,:,frame_before:index-frame_after]

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=forlder_to_work2+'/results/log_files/info_3.log',level=logging.DEBUG)    

save_last_frames_first = np.zeros((32,1025,4),dtype=complex)
save_last_frames_doa_first = np.zeros(32)
save_last_frames_second = np.zeros((32,1025,4),dtype=complex)
save_last_frames_doa_second = np.zeros(32)

for l in range(len(y2)):
    y_prob=y_prob_stat_mf[l]
    logging.info('frame number %d CSD association: %d'%(l,y_prob))
    
    if y_prob==0:
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

    if y_prob==1:
        if l<950: 
            y2_prob=y2_prob_stat_mf[l]

            if abs(y2_prob - Frame_classification_system[0,0])<3:
                if Frame_classification_system[0,0]!=0:
                    save_last_frames_first = np.roll(save_last_frames_first,1,axis=0)
                    save_last_frames_first[0] = z_k[l]
                    save_last_frames_doa_first = np.roll(save_last_frames_doa_first,1)
                    save_last_frames_doa_first[0] = y2_prob

            elif abs(y2_prob - Frame_classification_system[0,1])<3:
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

                current_save_last_frames_first = current_save_last_frames_first[np.where(abs(current_save_last_frames_doa_first-y2_prob)<3)]
                
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
start_overlap = np.nonzero(y_mf == 2)[0][0]
finish_overlap = np.nonzero(y_mf == 2)[0][-1]

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

sir_noisy = sir_noise.mean()
sdr_noisy = sdr_noise.mean()

###################################### alg results ##################################################3    

(sdr,sir,sar,perm)  = mir_eval.separation.bss_eval_sources(reference_sources.T+10**(-9), estimated_sources.T, compute_permutation=True)

sir_alg = sir.mean()
sdr_alg = sdr.mean()

first_stoi = max(stoi(z_k_first_one_speaker_time,s_hat_first_one_speaker_time,fs),stoi(z_k_first_one_speaker_time,s_hat_second_one_speaker_time,fs))
second_stoi = max(stoi(z_k_second_one_speaker_time,s_hat_first_one_speaker_time,fs),stoi(z_k_second_one_speaker_time,s_hat_second_one_speaker_time,fs))

first_stoi_noisy = stoi(z_k_first_one_speaker_time,z_k_noise_time,fs)
second_stoi_noisy = stoi(z_k_second_one_speaker_time,z_k_noise_time,fs)

first_pesq = max(pysepm.pesq(z_k_first_one_speaker_time,s_hat_first_one_speaker_time,fs)[1],pysepm.pesq(z_k_first_one_speaker_time,s_hat_second_one_speaker_time,fs)[1])
second_pesq = max(pysepm.pesq(z_k_second_one_speaker_time,s_hat_first_one_speaker_time,fs)[1],pysepm.pesq(z_k_second_one_speaker_time,s_hat_second_one_speaker_time,fs)[1])

first_pesq_noisy = pysepm.pesq(z_k_first_one_speaker_time,z_k_noise_time,fs)[1]
second_pesq_noisy = pysepm.pesq(z_k_second_one_speaker_time,z_k_noise_time,fs)[1]

#        separation = (first_separation+second_separation)/2
#        separation_total = np.append(separation_total,separation)
#    
stoi_alg = (first_stoi+second_stoi)/2
pesq_alg = (first_pesq+second_pesq)/2   
stoi_noisy = (first_stoi_noisy+second_stoi_noisy)/2
pesq_noisy = (first_pesq_noisy+second_pesq_noisy)/2

################################### save prediction to confusion matrix ######################

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

z_k_noise_time = torch.tensor(z_k_noise_time).unsqueeze(0)
z_k_first_one_speaker_time = torch.tensor(z_k_first_one_speaker_time).unsqueeze(0)
z_k_second_one_speaker_time = torch.tensor(z_k_second_one_speaker_time).unsqueeze(0)
s_hat_first_one_speaker_time = torch.tensor(s_hat_first_one_speaker_time).unsqueeze(0)
s_hat_second_one_speaker_time = torch.tensor(s_hat_second_one_speaker_time).unsqueeze(0)

si_sdr_input = (si_sdr_torchaudio_calc(z_k_noise_time, z_k_first_one_speaker_time)+si_sdr_torchaudio_calc(z_k_noise_time, z_k_second_one_speaker_time))/2
si_sdr_pred_first = torch.max(si_sdr_torchaudio_calc(s_hat_first_one_speaker_time, z_k_first_one_speaker_time),si_sdr_torchaudio_calc(s_hat_second_one_speaker_time, z_k_first_one_speaker_time))
si_sdr_pred_second = torch.max(si_sdr_torchaudio_calc(s_hat_first_one_speaker_time, z_k_second_one_speaker_time),si_sdr_torchaudio_calc(s_hat_second_one_speaker_time, z_k_second_one_speaker_time))
si_sdr_pred = (si_sdr_pred_second+si_sdr_pred_first)/2

print()
############# overlap area only #####################################        

# file_temp=(forlder_to_work2+'results_overlap/separating_speaker_number_0_{}.wav'.format(k))
# write(file_temp, fs,s_hat_first_one_speaker_time)
# file_temp=(forlder_to_work2+'results_overlap/separating_speaker_number_1_{}.wav'.format(k))
# write(file_temp, fs,s_hat_second_one_speaker_time)
# file_temp=(forlder_to_work2+'results_overlap/noisy_signal_%d.wav'%k)
# write(file_temp, fs,z_k_noise_time)
# file_temp=(forlder_to_work2+'results_overlap/first_clean_%d.wav'%k)
# write(file_temp, fs,z_k_first_one_speaker_time)
# file_temp=(forlder_to_work2+'results_overlap/second_clean_%d.wav'%k)
# write(file_temp, fs,z_k_second_one_speaker_time) 
