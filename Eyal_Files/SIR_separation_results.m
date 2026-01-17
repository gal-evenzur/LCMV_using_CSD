clear all
close all
results_dir = 'C:\\project\\dynamic_signals\\two_speakers_long\\SNR\\SNR_20_T60_300_SIR_0\\results\\';
first_clean = audioread(strcat(results_dir,'first_clean.wav'));
second_clean = audioread(strcat(results_dir,'second_clean.wav'));
noisy_signal = audioread(strcat(results_dir,'noisy_signal.wav'));
separating_speaker_number_0 = audioread(strcat(results_dir,'separating_speaker_number_0.wav'));
separating_speaker_number_1 = audioread(strcat(results_dir,'separating_speaker_number_1.wav'));


% Ax_SIR = 1/(10^(10/20));
% first_clean_low = Ax_SIR*first_clean;
% snr(first_clean_low,noisy_signal)
% snr(separating_speaker_number_1,noisy_signal)

[SDR,SIR,SAR,perm]=bss_eval_soures(first_clean.',noisy_signal.');
