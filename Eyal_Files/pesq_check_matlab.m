folder_to_work = 'C:\project\dynamic_signals\two_speakers_WSJ\SNR\SNR_20_T60_300_SIR_0\results_overlap\';
first_clean = strcat(folder_to_work,'first_clean_10.wav');
second_clean = strcat(folder_to_work,'second_clean_10.wav');
first_est = strcat(folder_to_work,'separating_speaker_number_0_10.wav');
second_est = strcat(folder_to_work,'separating_speaker_number_1_10.wav');
noisy = strcat(folder_to_work,'noisy_signal_10.wav');

pesq(first_clean,first_est,+8000)
pesq(second_clean,second_est,+8000)

pesq(first_clean,noisy,+8000)
pesq(second_clean,noisy,+8000)