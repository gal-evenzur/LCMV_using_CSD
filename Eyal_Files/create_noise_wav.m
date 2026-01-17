clear all
close all
addpath('C:\project\my_stft.m');
addpath('C:\project\my_istft.m');


fs = 16000; 
data=sprintf('C:\\project\\noise\\Car.wav');
[togheter,Fs1] = audioread(data);
togheter = interp(togheter,fs/Fs1);
data_file=sprintf('C:\\project\\noise\\noise_1.wav');
audiowrite(data_file,togheter(1:500000),fs)
P_N_H = mean((togheter(1:320000).^2),'all');


data=sprintf('C:\\project\\noise\\Factory.wav');
[togheter,Fs3] = audioread(data);
togheter = togheter(:,1) ;
togheter = interp(togheter',fs/Fs3);
data_file=sprintf('C:\\project\\noise\\noise_2.wav');
audiowrite(data_file,togheter(1:500000),fs)
P_N_H = mean((togheter(1:320000).^2),'all');

data=sprintf('C:\\project\\noise\\Room.wav');
[togheter,Fs4] = audioread(data);
togheter = interp(togheter,fs/Fs4);
data_file=sprintf('C:\\project\\noise\\noise_3.wav');
audiowrite(data_file,togheter(1:500000),fs)
P_N_H = mean((togheter(1:320000).^2),'all');

data=sprintf('C:\\project\\noise\\Speech.wav');
[togheter,Fs5] = audioread(data);
togheter = interp(togheter,fs/Fs5);
data_file=sprintf('C:\\project\\noise\\noise_4.wav');
audiowrite(data_file,togheter(1:500000),fs)
P_N_H = mean((togheter(1:320000).^2),'all');