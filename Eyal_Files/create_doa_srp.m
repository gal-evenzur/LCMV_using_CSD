clear all
close all

fs = 16000; 
wlen = 2048; 
hop = wlen/4;  
nfft = 2048;   
NUP = ceil((nfft+1)/2);
M=7;
win=hamming(wlen, 'periodic');

mics_setup =  [1 0 1 0 1 0 1];
Noise_only = 0;
window = 2048;
reso=10;
p=[1 13];
%'C:\\project\\val_dynamic\\together_%d.wav'
%'C:\\project\\dynamic_signals\\one_speaker\\dynamic_signal_%d.wav'

for i=1:1
    data=sprintf('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_550_SNR_20\\dynamic_signal_%d.wav',i);
    signals = audioread(data);
    index=fix(1+((length(signals(:,1))-wlen)/hop));
    
    angle=360;
    t = linspace(-pi,0,angle/2); 
    circ_mics_x = 0.1*cos(t);
    circ_mics_y = -0.1*sin(t);

    r= [circ_mics_x(1) circ_mics_y(1); circ_mics_x(25) circ_mics_y(25); circ_mics_x(55) circ_mics_y(55); circ_mics_x(80) circ_mics_y(80);...
        circ_mics_x(120) circ_mics_y(120); circ_mics_x(150) circ_mics_y(150); circ_mics_x(180) circ_mics_y(180)];

    xlen = length(signals);
    L = 1+fix((xlen-wlen)/hop);
    signals_stft = zeros(NUP,L,M);
    for l = 1:M
      signals_stft(:,:,l) = my_stft(signals(:,l), win, hop, nfft);
    end
    
    [DOAs_out] = 18-SRP(signals_stft,r,mics_setup,fs,Noise_only, window,1,reso,p);
   
    label_doa_file=sprintf('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_550_SNR_20\\label_doa_%d.mat',i);
    save(label_doa_file,'DOAs_out')
    
end

plot(DOAs_out)
