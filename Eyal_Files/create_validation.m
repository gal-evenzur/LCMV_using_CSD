clear all
close all
load handel.mat
%% variens

clear all
%this is the radios of the table.

c_k = 340;                                        % Sound velocity (m/s)
fs = 16000;
n = 4096;                                         % Number of samples
mtype = 'omnidirectional';                        % Type of microphone
order = -1;                                       % -1 equals maximum reflection order!
dim = 3;                                          % Room dimension
orientation = 0;                                  % Microphone orientation (rad)
hp_filter = 1;                                    % Disable high-pass filter
% define analysis parameters
NO_S_i=10;
lottery=2000;
R=1.3;
pad=25000;
start=1;
max1=100;
noise_R=0.2;
flag=0;
flag2=0;
nfft=2048;
hop=512;
M=4;
num_jumps=9;
%% rir generation & covariens sounds 
for i=1:50
    close all
    % define room dimension
    L1_temp=randi([1,20]);
    L1=4+0.1*L1_temp;
    L2_temp=randi([1,20]);
    L2=4+0.1*L2_temp;
    L = [L1 L2 3];
    
    SNR_difuse=10+randi([0,10]); 
    beta_temp=randi([0,250]);
    beta=0.3+0.001*beta_temp;
    
    [s_first,label_first,s_second,label_second,s_noise,r]=create_locations_18_dynamic(L,R,noise_R,num_jumps);
    
    h_noise = rir_generator(c_k, fs, r, s_noise, L, beta, n, mtype, order, dim, orientation, hp_filter);
    
    male_female_check=randi([1,2]);
    
    if male_female_check==1
       male_female1='male'; 
       NO_S=112;
    else
       male_female1='female'; 
       NO_S=56;
    end
    
    source=randi([1,NO_S]);
    source1 = strcat(male_female1,'_',int2str(source));

    
    male_female_check=randi([1,2]);
    if male_female_check==1
       male_female2='male'; 
       NO_S=112;
    else
       male_female2='female'; 
       NO_S=56;
    end    
    source=randi([1,NO_S]);
    source2 = strcat(male_female2,'_',int2str(source));
    
    for j=1:num_jumps
        
        path = strcat('C:\project\TIMIT CD\TIMIT\test\',male_female1,'\',source1);
        f=dir([path '\*.wav']);
        folder=f.folder;
        num_file = randi(10);
        file=f(num_file).name;
        source11 = strcat(folder,'\',file);
        speech_11_temp = audioread(source11);
        speech_1=speech_11_temp(:,1);
       
         
        path = strcat('C:\project\TIMIT CD\TIMIT\test\',male_female2,'\',source2);
        f=dir([path '\*.wav']);
        folder=f.folder;
        num_file = randi(10);
        file=f(num_file).name;
        source11 = strcat(folder,'\',file);
        speech_11_temp = audioread(source11);
        speech_2=speech_11_temp(:,1);
            
        h_first = rir_generator(c_k, fs, r, s_first(:,j)', L, beta, n, mtype, order, dim, orientation, hp_filter);    
        Receivers_first = conv2(speech_1',h_first);
        label_first_temp = ones(1,length(Receivers_first))*label_first(j);
        
        rand_padding = randi(5);
        pad_zeros_for_noise=zeros(M,fs/2*rand_padding)';
        pad_zeros_for_location=zeros(1,fs/2*rand_padding)';
        
        if j==1
            label_first_total=[pad_zeros_for_location ; pad_zeros_for_location ; label_first_temp'];
            Receivers_first_total=[pad_zeros_for_noise ; pad_zeros_for_noise ; Receivers_first'];
        else
            label_first_total=[label_first_total ; pad_zeros_for_location ; label_first_temp'];
            Receivers_first_total=[Receivers_first_total ; pad_zeros_for_noise ;  Receivers_first'];
        end 
        
        rand_padding = randi(5);
        pad_zeros_for_noise=zeros(M,fs/2*rand_padding)';
        pad_zeros_for_location=zeros(1,fs/2*rand_padding)';
        
        h_second = rir_generator(c_k, fs, r, s_second(:,j)', L, beta, n, mtype, order, dim, orientation, hp_filter);
        Receivers_second = conv2(speech_2',h_second);
        label_second_temp = ones(1,length(Receivers_second))*label_second(j);
        if j==1
            label_second_total=[pad_zeros_for_location ; pad_zeros_for_location ; label_second_temp'];
            Receivers_second_total=[pad_zeros_for_noise ; pad_zeros_for_noise ;  Receivers_second'];
        else
            label_second_total=[label_second_total ; pad_zeros_for_location ; label_second_temp'];
            Receivers_second_total=[Receivers_second_total ; pad_zeros_for_noise ; Receivers_second'];
        end
    end
    
    maxlen = max(length(Receivers_first_total), length(Receivers_second_total));
    Receivers_first_total(end+1:maxlen,:) = 0;
    Receivers_second_total(end+1:maxlen,:) = 0;
    label_first_total(end+1:maxlen,:) = 0;
    label_second_total(end+1:maxlen,:) = 0;
    
    
    
    noise_list = importdata('C:\\project\\noise\\noise_list.txt');
    noise_file_name = cell2mat(noise_list(randi(22),1));
    noise_file=sprintf('C:\\project\\noise\\pointsource_noises\\%s.wav',noise_file_name);

    [noise_temp,fs_noise] = audioread(noise_file);
    if length(noise_temp)<length(Receivers_second_total)
        repeat_times = ceil(length(Receivers_second_total)/length(noise_temp));
        noise_temp=repelem(noise_temp,repeat_times);
    end
    noise_temp=noise_temp(1:length(Receivers_second_total)-n+1)';
        
%% create input

    Receivers_noise = conv2(noise_temp,h_noise);

    Receivers_first_total = Receivers_first_total/mean(std(Receivers_first_total));
    Receivers_second_total = Receivers_second_total/mean(std(Receivers_second_total));
    Receivers_noise = Receivers_noise/mean(std(Receivers_noise'));
    
    % calc snr
    
    SNR_direction = 15;
    SNR_mic = 30;
    receivers=Receivers_first_total'+Receivers_second_total';
    
    M=size(receivers,1);
    length_receives=size(receivers,2);
    %clac An
    
    A_x=mean(std(receivers'));
    std_n=mean(std(Receivers_noise'));
    A_n_difuse = A_x/(10^(SNR_difuse/20));
    A_n_diraction = A_x/(10^(SNR_direction/20));
    A_n_mic = A_x/(10^(SNR_mic/20));
    
    %create mic moise
    mic_noise = A_n_mic*randn(M,length_receives);
    
    % create difuse noise
    difuse_noise = fun_create_deffuse_noise();
    difuse_noise = difuse_noise/mean(std(difuse_noise));
    
    if length(difuse_noise)<length(receivers)
        repeat_times = ceil(length(receivers)/length(difuse_noise));
        difuse_noise=repmat(difuse_noise,repeat_times);
    end
    difuse_noise=difuse_noise(1:length(receivers),1:M);
    
    
    SNR_calc = snr(squeeze(receivers(1,:)),squeeze(A_n_difuse*difuse_noise(:,1)).');
    
    noise_total = (mic_noise+A_n_difuse*difuse_noise.'+A_n_diraction*Receivers_noise).';
    receivers=(receivers+mic_noise+A_n_difuse*difuse_noise.'+A_n_diraction*Receivers_noise).'; 
    
    noise_total = noise_total/max(max(abs(noise_total)));
    receivers=receivers/max(max(abs(receivers)));
    Receivers_first_total=Receivers_first_total/max(max(abs(Receivers_first_total)));
    Receivers_second_total=Receivers_second_total/max(max(abs(Receivers_second_total)));

    data_first=sprintf('C:\\project\\val\\first_%d.wav',i);
    data_second=sprintf('C:\\project\\val\\second_%d.wav',i);
    data_noise=sprintf('C:\\project\\val\\noise_%d.wav',i);
    data_together=sprintf('C:\\project\\val\\together_%d.wav',i);
    audiowrite(data_first,Receivers_first_total,fs)
    audiowrite(data_second,Receivers_second_total,fs)
    %audiowrite(data_noise,noise_total,fs)
    audiowrite(data_together,receivers,fs)
    
    
    vad_first_speaker = create_vad_dynamic(label_first_total,hop, nfft);
    vad_second_speaker = create_vad_dynamic(label_second_total,hop, nfft);
    label_first_location_file=sprintf('C:\\project\\val\\label_location_first_%d.mat',i);
    label_second_location_file=sprintf('C:\\project\\val\\label_location_second_%d.mat',i);
    save(label_first_location_file,'vad_first_speaker')
    save(label_second_location_file,'vad_second_speaker')
end

figure;
plot(Receivers_first_total(:,1))
hold on
plot(Receivers_second_total(:,1))

figure;
plot(vad_first_speaker)
hold on
plot(vad_second_speaker)

figure;
plot(receivers(:,1))
audiowrite('receivers.wav',receivers(:,1),fs)

