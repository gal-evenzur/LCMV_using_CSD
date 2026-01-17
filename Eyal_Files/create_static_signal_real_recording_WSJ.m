clear all
close all
load handel.mat
addpath('C:\project\fillline.m');
%% variens

clear all
%this is the radios of the table.
M = 4;
fs = 16000;
close all
SNR=20;
SNR_direction=25;
beta=0.35;
SIR = 5;
SNR_mic = 30;
%% load signals   
path = 'C:\project\static_signals\real_recording';
f1 = dir(path);

direction_noise = audioread('C:\project\noise\Factory_record.wav');
noise = audioread('C:\project\noise\bubble.wav');

direction_noise(:,1) = -direction_noise(:,1);
direction_noise(:,4) = -direction_noise(:,4);


noise(:,1) = -noise(:,1);
noise(:,4) = -noise(:,4);



speech_female_40 = audioread(strcat(path,'\',f1(4).name));
speech_female_40(:,1) = -speech_female_40(:,1);
speech_female_40(:,4) = -speech_female_40(:,4);

speech_male_60 = audioread(strcat(path,'\',f1(5).name));
speech_male_60(:,1) = -speech_male_60(:,1);
speech_male_60(:,4) = -speech_male_60(:,4);

speech_male_90 = audioread(strcat(path,'\',f1(6).name));
speech_male_90(:,1) = -speech_male_90(:,1);
speech_male_90(:,4) = -speech_male_90(:,4);

speech_male_140 = audioread(strcat(path,'\',f1(7).name));
speech_male_140(:,1) = -speech_male_140(:,1);
speech_male_140(:,4) = -speech_male_140(:,4);

speech_female_20 = audioread(strcat(path,'\',f1(8).name));
speech_female_20(:,1) = -speech_female_20(:,1);
speech_female_20(:,4) = -speech_female_20(:,4);

speech_female_40_1 = speech_female_40(53000:275000,:);
speech_female_40_2 = speech_female_40(320000:660000,:);

speech_male_60_1 = speech_male_60(30000:470000,:);
speech_male_60_2 = speech_male_60(540000:850000,:);

speech_male_90_1 = speech_male_90(28000:220000,:);
speech_male_90_2 = speech_male_90(270000:430000,:);

speech_male_140_1 = speech_male_140(20000:470000,:);
speech_male_140_2 = speech_male_140(540000:780000,:);

speech_female_20_1 = speech_female_20(35000:450000,:);
speech_female_20_2 = speech_female_20(560000:end,:);

%%  resample

speech_female_20_1 = downsample(speech_female_20_1,3);
speech_female_20_2 = downsample(speech_female_20_2,3);

speech_female_40_1 = downsample(speech_female_40_1,3);
speech_female_40_2 = downsample(speech_female_40_2,3);

speech_male_60_1 = downsample(speech_male_60_1,3);
speech_male_60_2 = downsample(speech_male_60_2,3);

speech_male_90_1 = downsample(speech_male_90_1,3);
speech_male_90_2 = downsample(speech_male_90_2,3);

speech_male_140_1 = downsample(speech_male_140_1,3);
speech_male_140_2 = downsample(speech_male_140_2,3);

noise = downsample(noise,3);
direction_noise = downsample(direction_noise,3);

pad_zeros1=zeros(4,fs)';


%% create first signal
in_20_140 = [pad_zeros1 ; speech_female_20_1 ; pad_zeros1 ; zeros(M,length(speech_male_140_1))' ; pad_zeros1 ; speech_female_20_2];
in_140_20 = [pad_zeros1 ; zeros(4,length(speech_female_20_1))' ; pad_zeros1 ; speech_male_140_1 ; pad_zeros1 ; speech_male_140_2];

maxlen = max(length(in_20_140), length(in_140_20));
in_20_140(end+1:maxlen,:) = 0;
in_140_20(end+1:maxlen,:) = 0;

receiver_first_20_140 = in_20_140.';
receiver_second_140_20 = in_140_20.';

receivers_20_140 = receiver_first_20_140+receiver_second_140_20;

%% create second signal

in_40_60 = [pad_zeros1 ; speech_female_40_1 ; pad_zeros1 ; zeros(M,length(speech_male_60_1))' ; pad_zeros1 ; speech_female_40_2];
in_60_40 = [pad_zeros1 ; zeros(4,length(speech_female_40_1))' ; pad_zeros1 ; speech_male_60_1 ; pad_zeros1 ; speech_male_60_2];

maxlen = max(length(in_40_60), length(in_60_40));
in_40_60(end+1:maxlen,:) = 0;
in_60_40(end+1:maxlen,:) = 0;

receiver_first_40_60 = in_40_60.';
receiver_second_60_40 = in_60_40.';

receivers_40_60 = receiver_first_40_60+receiver_second_60_40;


%% create third signal

in_20_90 = [pad_zeros1 ; speech_female_20_1 ; pad_zeros1 ; zeros(M,length(speech_male_90_1))' ; pad_zeros1 ; speech_female_20_2];
in_90_20 = [pad_zeros1 ; zeros(4,length(speech_female_20_1))' ; pad_zeros1 ; speech_male_90_1 ; pad_zeros1 ; speech_male_90_2];

maxlen = max(length(in_20_90), length(in_90_20));
in_20_90(end+1:maxlen,:) = 0;
in_90_20(end+1:maxlen,:) = 0;

receiver_first_20_90 = in_20_90.';
receiver_second_90_20 = in_90_20.';

receivers_20_90 = receiver_first_20_90+receiver_second_90_20;

%% calc speakers STD and norm first

receiver_first_temp = receiver_first_20_140;
receiver_second_temp = receiver_second_140_20;
receiver_first_temp(receiver_first_temp==0)=NaN;
receiver_second_temp(receiver_second_temp==0)=NaN;
A_x_first = mean(nanstd(receiver_first_temp.',[],1));   %column by column std deviation
A_x_second = mean(nanstd(receiver_second_temp.',[],1));
receiver_first_20_140 = receiver_first_20_140/A_x_first/10;
receiver_second_140_20 = receiver_second_140_20/A_x_second/10;   

%% calc speakers STD and norm first

receiver_first_temp = receiver_first_40_60;
receiver_second_temp = receiver_second_60_40;
receiver_first_temp(receiver_first_temp==0)=NaN;
receiver_second_temp(receiver_second_temp==0)=NaN;
A_x_first = mean(nanstd(receiver_first_temp.',[],1));   %column by column std deviation
A_x_second = mean(nanstd(receiver_second_temp.',[],1));
receiver_first_40_60 = receiver_first_40_60/A_x_first/10;
receiver_second_60_40 = receiver_second_60_40/A_x_second/10;   

%% calc speakers STD and norm first

receiver_first_temp = receiver_first_20_90;
receiver_second_temp = receiver_second_90_20;
receiver_first_temp(receiver_first_temp==0)=NaN;
receiver_second_temp(receiver_second_temp==0)=NaN;
A_x_first = mean(nanstd(receiver_first_temp.',[],1));   %column by column std deviation
A_x_second = mean(nanstd(receiver_second_temp.',[],1));
receiver_first_20_90 = receiver_first_20_90/A_x_first/10;
receiver_second_90_20 = receiver_second_90_20/A_x_second/10;   

%% create noise

noise = noise/mean(std(noise'));
% noise = noise';


% difuse_noise = fun_create_deffuse_noise();
% noise = difuse_noise/mean(std(difuse_noise));

direction_noise = direction_noise/mean(std(direction_noise'));
direction_noise = direction_noise';

%% create input first

if length(noise)<length(receivers_20_140)
    repeat_times = ceil(length(receivers_20_140)/length(noise));
    noise=repmat(noise,repeat_times);
end
noise1=noise(1:length(receivers_20_140),1:M).';

% noise1 = noise(:,1:length(receivers_20_140));
direction_noise1 = direction_noise(:,1:length(receivers_20_140));
A_x=mean(std(receivers_20_140'));
A_n_noise = A_x/(10^(SNR/20));
A_n_direction_noise = A_x/(10^(SNR_direction/20));

A_n_mic = A_x/(10^(SNR_mic/20));
mic_noise = randn(M,length(receivers_20_140));
%create mic moise 
% create difuse noise

receivers=(receivers_20_140+A_n_mic*mic_noise+A_n_direction_noise*direction_noise1+A_n_noise*noise1);

path = 'C:\\project\\static_signals\\two_speakers_real_recording\\';
data_together=strcat(path,'dynamic_signal_1.wav');
data_first=strcat(path,'dynamic_signal_first_1.wav');
data_second=strcat(path,'dynamic_signal_second_1.wav');
data_noise=strcat(path,'dynamic_signal_noise_1.wav');

audiowrite(data_together,receivers.',fs) 
audiowrite(data_first,receiver_first_20_140.',fs) 
audiowrite(data_second,receiver_second_140_20.',fs) 
audiowrite(data_noise,+A_n_mic*mic_noise'+A_n_direction_noise*direction_noise1'+A_n_noise*noise1',fs)

location_first = 2;
angle_location_file=strcat(path,'angle_location_first_1.mat');
save(angle_location_file,'location_first')
location_second = 15;
angle_location_file=strcat(path,'angle_location_second_1.mat');
save(angle_location_file,'location_second')


%% create input second

if length(noise)<length(receivers_40_60)
    repeat_times = ceil(length(receivers_40_60)/length(noise));
    noise=repmat(noise,repeat_times);
end
noise2=noise(1:length(receivers_40_60),1:M).';

% noise2 = noise(:,1:length(receivers_40_60));
direction_noise2 = direction_noise(:,1:length(receivers_40_60));
A_x=mean(std(receivers_40_60'));
A_n_noise = A_x/(10^(SNR/20));
A_n_mic = A_x/(10^(SNR_mic/20));
mic_noise = A_n_mic*randn(M,length(receivers_40_60));

receivers=(receivers_40_60+A_n_mic*mic_noise+A_n_direction_noise*direction_noise2+A_n_noise*noise2);

path = 'C:\\project\\static_signals\\two_speakers_real_recording\\';
data_together=strcat(path,'dynamic_signal_2.wav');
data_first=strcat(path,'dynamic_signal_first_2.wav');
data_second=strcat(path,'dynamic_signal_second_2.wav');
data_noise=strcat(path,'dynamic_signal_noise_2.wav');

audiowrite(data_together,receivers.',fs) 
audiowrite(data_first,receiver_first_40_60.',fs) 
audiowrite(data_second,receiver_second_60_40.',fs) 
audiowrite(data_noise,A_n_mic*mic_noise'+A_n_direction_noise*direction_noise2'+A_n_noise*noise2',fs)

location_first = 4;
angle_location_file=strcat(path,'angle_location_first_2.mat');
save(angle_location_file,'location_first')
location_second = 7;
angle_location_file=strcat(path,'angle_location_second_2.mat');
save(angle_location_file,'location_second')

%% create input third
if length(noise)<length(receivers_20_90)
    repeat_times = ceil(length(receivers_20_90)/length(noise));
    noise=repmat(noise,repeat_times);
end
noise3=noise(1:length(receivers_20_90),1:M).';

% noise3 = noise(:,1:length(receivers_20_90));
direction_noise3 = direction_noise(:,1:length(receivers_20_90));
A_x=mean(std(receivers_20_90'));
A_n_noise = A_x/(10^(SNR/20));
A_n_mic = A_x/(10^(SNR_mic/20));
mic_noise = randn(M,length(receivers_20_90));

receivers=(receivers_20_90+A_n_mic*mic_noise+A_n_direction_noise*direction_noise3+A_n_noise*noise3);

path = 'C:\\project\\static_signals\\two_speakers_real_recording\\';
data_together=strcat(path,'dynamic_signal_3.wav');
data_first=strcat(path,'dynamic_signal_first_3.wav');
data_second=strcat(path,'dynamic_signal_second_3.wav');
data_noise=strcat(path,'dynamic_signal_noise_3.wav');

audiowrite(data_together,receivers.',fs) 
audiowrite(data_first,receiver_first_20_90.',fs) 
audiowrite(data_second,receiver_second_90_20.',fs) 
audiowrite(data_noise,A_n_mic*mic_noise'+A_n_direction_noise*direction_noise3'+A_n_noise*noise3',fs)

location_first = 2;
angle_location_file=strcat(path,'angle_location_first_3.mat');
save(angle_location_file,'location_first')
location_second = 9;
angle_location_file=strcat(path,'angle_location_second_3.mat');
save(angle_location_file,'location_second')

