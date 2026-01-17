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
SNR_direction=20;



%% load signals   
path = 'C:\project\static_signals\real_recording2';
f1 = dir(path);

direction_noise = audioread(strcat(path,'\noise.wav'));
noise = audioread(strcat(path,'\babble.wav'));

direction_noise(:,1) = -direction_noise(:,1);
direction_noise(:,4) = -direction_noise(:,4);


noise(:,1) = -noise(:,1);
noise(:,4) = -noise(:,4);

first1 = audioread(strcat(path,'\first_1.wav'));
first1(:,1) = -first1(:,1);
first1(:,4) = -first1(:,4);

first2 = audioread(strcat(path,'\first_2.wav'));
first2(:,1) = -first2(:,1);
first2(:,4) = -first2(:,4);

first3 = audioread(strcat(path,'\first_3.wav'));
first3(:,1) = -first3(:,1);
first3(:,4) = -first3(:,4);

second1 = audioread(strcat(path,'\second_1.wav'));
second1(:,1) = -second1(:,1);
second1(:,4) = -second1(:,4);

second2 = audioread(strcat(path,'\second_2.wav'));
second2(:,1) = -second2(:,1);
second2(:,4) = -second2(:,4);

second3 = audioread(strcat(path,'\second_3.wav'));
second3(:,1) = -second3(:,1);
second3(:,4) = -second3(:,4);

%%  resample

first1 = downsample(first1,3);
first2 = downsample(first2,3);
first3 = downsample(first3,3);

second1 = downsample(second1,3);
second2 = downsample(second2,3);
second3 = downsample(second3,3);

direction_noise = downsample(direction_noise,3);
noise = downsample(noise,3);


%% create first signal

maxlen = max(length(first1), length(second1));
first1(end+1:maxlen,:) = 0;
second1(end+1:maxlen,:) = 0;

maxlen = max(length(first2), length(second2));
first2(end+1:maxlen,:) = 0;
second2(end+1:maxlen,:) = 0;

maxlen = max(length(first3), length(second3));
first3(end+1:maxlen,:) = 0;
second3(end+1:maxlen,:) = 0;

first1 = first1.';
second1 = second1.';


first2 = first2.';
second2 = second2.';


first3 = first3.';
second3 = second3.';


%% calc speakers STD and norm first

receiver_first_temp = first1;
receiver_second_temp = second1;
receiver_first_temp(receiver_first_temp==0)=NaN;
receiver_second_temp(receiver_second_temp==0)=NaN;
A_x_first = mean(nanstd(receiver_first_temp.',[],1));   %column by column std deviation
A_x_second = mean(nanstd(receiver_second_temp.',[],1));
first1 = first1/A_x_first/10;
second1 = second1/A_x_second/10;   

%% calc speakers STD and norm first

receiver_first_temp = first2;
receiver_second_temp = second2;
receiver_first_temp(receiver_first_temp==0)=NaN;
receiver_second_temp(receiver_second_temp==0)=NaN;
A_x_first = mean(nanstd(receiver_first_temp.',[],1));   %column by column std deviation
A_x_second = mean(nanstd(receiver_second_temp.',[],1));
first2 = first2/A_x_first/10;
second2 = second2/A_x_second/10;    

%% calc speakers STD and norm first

receiver_first_temp = first3;
receiver_second_temp = second3;
receiver_first_temp(receiver_first_temp==0)=NaN;
receiver_second_temp(receiver_second_temp==0)=NaN;
A_x_first = mean(nanstd(receiver_first_temp.',[],1));   %column by column std deviation
A_x_second = mean(nanstd(receiver_second_temp.',[],1));
first3 = first3/A_x_first/10;
second3 = second3/A_x_second/10;     

%% create noise

noise = noise/mean(std(noise'));
direction_noise = direction_noise/mean(std(direction_noise'));

%% create input first

if length(noise)<length(first1)
    repeat_times = ceil(length(first1)/length(noise));
    noise=repmat(noise,repeat_times);
end
noise1=noise(1:length(first1),1:M).';

if length(direction_noise)<length(first1)
    repeat_times = ceil(length(first1)/length(direction_noise));
    direction_noise=repmat(direction_noise,repeat_times);
end
direction_noise1=direction_noise(1:length(first1),1:M).';

A_x=mean(std(first1'));
A_n_noise = A_x/(10^(SNR/20));
A_n_direction_noise = A_x/(10^(SNR_direction/20));

%create mic moise 
% create difuse noise

receivers=(first1+second1+A_n_direction_noise*direction_noise1+A_n_noise*noise1);

path = 'C:\\project\\static_signals\\two_speakers_real_recording\\';
data_together=strcat(path,'dynamic_signal_1.wav');
data_first=strcat(path,'dynamic_signal_first_1.wav');
data_second=strcat(path,'dynamic_signal_second_1.wav');
data_noise=strcat(path,'dynamic_signal_noise_1.wav');

audiowrite(data_together,receivers.',fs) 
audiowrite(data_first,first1.',fs) 
audiowrite(data_second,second1.',fs) 
audiowrite(data_noise,A_n_direction_noise*direction_noise1'+A_n_noise*noise1',fs)

location_first = 13;
angle_location_file=strcat(path,'angle_location_first_1.mat');
save(angle_location_file,'location_first')
location_second = 6;
angle_location_file=strcat(path,'angle_location_second_1.mat');
save(angle_location_file,'location_second')


%% create input first

if length(noise)<length(first2)
    repeat_times = ceil(length(first2)/length(noise));
    noise=repmat(noise,repeat_times);
end
noise2=noise(1:length(first2),1:M).';

if length(direction_noise)<length(first2)
    repeat_times = ceil(length(first2)/length(direction_noise));
    direction_noise=repmat(direction_noise,repeat_times);
end

direction_noise2=direction_noise(1:length(first2),1:M).';


A_x=mean(std(first2'));
A_n_noise = A_x/(10^(SNR/20));
A_n_direction_noise = A_x/(10^(SNR_direction/20));

receivers=(first2+second2+A_n_direction_noise*direction_noise2+A_n_noise*noise2);

path = 'C:\\project\\static_signals\\two_speakers_real_recording\\';
data_together=strcat(path,'dynamic_signal_2.wav');
data_first=strcat(path,'dynamic_signal_first_2.wav');
data_second=strcat(path,'dynamic_signal_second_2.wav');
data_noise=strcat(path,'dynamic_signal_noise_2.wav');

audiowrite(data_together,receivers.',fs) 
audiowrite(data_first,first2.',fs) 
audiowrite(data_second,second2.',fs) 
audiowrite(data_noise,A_n_direction_noise*direction_noise2'+A_n_noise*noise2',fs)

location_first = 13;
angle_location_file=strcat(path,'angle_location_first_2.mat');
save(angle_location_file,'location_first')
location_second = 6;
angle_location_file=strcat(path,'angle_location_second_2.mat');
save(angle_location_file,'location_second')

%% create input first

if length(noise)<length(first3)
    repeat_times = ceil(length(first3)/length(noise));
    noise=repmat(noise,repeat_times);
end
noise3=noise(1:length(first3),1:M).';

if length(direction_noise)<length(first3)
    repeat_times = ceil(length(first3)/length(direction_noise));
    direction_noise=repmat(direction_noise,repeat_times);
end

direction_noise3=direction_noise(1:length(first3),1:M).';

A_x=mean(std(first3'));
A_n_noise = A_x/(10^(SNR/20));
A_n_direction_noise = A_x/(10^(SNR_direction/20));

%create mic moise 
% create difuse noise

receivers=(first3+second3+A_n_direction_noise*direction_noise3+A_n_noise*noise3);

path = 'C:\\project\\static_signals\\two_speakers_real_recording\\';
data_together=strcat(path,'dynamic_signal_3.wav');
data_first=strcat(path,'dynamic_signal_first_3.wav');
data_second=strcat(path,'dynamic_signal_second_3.wav');
data_noise=strcat(path,'dynamic_signal_noise_3.wav');

audiowrite(data_together,receivers.',fs) 
audiowrite(data_first,first3.',fs) 
audiowrite(data_second,second3.',fs) 
audiowrite(data_noise,A_n_direction_noise*direction_noise3'+A_n_noise*noise3',fs)

location_first = 13;
angle_location_file=strcat(path,'angle_location_first_3.mat');
save(angle_location_file,'location_first')
location_second = 6;
angle_location_file=strcat(path,'angle_location_second_3.mat');
save(angle_location_file,'location_second')

