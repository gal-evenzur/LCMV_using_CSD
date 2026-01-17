clear all
close all
load handel.mat
addpath('C:\project\fillline.m');
%% variens

clear all
%this is the radios of the table.

c_k = 340;                                        % Sound velocity (m/s)
c = 340; 
fs = 16000;
n = 4096;                                         % Number of samples
mtype = 'omnidirectional';                        % Type of microphone
order = -1;                                       % -1 equals maximum reflection order!
dim = 3;                                          % Room dimension
orientation = 0;                                  % Microphone orientation (rad)
hp_filter = 1;                                    % Disable high-pass filter
% define analysis parameters
R=1.3;
R_small = 1.2;
noise_R=0.2;
hop=1024;
M = 4;
high=1;
angle=360;
distance_from_woll=0.5;
radius_mics = 0.1;

%% rir generation & covariens sounds 
for i=1:10
    close all
    % define room dimension
    L1_temp=randi([1,20]);
    L1=4+0.1*L1_temp;
    L2_temp=randi([1,20]);
    L2=4+0.1*L2_temp;
    L = [L1 L2 3];
    room_x = L(1);
    room_y = L(2);
    SNR_difuse=20;
    beta=0.3;
    SIR = 0;
    %% create circle & line & mic location
    
    distance_total=R+distance_from_woll+noise_R;
    end_point_x=room_x-(R+distance_from_woll+noise_R);
    end_point_y=room_y-(R+distance_from_woll+noise_R);
    Radius_X = (end_point_x-distance_total).*rand + distance_total;
    Radius_Y = (end_point_y-distance_total).*rand + distance_total;

    R_angle=randi([1,angle]); % take  rand 180 degrees of circle to create rand orientation of Microphone array
    t = linspace(-pi,2*pi,angle+angle/2); % 3 times 180 to create option to rand whole circ
    t=t(R_angle:R_angle+angle/2-1);
    x = R*sin(t)+Radius_X;
    y = R*cos(t)+Radius_Y;
    x_small = R_small*sin(t)+Radius_X;
    y_small = R_small*cos(t)+Radius_Y;
    
    z=0*t+high;
    circ_mics_x = radius_mics*sin(t)+Radius_X;
    circ_mics_y = radius_mics*cos(t)+Radius_Y;
    [line_x,line_y]=fillline([x(1) y(1)], [x(angle/2) y(angle/2)],R*2*100);
    
    r= [circ_mics_x(1) circ_mics_y(1) high; circ_mics_x(55) circ_mics_y(55) high;...
        circ_mics_x(120) circ_mics_y(120) high; circ_mics_x(180) circ_mics_y(180) high];
  
%% create speakers clean

    male_female_check=randi([1,2]);
    
    if male_female_check==1
       male_female1='male'; 
    else
       male_female1='female'; 
    end
    
   
    path1 = strcat('C:\project\WSJ\test\',male_female1);
    list_path1 = dir(path1);
    ridx = randi([3,numel(list_path1)]);
    speaker1_name = list_path1(ridx).name;
    speaker_path1 = strcat(path1,'\',speaker1_name);
    
    male_female_check=randi([1,2]);
    
    if male_female_check==1
       male_female2='male'; 
    else
       male_female2='female'; 
    end
    
    path2 = strcat('C:\project\WSJ\test\',male_female2);
    list_path2 = dir(path2);
    ridx = randi([3,numel(list_path2)]);
    speaker2_name = list_path2(ridx).name;
    speaker_path2 = strcat(path2,'\',speaker2_name);

    f1=dir([speaker_path1 '\*.wav']);
    f2=dir([speaker_path2 '\*.wav']);
    
    len_sig = 0;
    while len_sig<fs*4
        ridx = randi([3,numel(f1)]);
        speaker1_name = f1(ridx).name;
        source11 = strcat(speaker_path1,'\',speaker1_name);
        speech_11_temp = audioread(source11);
        speech_11=speech_11_temp(:,1);
        len_sig = length(speech_11);
    end
    
    len_sig = 0;
    while len_sig<fs*4
        ridx = randi([3,numel(f1)]);
        speaker1_name = f1(ridx).name;
        source12 = strcat(speaker_path1,'\',speaker1_name);
        speech_12_temp = audioread(source12);
        speech_12=speech_12_temp(:,1);
        len_sig = length(speech_12);
    end
    
    len_sig = 0;
    while len_sig<fs*4
        ridx = randi([3,numel(f2)]);
        speaker2_name = f2(ridx).name;
        source21 = strcat(speaker_path2,'\',speaker2_name);
        speech_21_temp = audioread(source21);
        speech_21=speech_21_temp(:,1);
        len_sig = length(speech_21);
    end
    
    len_sig = 0;
    while len_sig<fs*4
        ridx = randi([3,numel(f2)]);
        speaker2_name = f2(ridx).name;
        source22 = strcat(speaker_path2,'\',speaker2_name);
        speech_22_temp = audioread(source22);
        speech_22=speech_22_temp(:,1);
        len_sig = length(speech_22);
    end
    
    pad_zeros1=zeros(1,fs)';
    pad_zeros2=zeros(1,fs)';
    
    firtst_alone = speech_11;
    firtst_together = speech_12;
    second_alone = speech_21;
    second_together = speech_22;
    
    in1 = [pad_zeros2 ; firtst_alone ; pad_zeros1 ; zeros(1,length(second_alone))' ; pad_zeros1 ; firtst_together];
    vad1 = [pad_zeros2 ; ones(1,length(firtst_alone))' ; pad_zeros1 ; zeros(1,length(second_alone))' ; pad_zeros1 ; ones(1,length(firtst_together))'];
    
    start_move1 = length(pad_zeros2);
    finish_move1 = start_move1+length(firtst_alone)-2.5*fs;
    
    in2 = [pad_zeros2 ; zeros(1,length(firtst_alone))' ; pad_zeros1 ; second_alone ; pad_zeros1 ; second_together];
    vad2 = [pad_zeros2 ; zeros(1,length(firtst_alone))' ; pad_zeros1 ; ones(1,length(second_alone))' ; pad_zeros1 ; ones(1,length(second_together))'];

    start_move2 = length(pad_zeros2)+length(firtst_alone)+length(pad_zeros1);
    finish_move2 = start_move2+length(second_alone)-2*fs;
    
    maxlen = max(length(in1), length(in2));
    in1(end+1:maxlen,:) = 0;
    in2(end+1:maxlen,:) = 0;
    vad1(end+1:maxlen,:) = 0;
    vad2(end+1:maxlen,:) = 0;
        
    in1 = in1.';
    in2 = in2.';
    
    vad1 = vad1.';
    vad2 = vad2.';
    
    vad = vad1+vad2;
    
%% signal generator     
    
    len = length(in1);
    sp_path_first = zeros(len,3);
    sp_path_small_first = zeros(len,3);
    rp_path = zeros(len,3,M);
    location_first = zeros(1,len);
    ii_index = randi([2,40]);
    foward_index = 1;
    for ii = 1:hop:len
        
        if ii>start_move1 && ii<finish_move1
            ii_index = ii_index+foward_index;
        end
        
        if mod(ii_index,140)==0 || ii_index==1
            foward_index=foward_index*-1;
        end
        
        % Store source path
        sp_path_first(ii:1:min(ii+hop-1,len),1) = x(ii_index);
        sp_path_first(ii:1:min(ii+hop-1,len),2) = y(ii_index);
        sp_path_first(ii:1:min(ii+hop-1,len),3) = high;
        sp_path_small_first(ii:1:min(ii+hop-1,len),1) = x_small(ii_index);
        sp_path_small_first(ii:1:min(ii+hop-1,len),2) = y_small(ii_index);       
        sp_path_small_first(ii:1:min(ii+hop-1,len),3) = high;
        location_first(ii:1:min(ii+hop-1,len)) = ii_index;
        % Stationary receiver positions
        for mm=1:M
            rp_path(ii:1:min(ii+hop-1,len),1,mm) = r(mm,1);
            rp_path(ii:1:min(ii+hop-1,len),2,mm) = r(mm,2);
            rp_path(ii:1:min(ii+hop-1,len),3,mm) = high;    
        end
    end
    
    % signal generator
    
    [receiver_first,~] = signal_generator(in1,c,fs,rp_path,sp_path_first,L,beta,n,'o',order);
    

    
    len = length(in2);
    sp_path_second = zeros(len,3);
    sp_path_small_second = zeros(len,3);
    location_second = zeros(1,len);
    ii_index = 179;
    foward_index = -1;
    for ii = 1:hop:len
        
        if mod(ii_index,18)==0 
            foward_index=foward_index*-1;
        end
        
        if ii>start_move2 && ii<finish_move2
            ii_index = ii_index+foward_index;
        end
        
        % Store source path
        sp_path_second(ii:1:min(ii+hop-1,len),1) = x(ii_index);
        sp_path_second(ii:1:min(ii+hop-1,len),2) = y(ii_index);
        sp_path_second(ii:1:min(ii+hop-1,len),3) = high;
        sp_path_small_second(ii:1:min(ii+hop-1,len),1) = x_small(ii_index);
        sp_path_small_second(ii:1:min(ii+hop-1,len),2) = y_small(ii_index);       
        sp_path_small_second(ii:1:min(ii+hop-1,len),3) = high;
        location_second(ii:1:min(ii+hop-1,len)) = ii_index;
        % Stationary receiver positions
    end
    
    % signal generator
    
    [receiver_second,beta_hat] = signal_generator(in2,c,fs,rp_path,sp_path_second,L,beta,n,'o',order);
    
%% calc speakers STD and norm  
    
    receiver_first_temp = receiver_first;
    receiver_second_temp = receiver_second;
    receiver_first_temp(receiver_first_temp==0)=NaN;
    receiver_second_temp(receiver_second_temp==0)=NaN;
    A_x_first = mean(nanstd(receiver_first_temp.',[],1));   %column by column std deviation
    A_x_second = mean(nanstd(receiver_second_temp.',[],1));
    receiver_first = receiver_first/A_x_first/10;
    receiver_second = receiver_second/A_x_second/10;   
    
%     A_x_before=(mean(std(receiver_first'))+mean(std(receiver_second')))/2;
    A_x_before = 1/10;
%% create noise

    middle = [Radius_X Radius_Y high];
    s_noise = [Radius_X Radius_Y high];
    d_noise = norm(s_noise-middle);
    while d_noise<2
        x_noise=distance_from_woll+0.01*randi(100)*(room_x-2*distance_from_woll);
        y_noise=distance_from_woll+0.01*randi(100)*(room_y-2*distance_from_woll);
        s_noise = [x_noise y_noise high];
        d_noise = norm(s_noise-middle);
    end
    
    noise_list = importdata('C:\\project\\noise\\noise_list.txt');
    noise_file_name = cell2mat(noise_list(randi(22),1));
    noise_file=sprintf('C:\\project\\noise\\pointsource_noises\\%s.wav',noise_file_name);

    [noise_temp,fs_noise] = audioread(noise_file);
    if length(noise_temp)<length(receiver_second)
        repeat_times = ceil(length(receiver_second)/length(noise_temp));
        noise_temp=repelem(noise_temp,repeat_times);
    end
    noise_temp=noise_temp(1:length(receiver_second)-n+1)';
    
    h_noise = rir_generator(c_k, fs, r, s_noise, L, beta, n, mtype, order, dim, orientation, hp_filter);
    Receivers_noise = conv2(noise_temp,h_noise);    
    Receivers_noise = Receivers_noise/mean(std(Receivers_noise'));
    
%     Receivers_noise_new = noise_temp_new/std(noise_temp_new);
%     Receivers_noise_new4 = [Receivers_noise_new ; Receivers_noise_new ; Receivers_noise_new ; Receivers_noise_new];
%% create input

    % calc snr
    SNR_direction = 20;
    SNR_mic = 30;

    M=size(receiver_first,1);
    length_receives=size(receiver_first,2);
    % create difuse noise
    difuse_noise = fun_create_deffuse_noise();
    difuse_noise = difuse_noise/mean(std(difuse_noise));
    
    if length(difuse_noise)<length(receiver_second)
        repeat_times = ceil(length(receiver_second)/length(difuse_noise));
        difuse_noise=repmat(difuse_noise,repeat_times);
    end
    difuse_noise=difuse_noise(1:length(receiver_second),1:M);

    
    for j=10:2.5:20
        %clac An
        receivers = receiver_first+receiver_second;
        A_x=mean(std(receivers'));
        A_n_noise = A_x/(10^(j/20));
        A_n_diraction = A_x/(10^(SNR_direction/20));
        A_n_mic = A_x/(10^(SNR_mic/20));
        %create mic moise
        mic_noise = A_n_mic*randn(M,length_receives);    
        % create difuse noise

        receivers=(receivers+mic_noise+A_n_diraction*Receivers_noise+A_n_noise*difuse_noise.').'; 
        noise_only = (mic_noise+A_n_diraction*Receivers_noise+A_n_noise*difuse_noise.').';

        value = num2str(j);
        path = sprintf('C:\\project\\dynamic_signals\\two_speakers_WSJ\\SNR\\SNR_%s_T60_300_SIR_0\\',value);
        data_together=strcat(path,sprintf('dynamic_signal_%d.wav',i));
        data_first=strcat(path,sprintf('dynamic_signal_first_%d.wav',i));
        data_second=strcat(path,sprintf('dynamic_signal_second_%d.wav',i));
        data_noise=strcat(path,sprintf('dynamic_signal_noise_%d.wav',i));
        
        audiowrite(data_together,receivers,fs) 
        audiowrite(data_first,receiver_first.',fs) 
        audiowrite(data_second,receiver_second.',fs) 
        audiowrite(data_noise,noise_only,fs) 
        
        angle_location_file=strcat(path,sprintf('angle_location_first_%d.mat',i));
        save(angle_location_file,'location_first')
        angle_location_file=strcat(path,sprintf('angle_location_second_%d.mat',i));
        save(angle_location_file,'location_second')
        mic_array_file=strcat(path,sprintf('mic_array_%d.mat',i));
        save(mic_array_file,'r')
        
        location_file=strcat(path,sprintf('locations_first_%d.mat',i));
        save(location_file,'sp_path_first') 
        location_file=strcat(path,sprintf('locations_second_%d.mat',i));
        save(location_file,'sp_path_second') 
        location_small_file=strcat(path,sprintf('locations_small_first_%d.mat',i));
        save(location_small_file,'sp_path_small_first')   
        location_small_file=strcat(path,sprintf('locations_small_second_%d.mat',i));
        save(location_small_file,'sp_path_small_second')   
    end
    
    for j=0:5:15
        
        Ax_SIR = 1/(10^(j/20));
        receivers = Ax_SIR*receiver_first+receiver_second;
        receiver_first_sir = Ax_SIR*receiver_first;
        %clac An
        A_x=mean(std(receivers'));
        A_n_difuse = A_x/(10^(SNR_difuse/20));
        A_n_diraction = A_x/(10^(SNR_direction/20));
        A_n_mic = A_x/(10^(SNR_mic/20));

        %create mic moise
        mic_noise = A_n_mic*randn(M,length_receives);    
        % create difuse noise
        receivers=(receivers+mic_noise+A_n_diraction*Receivers_noise+A_n_difuse*difuse_noise.').'; 
        noise_only = (mic_noise+A_n_diraction*Receivers_noise+A_n_difuse*difuse_noise.').'; 

        value = num2str(j);
        path = sprintf('C:\\project\\dynamic_signals\\two_speakers_WSJ\\SIR\\SNR_20_T60_300_SIR_%s\\',value);
        data_together=strcat(path,sprintf('dynamic_signal_%d.wav',i));
        data_first=strcat(path,sprintf('dynamic_signal_first_%d.wav',i));
        data_second=strcat(path,sprintf('dynamic_signal_second_%d.wav',i));
        data_noise=strcat(path,sprintf('dynamic_signal_noise_%d.wav',i));
        
        audiowrite(data_together,receivers,fs) 
        audiowrite(data_first,receiver_first_sir.',fs) 
        audiowrite(data_second,receiver_second.',fs) 
        audiowrite(data_noise,noise_only,fs) 
       
        angle_location_file=strcat(path,sprintf('angle_location_first_%d.mat',i));
        save(angle_location_file,'location_first')
        angle_location_file=strcat(path,sprintf('angle_location_second_%d.mat',i));
        save(angle_location_file,'location_second')
        mic_array_file=strcat(path,sprintf('mic_array_%d.mat',i));
        save(mic_array_file,'r')
        
        location_file=strcat(path,sprintf('locations_first_%d.mat',i));
        save(location_file,'sp_path_first') 
        location_file=strcat(path,sprintf('locations_second_%d.mat',i));
        save(location_file,'sp_path_second') 
        location_small_file=strcat(path,sprintf('locations_small_first_%d.mat',i));
        save(location_small_file,'sp_path_small_first')   
        location_small_file=strcat(path,sprintf('locations_small_second_%d.mat',i));
        save(location_small_file,'sp_path_small_second')   
    end
end

%% Plot source path and signals
figure(1);
plot3(r(1,1),r(1,2),r(1,3),'x');
hold on;
for mm = 2:M
    plot3(r(mm,1),r(mm,2),r(mm,3),'x');
end

plot3(sp_path_first(:,1),sp_path_first(:,2),sp_path_first(:,3),'b.');
hold on
plot3(sp_path_small_first(:,1),sp_path_small_first(:,2),sp_path_small_first(:,3),'b.');
hold on
plot3(sp_path_second(:,1),sp_path_second(:,2),sp_path_second(:,3),'r.');
hold on
plot3(sp_path_small_second(:,1),sp_path_small_second(:,2),sp_path_small_second(:,3),'r.');

axis([0 L(1) 0 L(2) 0 L(3)]);
grid on;
box on;
axis square;
hold off;

figure(2)
t = 0:1/fs:(length(in1)-1)/fs;
subplot(211); plot(t,in1); title('in(n)');xlabel('Time [Seconds]');ylabel('Amplitude');
subplot(212); plot(t,receivers'); title('out(n)');xlabel('Time [Seconds]');ylabel('Amplitude');

figure;
plot(receiver_first_sir(1,:))
hold on
plot(receiver_second(1,:))
