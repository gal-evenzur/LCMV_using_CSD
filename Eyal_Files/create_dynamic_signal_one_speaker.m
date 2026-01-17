clear all
close all
load handel.mat
addpath('C:\project\fillline.m');
%% variens

clear all
%this is the radios of the table.

c = 340;                                        % Sound velocity (m/s)
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
R_small = 1.2;
pad=25000;
start=1;
max1=100;
noise_R=0.2;
flag=0;
flag2=0;
nfft=2048; 
hop=1024;
M = 7;
num_jumps=9;
high=1;
angle=360;
distance_from_woll=0.5;
radius_mics = 0.1;
%% rir generation & covariens sounds 
for p=1:3
    beta=0.45+0.05*(p-1);
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
        %SNR_difuse=10+randi([0,10]); 
        %beta_temp=randi([0,250]);
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

        r= [circ_mics_x(1) circ_mics_y(1) high; circ_mics_x(25) circ_mics_y(25) high; circ_mics_x(55) circ_mics_y(55) high; circ_mics_x(80) circ_mics_y(80) high;...
            circ_mics_x(120) circ_mics_y(120) high; circ_mics_x(150) circ_mics_y(150) high; circ_mics_x(180) circ_mics_y(180) high];

    %% load signal    
        male_female_check=randi([1,2]);

        if male_female_check==1
           male_female1='male'; 
           NO_S=112;
        else
           male_female1='female'; 
           NO_S=56;
        end

        source=randi([1,NO_S]);
        source1 = int2str(source);

        path = strcat('C:\project\TIMIT CD\TIMIT\test\',male_female1,'\',male_female1,'_',source1);
        f=dir([path '\*.wav']);
        folder=f.folder;

        for g=1:8
            num_file = randi([2,10]);
            file=f(num_file).name;
            source11 = strcat(folder,'\',file);
            speech_11_temp = audioread(source11);
            speech_1=speech_11_temp(:,1);
            if g==1
                in = speech_1;
            else
                in = [in ; speech_1];
            end
        end
        in = in.';
        in = in(1:hop*angle/2);

        len = length(in);
        sp_path = zeros(len,3);
        sp_path_small = zeros(len,3);
        rp_path = zeros(len,3,M);
        locations = zeros(1,len);
        ii_index = 0;
        for ii = 1:hop:len
            ii_index = ii_index+1;
            % Store source path
            sp_path(ii:1:min(ii+hop-1,len),1) = x(ii_index);
            sp_path(ii:1:min(ii+hop-1,len),2) = y(ii_index);
            sp_path(ii:1:min(ii+hop-1,len),3) = high;
            sp_path_small(ii:1:min(ii+hop-1,len),1) = x_small(ii_index);
            sp_path_small(ii:1:min(ii+hop-1,len),2) = y_small(ii_index);       
            sp_path_small(ii:1:min(ii+hop-1,len),3) = high;
            locations(ii:1:min(ii+hop-1,len)) = ii_index;
            % Stationary receiver positions
            for mm=1:M
                rp_path(ii:1:min(ii+hop-1,len),1,mm) = r(mm,1);
                rp_path(ii:1:min(ii+hop-1,len),2,mm) = r(mm,2);
                rp_path(ii:1:min(ii+hop-1,len),3,mm) = high;    
            end
        end

        % signal generator

        [receivers,beta_hat] = signal_generator(in,c,fs,rp_path,sp_path,L,beta,n,'o',order);

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

        p=randi([1,5]);
        noise_file=sprintf('C:\\project\\noise\\noise_%d.wav',p);
        noise_temp = audioread(noise_file);
        noise_temp = [noise_temp.' noise_temp.' noise_temp.'].';
        noise_temp=noise_temp(1:length(receivers)-n+1)';
        h_noise = rir_generator(c, fs, r, s_noise, L, beta, n, mtype, order, dim, orientation, hp_filter);
        Receivers_noise = conv2(noise_temp,h_noise);    
        Receivers_noise = Receivers_noise/mean(std(Receivers_noise'));
    %% create input

        % calc snr
        SNR_direction = 20;
        SNR_mic = 30;

        M=size(receivers,1);
        length_receives=size(receivers,2);
        %clac An
        A_x=mean(std(receivers'));
        std_n=mean(std(Receivers_noise'));

        A_n_diraction = A_x/(10^(SNR_direction/20));
        A_n_mic = A_x/(10^(SNR_mic/20));

        %create mic moise
        mic_noise = A_n_mic*randn(M,length_receives);    
        % create difuse noise
        difuse_noise = create_difuse_noise(length_receives);
        difuse_noise = difuse_noise/mean(std(difuse_noise));

        %SNR_calc = snr(squeeze(receivers(1,:)),squeeze(A_n_difuse*difuse_noise(:,1)).');
        receivers_clean = receivers;
        receivers_clean=receivers_clean/max(max(abs(receivers_clean)));
        for j=1:1
            SNR_difuse =20-2.5*(j-1);
            A_n_difuse = A_x/(10^(SNR_difuse/20));
            receivers_save=(receivers+mic_noise+A_n_difuse*difuse_noise.'+A_n_diraction*Receivers_noise).'; 
            receivers_save=receivers_save/max(max(abs(receivers_save)));
            data_together = strcat('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_',num2str(int16(1000*beta)),'_snr_',num2str(SNR_difuse),'\\dynamic_signal_',num2str(i),'.wav');
            audiowrite(data_together,receivers_save,fs)
            data_together_clean = strcat('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_',num2str(int16(1000*beta)),'_snr_',num2str(SNR_difuse),'\\dynamic_signal_clean_',num2str(i),'.wav'); 
            angle_location_file = strcat('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_',num2str(int16(1000*beta)),'_snr_',num2str(SNR_difuse),'\\angle_location_',num2str(i),'.mat'); 
            location_file = strcat('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_',num2str(int16(1000*beta)),'_snr_',num2str(SNR_difuse),'\\locations_',num2str(i),'.mat'); 
            location_small_file = strcat('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_',num2str(int16(1000*beta)),'_snr_',num2str(SNR_difuse),'\\locations_small_',num2str(i),'.wav'); 
            mic_array_file = strcat('C:\\project\\dynamic_signals\\one_speaker\\category\\T60_',num2str(int16(1000*beta)),'_snr_',num2str(SNR_difuse),'\\mic_array_',num2str(i),'.wav');   
            audiowrite(data_together_clean,receivers_clean.',fs)
            save(angle_location_file,'locations')    
            save(location_file,'sp_path')    
            save(location_small_file,'sp_path_small')   
            save(mic_array_file,'r')
        end
    end  
end
%% Plot source path and signals
figure(1);
plot3(r(1,1),r(1,2),r(1,3),'x');
hold on;
for mm = 2:M
    plot3(r(mm,1),r(mm,2),r(mm,3),'x');
end
plot3(sp_path(:,1),sp_path(:,2),sp_path(:,3),'r.');
hold on
plot3(sp_path_small(:,1),sp_path_small(:,2),sp_path_small(:,3),'r.');
axis([0 L(1) 0 L(2) 0 L(3)]);
grid on;
box on;
axis square;
hold off;

figure(2)
t = 0:1/fs:(length(in)-1)/fs;
subplot(211); plot(t,in); title('in(n)');xlabel('Time [Seconds]');ylabel('Amplitude');
subplot(212); plot(t,receivers'); title('out(n)');xlabel('Time [Seconds]');ylabel('Amplitude');
