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

Radius_X = 0;
Radius_Y = 0;
t = linspace(-pi/2,pi/2,angle/2); % 3 times 180 to create option to rand whole circ

z=0*t+high;

circ_mics_x = radius_mics*sin(t)+Radius_X;
circ_mics_y = radius_mics*cos(t)+Radius_Y;
circ_mics_z = ones(1,180);

r= [circ_mics_x(1) circ_mics_y(1) high; circ_mics_x(55) circ_mics_y(55) high;...
    circ_mics_x(120) circ_mics_y(120) high; circ_mics_x(180) circ_mics_y(180) high];


figure();
plot3(r(:,1),r(:,2),r(:,3),'*')
hold on
plot3(circ_mics_x,circ_mics_y,circ_mics_z,'r')





    