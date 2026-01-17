% TEST_PESQBIN Demonstrates the use of the PESQBIN function.
%
%   See also PESQBIN.

%   Author: Kamil Wojcicki, UTD, November 2011

clear all; close all; clc; randn('seed',0); rand('seed',0); fprintf('.\n');

     % name of executable file for PESQ calculation
    binary = 'pesq2.exe';

    % specify path to folder with reference and degraded audio files in it
    pathaudio = 'sounds';
    folder_to_work = 'C:\project\dynamic_signals\two_speakers_WSJ\SNR\SNR_20_T60_300_SIR_0\results_overlap\';
    first_clean = audioread(strcat(folder_to_work,'first_clean_10.wav'));
    second_clean = audioread(strcat(folder_to_work,'second_clean_10.wav'));
    first_est = audioread(strcat(folder_to_work,'separating_speaker_number_0_10.wav'));
    second_est = audioread(strcat(folder_to_work,'separating_speaker_number_1_10.wav'));
    noisy = audioread(strcat(folder_to_work,'noisy_signal_10.wav'));
    
    % compute NB-PESQ and WB-PESQ scores
    scores.nb = pesq2_mtlb( first_clean, noisy, 16000, 'nb',binary,pathaudio );
    scores.wb = pesq2_mtlb( first_clean, noisy, 16000, 'wb',binary,pathaudio );

    % display results to screen
    fprintf( 'NB PESQ MOS = %5.3f\n', scores.nb(1) );
    fprintf( 'NB MOS LQO  = %5.3f\n', scores.nb(2) );
    fprintf( 'WB MOS LQO  = %5.3f\n', scores.wb );
 
    % example output
    %    NB PESQ MOS = 1.969
    %    NB MOS LQO  = 1.607
    %    WB MOS LQO  = 1.083


% EOF
