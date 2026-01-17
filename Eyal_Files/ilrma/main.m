clear all;
close all;

% Set parameters
seed = 1; % pseudo random seed
refMic = 1; % reference microphone for back projection
resampFreq = 16000; % resampling frequency [Hz]
nSrc = 2; % number of sources
fftSize = 2048; % window length in STFT [points]
shiftSize = 1024; % shift length in STFT [points]
windowType = "hamming"; % window function used in STFT
nBases = 10; % number of bases (for ilrmaType=1, nBases is # of bases for "each" source. for ilrmaType=2, nBases is # of bases for "all" sources)
nIter = 50; % number of iterations (define by checking convergence behavior with drawConv=true)
ilrmaType = 1; % 1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)
applyNormalize = 1; % 0 or 1 or 2 (0: do not apply normalization in each iteration, 1: apply average-power-based normalization in each iteration to improve numerical stability (the monotonic decrease of the cost function may be lost), 2: apply back projection in each iteration)
applyWhitening = false; % true or false (true: apply whitening to the observed multichannel spectrograms)
drawConv = true; % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)

% Fix random seed
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed))
% Output separated signals
% for j=10:2.5:20  
%     value = num2str(j);
%     outputDir = sprintf('C:\\project\\static_signals\\two_speakers\\SNR\\SNR_%s_T60_300_SIR_0\\',value);
%     for i = 1:10
%         mixSig = audioread(sprintf('%s\\dynamic_signal_%d.wav', outputDir,i));
%         [estSig, cost] = ILRMAISS(mixSig, nSrc, resampFreq, nBases, fftSize, shiftSize, windowType, nIter, ilrmaType, refMic, applyNormalize, applyWhitening, drawConv);
%         audiowrite(sprintf('%s\\estimated_first_channel_%d.wav', outputDir,i), estSig(:,1), resampFreq); % estimated signal 1
%         audiowrite(sprintf('%s\\estimated_second_channel_%d.wav', outputDir,i), estSig(:,2), resampFreq); % estimated signal 2
%     end
% end
% 
% for j=0:5:15  
%     value = num2str(j);
%     outputDir = sprintf('C:\\project\\static_signals\\two_speakers\\SIR\\SNR_20_T60_300_SIR_%s\\',value);
%     for i = 1:10
%         mixSig = audioread(sprintf('%s\\dynamic_signal_%d.wav', outputDir,i));
%         [estSig, cost] = ILRMAISS(mixSig, nSrc, resampFreq, nBases, fftSize, shiftSize, windowType, nIter, ilrmaType, refMic, applyNormalize, applyWhitening, drawConv);
%         audiowrite(sprintf('%s\\estimated_first_channel_%d.wav', outputDir,i), estSig(:,1), resampFreq); % estimated signal 1
%         audiowrite(sprintf('%s\\estimated_second_channel_%d.wav', outputDir,i), estSig(:,2), resampFreq); % estimated signal 2
%     end
% end

for j=0:5:15  
    value = num2str(j);
    outputDir = sprintf('C:\\project\\static_signals\\two_speakers\\SIR\\SNR_20_T60_300_SIR_%s\\',value);
    for i = 1:10
        mixSig = audioread(sprintf('%s\\dynamic_signal_%d.wav', outputDir,i));
        [estSig, cost] = ILRMAISS(mixSig, nSrc, resampFreq, nBases, fftSize, shiftSize, windowType, nIter, ilrmaType, refMic, applyNormalize, applyWhitening, drawConv);
        audiowrite(sprintf('%s\\estimated_first_channel_%d.wav', outputDir,i), estSig(:,1), resampFreq); % estimated signal 1
        audiowrite(sprintf('%s\\estimated_second_channel_%d.wav', outputDir,i), estSig(:,2), resampFreq); % estimated signal 2
    end
end

figure();
plot(mixSig(:,1))
hold on
plot(estSig(:,1))
hold on
plot(estSig(:,2))




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%