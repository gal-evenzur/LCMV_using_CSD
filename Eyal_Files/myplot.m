function myplot(data, fs, start_time, end_time)

N_h = 0.032*fs;
pos1=[0.09 0.385 0.875 0.55];
pos2=[0.09 0.15 0.875 0.2];

if nargin < 3
    start_time = 0;
    end_time = length(data)/fs;
end

% subplot(212);
% plot(0:1/fs:end_time-start_time-1/fs,1.2*data(start_time*fs+1:end_time*fs));
% axis([0 end_time-start_time -1 1]);
% xlabel('Time [s]');
% ylabel('Amplitude');
% box on;
% gca2 = gca;
% set(gca2,'position',pos2,'YTick',[-1 0 1],'XTick',0:0.5:end_time-start_time);
% %colorbar;
% subplot(211);

spectro(.3*data(start_time*fs+1:end),2*N_h,fs,N_h,75,@hann,-75);
xlabel('Sec');
axis([0 end_time-start_time 0 fs/2e3]);
gca1 = gca;
%set(gca1,'position',pos1,'XTick',[]);
%colorbar

function varargout = spectro(x,N,fs,L,overlap,wintype,mag_th)
%SPECTRO Function to generate a spectrogram
%   By Danny Nguyen - ECE 6255 - Spring 2005
%   x: Data to plot
%   N: Length of FFT to be used
%   fs: Sampling frequency of input
%   L: Length of STFT to be used
%   overlap: Percentage of overlap for the STFT's (between 0 and 99)
%   wintype: Window type to be used

% Set defaults if undefined
if nargin < 2 || isempty(N), N = 512; end             % 512 Length FFT
if nargin < 3 || isempty(fs), fs = 16000; end         % 16kHz sampling rate
if nargin < 4 || isempty(L), L = 40; end              % Wideband STFT
if nargin < 5 || isempty(overlap), overlap = 90; end  % 90% overlap
if nargin < 6 || isempty(wintype), wintype = @hamming; end
if nargin < 7 || isempty(mag_th), mag_th = -85; end

% Initializations
win = window(wintype,L);            % Set up window
inc = floor(L*(1-overlap/100));     % Determine shift distance for window
x = x(:);                           % Arranges data into a vertical column
j = 1;                              % Initialize counter
tend = floor((length(x)-L)/inc)+1;  % End time for sample

spec=zeros(L,tend);         % Initialize matrix for more speed
for i=1:inc:length(x)-L
    spec(:,j)=x(i:i+L-1).*win;  % Copy windowed values to matrix
    j=j+1;
end
spec=fft(spec,N,1);         % Perform FFT on samples

% Remove symmetry due to FFT
spec = abs(spec(1:floor(size(spec,1)/2)+1,:));

% spec = spec./max(max(spec)); % normalize so max magnitude will be 0 db
spec = max(spec, 10^(mag_th/20)); % clip everything below mag_th dB

% Display matrix
taxis=0:(inc)/fs:(length(x))/fs;
faxis=(0:fs/N:fs/2)/1e3;

warning off MATLAB:log:logOfZero
imagesc(taxis,faxis,db(spec));
warning on MATLAB:log:logOfZero

% Set colors and labels
axis xy;
caxis([-60 0]);
colormap(flipud(hot));
shading interp;
xlabel('Time [s]'),ylabel('Frequency [kHz]')
% Allows the user to save the spectrogram data matrix
if nargout==1
    varargout(1)={spec};
end