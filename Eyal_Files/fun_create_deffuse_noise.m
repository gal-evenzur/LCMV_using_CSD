function [noise]=fun_create_deffuse_noise()
% Initialization
Fs = 16000; % Sample frequency (Hz)
c = 340; % Sound velocity (m/s)
K = 256; % FFT length
M = 4; % Number of sensors
type_nf = 'spherical'; % Type of noise field: 'spherical' or 'cylindrical'
L = 20*Fs; % Data length

%% create microphones matrix distance

t = linspace(0,pi,180); 
circ_mics_x = 0.1*sin(t);
circ_mics_y = 0.1*cos(t);

r= [circ_mics_x(1) circ_mics_y(1); circ_mics_x(55) circ_mics_y(55);...
    circ_mics_x(120) circ_mics_y(120); circ_mics_x(180) circ_mics_y(180)];

mic_dis = zeros(M);
for i =1:M
    for j=i:M
        dis = sqrt( (r(i,1)-r(j,1))^2 + (r(i,2)-r(j,2))^2 );
        mic_dis(i,j) = dis;
        mic_dis(j,i) = dis;
    end
end

%% Generate M mutually 'independent' babble speech input signals
noise_file=sprintf('C:\\Users\\user\\Desktop\\ANF-Generator-master\\cafe_ambience_%d.wav',randi(3));
[data,~] = audioread(noise_file);
% data = resample(data,1,2);
data = resample(data,1,3);
% data = randn(size(data));
Fs_data = 16000;
if Fs ~= Fs_data
    error('Sample frequency of input file is incorrect.');
end

start_signal = randi(length(data)-M*L);
data = data - mean(data);
data = data(start_signal:end);
babble = zeros(L,M);
for m=1:M
    babble(:,m) = data((m-1)*L+1:m*L);
end

%% Generate matrix with desired spatial coherence
ww = 2*pi*Fs*(0:K/2)/K;
DC = zeros(M,M,K/2+1);
for p = 1:M
    for q = 1:M
        if p == q
            DC(p,q,:) = ones(1,1,K/2+1);
        else
            switch lower(type_nf)
                case 'spherical'
                    DC(p,q,:) = sinc(ww*mic_dis(p,q)/(c*pi));
            end
        end
    end
end

%% Generate sensor signals with desired spatial coherence
noise = mix_signals(babble,DC,'cholesky');


% %% Compare desired and generated coherence
% K_eval = K;
% ww = 2*pi*Fs*(0:K_eval/2)/K_eval;
% sc_theory = zeros(M-1,K_eval/2+1);
% sc_generated = zeros(M-1,K_eval/2+1);
% 
% % Calculalte STFT and PSD of all output signals
% X = stft(x,'Window',hanning(K_eval),'OverlapLength',0.75*K_eval,'FFTLength',K_eval,'Centered',false);
% X = X(1:K_eval/2+1,:,:);
% phi_x = mean(abs(X).^2,2);
% 
% % Calculate spatial coherence of desired and generated signals
% for m = 1:M-1
%     switch lower(type_nf)
%         case 'spherical'
% %             sc_theory(m,:) = sinc(ww*m*d/(c*pi));
%             sc_theory(m,:) = sinc(ww*mic_dis(1,m+1)/(c*pi));
%             
%         case 'cylindrical'
%             sc_theory(m,:) = bessel(0,ww*m*d/c);
% %             sc_theory(m,:) =  bessel(0,ww*mic_dis(1,m+1)/c);
%     end
%     
%     % Compute cross-PSD of x_1 and x_(m+1)
%     psi_x =  mean(X(:,:,1) .* conj(X(:,:,m+1)),2);
%     
%     % Compute real-part of complex coherence between x_1 and x_(m+1)
%     sc_generated(m,:) = real(psi_x ./ sqrt(phi_x(:,1,1) .* phi_x(:,1,m+1))).';
% end
% 
% % Calculate normalized mean square error
% NMSE = zeros(M,1);
% for m = 1:M-1
%     NMSE(m) = 10*log10(sum(((sc_theory(m,:))-(sc_generated(m,:))).^2)./sum((sc_theory(m,:)).^2));
% end
% 
% % Plot spatial coherence of two sensor pairs
% figure(1);
% MM=min(3,M-1);
% Freqs=0:(Fs/2)/(K/2):Fs/2;
% for m = 1:MM
%     subplot(MM,1,m);
%     plot(Freqs/1000,sc_theory(m,:),'-k','LineWidth',1.5)
%     hold on;
%     plot(Freqs/1000,sc_generated(m,:),'-.b','LineWidth',1.5)
%     hold off;
%     xlabel('Frequency [kHz]');
%     ylabel('Real(Spatial Coherence)');
%     title(sprintf('Inter sensor distance %1.2f m',m*d));
%     legend('Theory',sprintf('Proposed Method (NMSE = %2.1f dB)',NMSE(m)));
%     grid on;
% end




