function  [x]=create_difuse_noise(L)



    % Initialization
    Fs = 16000;                % Sample frequency (Hz)
    c = 340;                  % Sound velocity (m/s)
    K = 256;                  % FFT length
    M = 4;                    % Number of sensors
    d = 0.2;                  % Inter sensor distance (m)
    type_nf = 'spherical';    % Type of noise field:
                              % 'spherical' or 'cylindrical'

    %% Generate M mutually independent input signals of length L
    n = randn(L,M);

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
                        DC(p,q,:) = sinc(ww*abs(p-q)*d/(c*pi));

                    case 'cylindrical'
                        DC(p,q,:) = bessel(0,ww*abs(p-q)*d/c);

                    otherwise
                        error('Unknown noise field.')
                end
            end
        end
    end

    %% Generate sensor signals with desired spatial coherence
    % Mix signals
    x = mix_signals(n,DC,'cholesky');