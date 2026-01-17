function vad = create_vad_dynamic(x,hop, nfft)
    xlen = length(x);
    wlen = nfft;
    L = 1+fix((xlen-wlen)/hop);
    vad = zeros(1,L);
    for l = 0:L-1
        x_w = x(1+l*hop : wlen+l*hop);
        X = mode(x_w);
        vad(1+l) = X;
    end
end