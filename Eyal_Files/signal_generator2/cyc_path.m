function p = cyc_path(rm, rc_start, rc_end, v, T, fs_dyn, fs)
M = size(rm, 2);
pfwd = lin_path(rm, rc_start, rc_end, v, T, fs_dyn, fs);
pbwd = lin_path(rm, rc_end, rc_start, v, T, fs_dyn, fs);
pcyc = [pfwd; pbwd];
Ncyc = size(pcyc, 1);
N = T*fs;
L = floor(N/Ncyc);

if L>0
    p = repmat(pcyc, [L, 1, 1]);
else
    p = zeros(0, 3, M);
end

p = [p; pcyc(1:mod(N, Ncyc), :, :)];
