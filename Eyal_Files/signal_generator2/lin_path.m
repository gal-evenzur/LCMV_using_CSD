function [p] = lin_path(rm, rc_start, rc_end, v, T, fs_dyn, fs)
M = size(rm, 2);
rm0 = mean(rm, 2);
rmc = rm-repmat(rm0, [1, M]);
Tf = norm(rc_end-rc_start)/v;
Nf = floor(Tf*fs_dyn);
vr = (rc_end-rc_start)/norm(rc_end-rc_start);
dr = vr*v/fs_dyn;
N = round(T*fs_dyn);

Nt = min(Nf, N);
p = repmat(permute(rmc, [3, 1, 2]), [Nt, 1, 1])+repmat(repmat(rc_start.', [Nt, 1])+[0:Nt-1].'*dr.', [1, 1, M]);
p = reshape(repmat(permute(p, [4, 1, 2, 3]), [ceil(fs/fs_dyn), 1, 1, 1]), [Nt*ceil(fs/fs_dyn), 3, M]);
    
