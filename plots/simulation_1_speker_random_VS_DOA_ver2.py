import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
from scipy import linalg as LA

# ==============================================================================
# 1. Subspace Tracking Functions (PASTd) & Steering Vector
# ==============================================================================

def compute_steering_vector(sector, num_freqs, fs, nperseg=2048, M=4, radius=0.1, c=343.0):
    """ Computes the geometric steering vector for a given sector. """
    angle_deg = sector * 10.0 + 5.0
    angle_rad = np.deg2rad(angle_deg)

    mic_angles = np.linspace(0, np.pi, M)
    x_m = radius * np.cos(mic_angles)
    y_m = radius * np.sin(mic_angles)

    freqs = np.fft.rfftfreq(nperseg, 1/fs)
    delays = -(x_m * np.cos(angle_rad) + y_m * np.sin(angle_rad)) / c

    v = np.exp(-1j * 2 * np.pi * freqs[:num_freqs, None] * delays[None, :])
    return torch.from_numpy(v).to(torch.complex64)

@torch.no_grad()
def pastd_rank1_whitened_precomputed(Y_speech, Rnn, steering_vector=None, beta=0.95, d_init_value=0.05, warmup_boost=5.0, tau=10.0):
    """ Runs rank-1 PASTd on a speech segment. Uses forced learning for smart init. """
    B, M, F, T = Y_speech.shape
    device = Y_speech.device
    dtype = Y_speech.dtype

    I = torch.eye(M, dtype=Rnn.dtype, device=Rnn.device).expand_as(Rnn)
    Rnn_loaded = Rnn + 1e-6 * I 
    
    eigvals, eigvecs = torch.linalg.eigh(Rnn_loaded)
    eigvals = eigvals.to(dtype=torch.complex64)  
    inv_sqrt_vals = 1 / torch.sqrt(eigvals)
    inv_sqrt_vals = inv_sqrt_vals.to(torch.complex64)

    inv_Rnn12 = eigvecs @ torch.diag_embed(inv_sqrt_vals) @ eigvecs.conj().transpose(-2, -1) 

    W = torch.zeros((B, F, M, T), dtype=dtype, device=device)

    # --- INITIALIZATION ---
    if steering_vector is not None:
        v_tensor = steering_vector.unsqueeze(0).to(device)
        w_init = torch.einsum('bfmn,bfn->bfm', inv_Rnn12, v_tensor)
        w = w_init / torch.linalg.norm(w_init, dim=-1, keepdim=True)
        # Start with a reasonable d to avoid explosion in the first frame
        d = torch.full((B, F), d_init_value, dtype=dtype, device=device) 
    else:
        w = torch.randn(B, F, M, dtype=dtype, device=device)
        w = w / torch.linalg.norm(w, dim=-1, keepdim=True)
        d = torch.full((B, F), 1e-3, dtype=dtype, device=device)

    # Time loop
    for t in range(T):
        y_t = Y_speech[:, :, :, t].permute(0, 2, 1).to(torch.complex64)
        
        x_t = torch.einsum('bfmn,bfn->bfm', inv_Rnn12, y_t).to(torch.complex64)
        w = w.to(torch.complex64)
        
        y_proj = torch.einsum('bfm,bfm->bf', w.conj(), x_t)
        d = beta * d + torch.abs(y_proj) ** 2
        gain = y_proj.conj() / d
        
        # ====================================================================
        # FORCED LEARNING (Forced learning for smart initialization)
        # Multiply the update step by a factor that starts high and decays 
        # exponentially to 1.0
        # ====================================================================
        if steering_vector is not None:
            # tau controls the decay rate (boost weakens over ~3 * tau frames)
            current_boost = 1.0 + (warmup_boost - 1.0) * np.exp(-t / tau)
            gain = gain * current_boost

        residual = x_t - w * y_proj.unsqueeze(-1)
        w = w + gain.unsqueeze(-1) * residual
        w = w / torch.linalg.norm(w, dim=-1, keepdim=True)

        W[:, :, :, t] = w

    return W, eigvals, eigvecs  

def rtf_from_subspace_tracking(W, eigvals, eigvecs, mic_ref):
    eigvals = eigvals.to(torch.float32)  
    eigvecs = eigvecs.to(torch.complex64)
    W = W.to(torch.complex64)

    sqrt_vals = torch.sqrt(eigvals).to(torch.complex64)  
    Rnn12 = eigvecs @ torch.diag_embed(sqrt_vals) @ eigvecs.conj().transpose(-2, -1)  

    a_hat = torch.einsum('bfmn,bfnt->bfmt', Rnn12, W)  
    a_hat = a_hat / a_hat[:, :, mic_ref, :].unsqueeze(-2)  
    return a_hat

# ==============================================================================
# 2. Window-based Execution & Evaluation
# ==============================================================================

def run_dynamic_experiment(wav_path, labels_path, mic_ref=0, convergence_threshold_db=-12.0):
    print(f"Loading audio from {wav_path}...")
    try:
        fs, audio = wavfile.read(wav_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{wav_path}'.")
        return

    audio = audio[:, :4] / np.max(np.abs(audio))
    
    print("Computing STFT...")
    nperseg = 2048
    noverlap = 2048 - 512
    f, t, Z = stft(audio.T, fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
    Z = np.transpose(Z, (2, 1, 0)) # Shape: (frames, freq, mics)
    num_stft_frames = Z.shape[0]

    try:
        locations = np.load(labels_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{labels_path}'.")
        return

    is_speech = locations > 0 
    diff = np.diff(is_speech.astype(int))
    starts = np.where(diff == 1)[0] + 1 
    ends = np.where(diff == -1)[0] + 1

    if is_speech[0]: starts = np.insert(starts, 0, 0)
    if is_speech[-1]: ends = np.append(ends, len(locations))

    print(f"Found {len(starts)} speech windows.")

    # --- 1. Global Noise Estimation ---
    first_speech_start = starts[0]
    print(f"Estimating global noise from frame 0 to {first_speech_start}...")
    
    z_noise = Z[:first_speech_start]
    noise_frames_count = len(z_noise)
    
    Qvv_baseline = np.einsum('tfi,tfj->fij', z_noise, z_noise.conj()) / noise_frames_count
    
    Y_noise_torch = torch.from_numpy(z_noise).permute(2, 1, 0).unsqueeze(0).to(torch.complex64)
    Y_noise_permuted = Y_noise_torch.permute(0, 2, 1, 3) 
    Rnn_torch = torch.matmul(Y_noise_permuted, Y_noise_permuted.conj().transpose(-2, -1)) / noise_frames_count

    NUP = Z.shape[1]
    M = Z.shape[2]

    global_error_rand_db = np.full(num_stft_frames, np.nan)
    global_error_smart_db = np.full(num_stft_frames, np.nan)
    
    conv_frames_rand_list = []
    conv_frames_smart_list = []
    table_results = []

    # --- 2. Process Each Window ---
    for i, (s, e) in enumerate(zip(starts, ends)):
        e = min(e, num_stft_frames) 
        if e <= s: continue

        T_window = e - s
        sector = locations[s]
        print(f"\nProcessing Window {i+1}: Frames {s} to {e} (Sector {sector})")

        z_speech = Z[s:e]

        # -- Calculate GEVD (Baseline) --
        Ryy_window = np.einsum('tfi,tfj->fij', z_speech, z_speech.conj()) / T_window
        gevd_rtf = np.zeros((NUP, M), dtype=complex)
        for freq in range(NUP):
            w_eig, v_eig = LA.eig(Ryy_window[freq], Qvv_baseline[freq] + 1e-6 * np.eye(M))
            max_idx = np.argmax(np.real(w_eig))
            principal_vec = v_eig[:, max_idx]
            gevd_rtf[freq, :] = principal_vec / principal_vec[mic_ref]

        Y_speech_torch = torch.from_numpy(z_speech).permute(2, 1, 0).unsqueeze(0).to(torch.complex64)

        # -- Method A: PASTd (Random Init) --
        W_rand, eigv_r, eigvec_r = pastd_rank1_whitened_precomputed(Y_speech_torch, Rnn_torch, beta=0.95)
        ahat_rand = rtf_from_subspace_tracking(W_rand, eigv_r, eigvec_r, mic_ref).squeeze(0).numpy()

        # -- Method B: PASTd (Smart Init with FORCED LEARNING) --
        steer_vec = compute_steering_vector(sector=sector, num_freqs=NUP, fs=fs)
        
        # Adjust d_init_value, warmup_boost, and tau here to tune performance
        W_smart, eigv_s, eigvec_s = pastd_rank1_whitened_precomputed(
    Y_speech_torch, 
    Rnn_torch, 
    steering_vector=steer_vec, 
    beta=0.95, 
    d_init_value=0.01,     # התנגדות נמוכה יותר (מאפשר למידה טבעית מהירה יותר)
    warmup_boost=2.5,      # בוסט עדין מאוד במקום 15
    tau=5.0                # דועך מהר מאוד (תוך ~15 פריימים הבוסט נעלם)
)
        ahat_smart = rtf_from_subspace_tracking(W_smart, eigv_s, eigvec_s, mic_ref).squeeze(0).numpy()

        # -- Calculate Errors --
        err_rand = np.zeros(T_window)
        err_smart = np.zeros(T_window)
        ref_norm = np.linalg.norm(gevd_rtf, axis=1) + 1e-10
        
        for t_idx in range(T_window):
            err_rand[t_idx] = np.mean(np.linalg.norm(ahat_rand[:, :, t_idx] - gevd_rtf, axis=1) / ref_norm)
            err_smart[t_idx] = np.mean(np.linalg.norm(ahat_smart[:, :, t_idx] - gevd_rtf, axis=1) / ref_norm)

        err_rand_db = 20 * np.log10(err_rand + 1e-10)
        err_smart_db = 20 * np.log10(err_smart + 1e-10)
        
        global_error_rand_db[s:e] = err_rand_db
        global_error_smart_db[s:e] = err_smart_db

        # -- Convergence Analysis --
        conv_idx_rand = np.where(err_rand_db < convergence_threshold_db)[0]
        conv_idx_smart = np.where(err_smart_db < convergence_threshold_db)[0]

        c_rand = conv_idx_rand[0] if len(conv_idx_rand) > 0 else None
        c_smart = conv_idx_smart[0] if len(conv_idx_smart) > 0 else None

        if c_rand is not None: conv_frames_rand_list.append(c_rand)
        if c_smart is not None: conv_frames_smart_list.append(c_smart)

        max_conv = max((c for c in [c_rand, c_smart] if c is not None), default=T_window)
        eval_frames = min(max_conv + 1, T_window) 
        
        avg_err_transient_rand = np.mean(err_rand_db[:eval_frames])
        avg_err_transient_smart = np.mean(err_smart_db[:eval_frames])

        table_results.append({
            'window': i + 1,
            'frames': T_window,
            'conv_rand': c_rand if c_rand is not None else "DNC",
            'conv_smart': c_smart if c_smart is not None else "DNC",
            'trans_err_rand': avg_err_transient_rand,
            'trans_err_smart': avg_err_transient_smart
        })

    # --- 3. Summary & Table ---
    print("\n" + "="*80)
    print(" " * 25 + "PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Window':<8} | {'Total Frames':<13} | {'Conv. Frames (Rand)':<20} | {'Conv. Frames (Smart)':<20} | {'Avg Pre-Conv Err (Rand)':<25} | {'Avg Pre-Conv Err (Smart)'}")
    print("-" * 120)
    for res in table_results:
        print(f"{res['window']:<8} | {res['frames']:<13} | {str(res['conv_rand']):<20} | {str(res['conv_smart']):<20} | {res['trans_err_rand']:<20.2f} dB | {res['trans_err_smart']:.2f} dB")
    print("-" * 120)
    
    avg_conv_rand = np.mean(conv_frames_rand_list) if conv_frames_rand_list else np.nan
    avg_conv_smart = np.mean(conv_frames_smart_list) if conv_frames_smart_list else np.nan
    print(f"AVERAGE CONVERGENCE: Random = {avg_conv_rand:.2f} frames | Smart = {avg_conv_smart:.2f} frames")
    print("="*80 + "\n")

    # --- 4. Plotting ---
    plt.figure(figsize=(14, 7))
    
    plt.plot(np.arange(num_stft_frames), global_error_rand_db, 
             linewidth=2.5, color='tab:blue', alpha=0.5, label='Random Init Error')
    
    plt.plot(np.arange(num_stft_frames), global_error_smart_db, 
             linewidth=2, color='tab:orange', linestyle='--', alpha=1.0, label='Smart Init (Steering) Error')
    
    plt.axhline(y=convergence_threshold_db, color='red', linestyle=':', alpha=0.8, label=f'Convergence Threshold ({convergence_threshold_db} dB)')

    for s, e in zip(starts, ends):
        plt.axvspan(s, min(e, num_stft_frames), color='gray', alpha=0.1)

    plt.title('PASTd RTF Estimation: Random vs. Smart Initialization', fontsize=14)
    plt.xlabel('Global Time Frame (STFT index)', fontsize=12)
    plt.ylabel('Misalignment Error vs Window GEVD (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure clear visibility for the legend
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_dynamic_experiment(
        wav_path='first_1.wav', 
        labels_path='label_location_first_1.npy',
        mic_ref=0,
        convergence_threshold_db=-12.0
    )