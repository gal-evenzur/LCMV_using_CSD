import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

def visualize_audio_and_angles(sample_idx, data_dir='data'):
    """
    Plots the mixed audio with speaker highlights and their ground truth angles.
    
    Parameters
    ----------
    sample_idx : int
        The index of the sample to visualize (e.g., 1).
    data_dir : str
        Path to the folder containing the generated .wav and .npy files.
    """
    
    # --- 1. Load Data ---
    # Construct file paths
    mixed_wav = os.path.join(data_dir, f'together_{sample_idx}.wav')
    s1_wav = os.path.join(data_dir, f'first_{sample_idx}.wav')
    s2_wav = os.path.join(data_dir, f'second_{sample_idx}.wav')
    label1_npy = os.path.join(data_dir, f'label_location_first_{sample_idx}.npy')
    label2_npy = os.path.join(data_dir, f'label_location_second_{sample_idx}.npy')
    
    # Check if files exist
    for f in [mixed_wav, s1_wav, s2_wav, label1_npy, label2_npy]:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return

    # Read Audio
    y_mix, fs = sf.read(mixed_wav)
    y_s1, _ = sf.read(s1_wav)
    y_s2, _ = sf.read(s2_wav)
    
    # Read Labels (Frame-based classes 0-17)
    l1 = np.load(label1_npy)
    l2 = np.load(label2_npy)
    
    # --- 2. Configuration & alignment ---
    # From your create_data_base.py config
    nfft = 2048
    hop = 512
    
    # Create time axes
    time_audio = np.linspace(0, len(y_mix)/fs, len(y_mix))
    num_frames = len(l1)
    time_frames = np.arange(num_frames) * hop / fs
    
    # Convert labels (0-17) to degrees (5-175)
    # Formula: angle = label * 10 + 5
    ang1 = l1 * 10 + 5
    ang2 = l2 * 10 + 5
    
    # --- 3. Compute VAD for Masking ---
    # We calculate the envelope of the clean signals to know when to highlight/plot
    # Smooth the signal for better detection
    def get_envelope(sig, frame_size=1024):
        return np.convolve(np.abs(sig[:, 0] if sig.ndim > 1 else sig), np.ones(frame_size)/frame_size, mode='same')

    env1 = get_envelope(y_s1)
    env2 = get_envelope(y_s2)
    
    # Define a simple threshold (e.g., 5% of max volume)
    threshold = 0.05 * np.max(np.abs(y_mix))
    vad1_mask = env1 > threshold
    vad2_mask = env2 > threshold
    
    # Downsample VAD mask to match frame rate for the Angle plot
    # (Simple interpolation)
    vad1_frame_mask = np.interp(time_frames, time_audio, vad1_mask) > 0.5
    vad2_frame_mask = np.interp(time_frames, time_audio, vad2_mask) > 0.5

    # --- 4. Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # === Subplot 1: Audio Waveform with Highlights ===
    # Plot the background mixed signal (light gray)
    ax1.plot(time_audio, y_mix[:, 0] if y_mix.ndim > 1 else y_mix, color='lightgray', label='Mixture')
    
    # Highlight Speaker 1 (Blue)
    # We use `fill_between` where the VAD is active
    ax1.fill_between(time_audio, -1, 1, where=vad1_mask, color='blue', alpha=0.3, label='Speaker 1 Active')
    
    # Highlight Speaker 2 (Orange)
    ax1.fill_between(time_audio, -1, 1, where=vad2_mask, color='orange', alpha=0.3, label='Speaker 2 Active')
    
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Audio Mixture: Sample {sample_idx}")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # === Subplot 2: Angles over Time ===
    # We plot the angles, but set values to NaN where VAD is false to hide them
    ang1_masked = ang1.copy().astype(float)
    ang2_masked = ang2.copy().astype(float)
    
    ang1_masked[~vad1_frame_mask] = np.nan
    ang2_masked[~vad2_frame_mask] = np.nan
    
    ax2.plot(time_frames, ang1_masked, 'b-o', markersize=3, label='Speaker 1 Angle')
    ax2.plot(time_frames, ang2_masked, 'orange', marker='o', markersize=3, label='Speaker 2 Angle')
    
    ax2.set_ylabel("Angle of Arrival (Degrees)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, 180)
    ax2.set_yticks(np.arange(0, 181, 30))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'visualization_sample_{sample_idx}.png'))
    

# Example Usage:
visualize_audio_and_angles(1, data_dir='/home/evenzug/LCMV_using_CSD/Python_Create_Data_Base/data')