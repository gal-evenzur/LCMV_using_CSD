"""
Test script for fun_create_diffuse_noise function.

This script tests the diffuse noise generation and visualizes:
1. Microphone array geometry
2. Generated noise waveforms for each microphone
3. Theoretical vs generated spatial coherence between microphone pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

from dataset_funcs import fun_create_diffuse_noise


def compute_inter_mic_distances(mic_positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between microphones."""
    M = mic_positions.shape[0]
    mic_dis = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            dis = np.sqrt(np.sum((mic_positions[i, :2] - mic_positions[j, :2])**2))
            mic_dis[i, j] = dis
            mic_dis[j, i] = dis
    return mic_dis


def compute_spatial_coherence(noise: np.ndarray, fs: int, K: int):
    """
    Compute the spatial coherence between the first microphone and all others.
    
    Parameters
    ----------
    noise : np.ndarray
        Generated noise signals, shape (L, M)
    fs : int
        Sample frequency
    K : int
        FFT length
        
    Returns
    -------
    freqs : np.ndarray
        Frequency vector
    sc_generated : np.ndarray
        Generated spatial coherence, shape (M-1, K//2+1)
    """
    M = noise.shape[1]
    
    # Compute STFT for all channels
    # stft returns: frequencies, times, Zxx (complex STFT)
    f, t, X = stft(noise.T, fs=fs, window='hann', nperseg=K, 
                   noverlap=int(0.75 * K), nfft=K, return_onesided=True)
    
    # X shape: (M, K//2+1, num_frames)
    
    # Compute PSD for all channels: mean over time
    phi_x = np.mean(np.abs(X)**2, axis=2)  # Shape: (M, K//2+1)
    
    # Compute spatial coherence for each pair (mic 0 vs mic m)
    sc_generated = np.zeros((M - 1, len(f)))
    
    for m in range(1, M):
        # Cross-PSD between mic 0 and mic m
        psi_x = np.mean(X[0, :, :] * np.conj(X[m, :, :]), axis=1)
        
        # Real part of complex coherence
        sc_generated[m - 1, :] = np.real(psi_x / np.sqrt(phi_x[0, :] * phi_x[m, :]))
    
    return f, sc_generated


def theoretical_coherence_spherical(freqs: np.ndarray, distances: np.ndarray, c: float):
    """
    Compute theoretical spatial coherence for spherical diffuse field.
    
    For a spherical isotropic noise field, the coherence is:
    sinc(2 * pi * f * d / c) = sin(2*pi*f*d/c) / (2*pi*f*d/c)
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency vector in Hz
    distances : np.ndarray
        Inter-microphone distances in meters
    c : float
        Speed of sound in m/s
        
    Returns
    -------
    sc_theory : np.ndarray
        Theoretical coherence, shape (len(distances), len(freqs))
    """
    sc_theory = np.zeros((len(distances), len(freqs)))
    
    for i, d in enumerate(distances):
        if d == 0:
            sc_theory[i, :] = 1.0
        else:
            # sinc(x) in numpy is sin(pi*x)/(pi*x), so we need to adjust
            # We want sin(2*pi*f*d/c) / (2*pi*f*d/c) = sinc(2*f*d/c)
            sc_theory[i, :] = np.sinc(2 * freqs * d / c)
    
    return sc_theory


def test_and_visualize_diffuse_noise():
    """Main test and visualization function."""
    
    # --- Parameters ---
    fs = 16000
    c = 340.0
    K = 256
    duration_sec = 5  # Shorter duration for testing
    L = duration_sec * fs
    
    # --- Create microphone array geometry ---
    # Semicircular array with 4 microphones, radius 0.1m
    t = np.linspace(0, np.pi, 180)
    circ_mics_x = 0.1 * np.sin(t)
    circ_mics_y = 0.1 * np.cos(t)
    
    # Select 4 microphones at specific positions (same as MATLAB code)
    # MATLAB indices: 1, 55, 120, 180 -> Python indices: 0, 54, 119, 179
    mic_positions = np.array([
        [circ_mics_x[0], circ_mics_y[0]],
        [circ_mics_x[54], circ_mics_y[54]],
        [circ_mics_x[119], circ_mics_y[119]],
        [circ_mics_x[179], circ_mics_y[179]]
    ])
    
    M = mic_positions.shape[0]
    
    print("=" * 60)
    print("Testing fun_create_diffuse_noise")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Sample frequency: {fs} Hz")
    print(f"  Sound velocity: {c} m/s")
    print(f"  FFT length: {K}")
    print(f"  Duration: {duration_sec} seconds ({L} samples)")
    print(f"  Number of microphones: {M}")
    print(f"\nMicrophone positions (x, y) in meters:")
    for i, pos in enumerate(mic_positions):
        print(f"  Mic {i+1}: ({pos[0]:.4f}, {pos[1]:.4f})")
    
    # Compute inter-microphone distances
    mic_dis = compute_inter_mic_distances(mic_positions)
    print(f"\nInter-microphone distances (m):")
    print(f"  Mic 1 to Mic 2: {mic_dis[0, 1]:.4f}")
    print(f"  Mic 1 to Mic 3: {mic_dis[0, 2]:.4f}")
    print(f"  Mic 1 to Mic 4: {mic_dis[0, 3]:.4f}")
    
    # --- Generate diffuse noise ---
    print("\nGenerating diffuse noise...")
    noise = fun_create_diffuse_noise(
        mic_positions=mic_positions,
        fs=fs,
        c=c,
        K=K,
        L=L,
        type_nf='spherical'
    )
    
    print(f"Generated noise shape: {noise.shape}")
    print(f"  Expected: ({L}, {M})")
    
    # --- Basic statistics ---
    print(f"\nNoise statistics:")
    print(f"  Min: {noise.min():.4f}")
    print(f"  Max: {noise.max():.4f}")
    print(f"  Mean: {noise.mean():.6f}")
    print(f"  Std: {noise.std():.4f}")
    
    # --- Compute spatial coherence ---
    print("\nComputing spatial coherence...")
    freqs, sc_generated = compute_spatial_coherence(noise, fs, K)
    
    # Distances from mic 0 to mics 1, 2, 3
    distances = [mic_dis[0, m] for m in range(1, M)]
    
    # Theoretical coherence
    sc_theory = theoretical_coherence_spherical(freqs, distances, c)
    
    # --- Compute NMSE (Normalized Mean Square Error) ---
    print("\nNormalized Mean Square Error (NMSE):")
    for m in range(M - 1):
        nmse = 10 * np.log10(
            np.sum((sc_theory[m, :] - sc_generated[m, :])**2) / 
            np.sum(sc_theory[m, :]**2)
        )
        print(f"  Mic 1 vs Mic {m+2}: {nmse:.1f} dB")
    
    # --- Create visualization ---
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # --- Plot 1: Microphone array geometry ---
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Plot full semicircle
    ax1.plot(circ_mics_x, circ_mics_y, 'b-', alpha=0.3, label='Semicircle')
    
    # Plot microphone positions
    ax1.scatter(mic_positions[:, 0], mic_positions[:, 1], 
                s=100, c='red', marker='o', zorder=5, label='Microphones')
    
    # Add microphone labels
    for i, pos in enumerate(mic_positions):
        ax1.annotate(f'Mic {i+1}', (pos[0], pos[1]), 
                     textcoords="offset points", xytext=(5, 5), fontsize=10)
    
    # Plot center
    ax1.scatter(0, 0, s=100, c='black', marker='x', zorder=5, label='Center')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Microphone Array Geometry')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Noise waveforms ---
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Plot first 0.1 seconds of noise for each microphone
    t_plot = np.arange(int(0.1 * fs)) / fs * 1000  # in ms
    
    for m in range(M):
        ax2.plot(t_plot, noise[:len(t_plot), m] + m * 0.5, 
                 label=f'Mic {m+1}', alpha=0.8)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (offset for clarity)')
    ax2.set_title('Noise Waveforms (first 100 ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Spectrogram of first channel ---
    ax3 = fig.add_subplot(2, 3, 3)
    
    f_spec, t_spec, Sxx = stft(noise[:, 0], fs=fs, window='hann', 
                               nperseg=K, noverlap=int(0.75 * K))
    
    ax3.pcolormesh(t_spec, f_spec / 1000, 20 * np.log10(np.abs(Sxx) + 1e-10), 
                   shading='gouraud', cmap='viridis')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (kHz)')
    ax3.set_title('Spectrogram (Mic 1)')
    
    # --- Plots 4-6: Spatial coherence comparison ---
    num_pairs = min(3, M - 1)
    
    for m in range(num_pairs):
        ax = fig.add_subplot(2, 3, 4 + m)
        
        # Theoretical coherence
        ax.plot(freqs / 1000, sc_theory[m, :], 'k-', linewidth=1.5, 
                label='Theory')
        
        # Generated coherence
        ax.plot(freqs / 1000, sc_generated[m, :], 'b--', linewidth=1.5, 
                label='Generated')
        
        # Compute NMSE for title
        nmse = 10 * np.log10(
            np.sum((sc_theory[m, :] - sc_generated[m, :])**2) / 
            np.sum(sc_theory[m, :]**2)
        )
        
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Real(Spatial Coherence)')
        ax.set_title(f'Mic 1 vs Mic {m+2}, d={distances[m]:.3f}m\nNMSE = {nmse:.1f} dB')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, 1.1])
    
    plt.tight_layout()
    
    # Save figure
    py_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(py_path, 'Results')
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, 'test_diffuse_noise.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return noise


if __name__ == "__main__":
    noise = test_and_visualize_diffuse_noise()
