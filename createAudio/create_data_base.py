"""
create_data_base.py - Generate a database of reverberant multi-speaker audio mixtures.

This script generates a dataset of multi-channel audio recordings with:
- Two speakers at different locations in a simulated room
- Directional point-source noise
- Diffuse noise (spatially coherent)
- Microphone self-noise
- Ground truth VAD/location labels for each speaker

Translated from MATLAB: create_data_base.m

Dependencies:
    - numpy
    - scipy
    - soundfile
    - rir_generator (pip install rir-generator)
    - anf_generator (for diffuse noise)
"""

import numpy as np
import scipy.signal as ss
import soundfile as sf
import os
import glob
from typing import Tuple, List, Optional

import rir_generator as rir

# Import from dataset_funcs
from dataset_funcs import *


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration parameters for dataset generation."""
    
    # Acoustic parameters
    c = 340                     # Sound velocity (m/s)
    fs = 16000                  # Sample frequency (Hz)
    n_rir_samples = 4096        # Number of RIR samples
    
    # Room parameters (height is fixed at 3m)
    room_height = 3.0
    
    # Analysis parameters
    nfft = 2048                 # FFT length
    hop = 512                   # Hop size
    M = 4                       # Number of microphones
    
    # Speaker trajectory parameters
    R = 1.3                     # Speaker radius from array center (m)
    noise_R = 0.2               # Position noise radius (m)
    num_jumps = 9               # Number of trajectory segments
    
    # SNR parameters
    SNR_direction = 15          # Directional noise SNR (dB)
    SNR_mic = 30                # Microphone noise SNR (dB)
    
    # TIMIT paths
    timit_base_path = None      # Will be set at runtime
    
    # Output path
    output_path = None          # Will be set at runtime
    
    # Number of samples to generate
    num_samples = 500


def get_timit_speakers(timit_path: str) -> Tuple[List[str], List[str]]:
    """
    Scan TIMIT directory and return lists of male and female speaker paths.
    
    TIMIT naming convention:
    - F* = Female speakers (e.g., FCJF0, FDAW0)
    - M* = Male speakers (e.g., MCPM0, MDAC0)
    
    Parameters
    ----------
    timit_path : str
        Path to TIMIT root directory (containing TRAIN/ and TEST/)
    
    Returns
    -------
    male_speakers : list
        List of full paths to male speaker directories
    female_speakers : list
        List of full paths to female speaker directories
    """
    male_speakers = []
    female_speakers = []
    
    # Search in both TRAIN and TEST directories
    for subset in ['TRAIN', 'TEST']:
        subset_path = os.path.join(timit_path, subset)
        if not os.path.exists(subset_path):
            continue
        
        # Iterate through dialect regions (DR1-DR8)
        for dr_dir in glob.glob(os.path.join(subset_path, 'DR*')):
            # Iterate through speakers
            for speaker_dir in glob.glob(os.path.join(dr_dir, '*')):
                if not os.path.isdir(speaker_dir):
                    continue
                
                speaker_id = os.path.basename(speaker_dir)
                
                # Check first letter for gender
                if speaker_id.startswith('F'):
                    female_speakers.append(speaker_dir)
                elif speaker_id.startswith('M'):
                    male_speakers.append(speaker_dir)
    
    return male_speakers, female_speakers


def get_random_speech_file(speaker_dir: str) -> str:
    """
    Get a random WAV file from a speaker directory.
    
    Parameters
    ----------
    speaker_dir : str
        Path to speaker directory
    
    Returns
    -------
    wav_path : str
        Path to a random WAV file
    """
    wav_files = glob.glob(os.path.join(speaker_dir, '*.WAV'))
    if not wav_files:
        # Try lowercase extension
        wav_files = glob.glob(os.path.join(speaker_dir, '*.wav'))
    
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {speaker_dir}")
    
    return wav_files[np.random.randint(len(wav_files))]


def load_speech(wav_path: str, target_fs: int = 16000) -> np.ndarray:
    """
    Load a speech file and resample if necessary.
    
    Parameters
    ----------
    wav_path : str
        Path to WAV file
    target_fs : int
        Target sample frequency
    
    Returns
    -------
    speech : np.ndarray
        Speech signal (1D array)
    """
    speech, fs = sf.read(wav_path)
    
    # Handle stereo files
    if len(speech.shape) > 1:
        speech = speech[:, 0]
    
    # Resample if necessary
    if fs != target_fs:
        num_samples = int(len(speech) * target_fs / fs)
        speech = ss.resample(speech, num_samples)
    
    return speech


def generate_rir(
    c: float,
    fs: int,
    receiver_positions: np.ndarray,
    source_position: np.ndarray,
    room_dim: np.ndarray,
    reverberation_time: float,
    n_samples: int
) -> np.ndarray:
    """
    Generate room impulse response using rir-generator.
    
    Parameters
    ----------
    c : float
        Sound velocity (m/s)
    fs : int
        Sample frequency (Hz)
    receiver_positions : np.ndarray
        Receiver positions, shape (M, 3) for M receivers
    source_position : np.ndarray
        Source position, shape (3,)
    room_dim : np.ndarray
        Room dimensions [x, y, z] in meters
    reverberation_time : float
        Reverberation time T60 (seconds)
    n_samples : int
        Number of RIR samples
    
    Returns
    -------
    h : np.ndarray
        Room impulse response, shape (n_samples, M)
    """
    h = rir.generate(
        c=c,
        fs=fs,
        r=receiver_positions.tolist(),
        s=source_position.tolist(),
        L=room_dim.tolist(),
        reverberation_time=reverberation_time,
        nsample=n_samples
    )
    
    return h


def convolve_with_rir(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Convolve a mono signal with multi-channel RIR.
    
    Parameters
    ----------
    signal : np.ndarray
        Mono input signal, shape (N,)
    h : np.ndarray
        Room impulse response, shape (n_samples, M)
    
    Returns
    -------
    output : np.ndarray
        Convolved signal, shape (N + n_samples - 1, M)
    """
    M = h.shape[1]
    output_len = len(signal) + h.shape[0] - 1
    output = np.zeros((output_len, M))
    
    for m in range(M):
        output[:, m] = np.convolve(signal, h[:, m])
    
    return output


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal by mean of channel standard deviations."""
    if len(signal.shape) == 1:
        return signal / np.std(signal)
    else:
        return signal / np.mean(np.std(signal, axis=0))


def create_database_sample(
    sample_idx: int,
    config: Config,
    male_speakers: List[str],
    female_speakers: List[str],
    verbose: bool = True
) -> dict:
    """
    Create a single database sample.
    
    Parameters
    ----------
    sample_idx : int
        Sample index (for file naming)
    config : Config
        Configuration object
    male_speakers : list
        List of male speaker directories
    female_speakers : list
        List of female speaker directories
    verbose : bool, optional
        Verbosity flag for printing progress information. Default is False.
    Returns
    -------
    result : dict
        Dictionary with generated data and metadata
    """
    
    # --- Random room dimensions ---
    L1 = 4.0 + 0.1 * np.random.randint(1, 21)  # 4.1 to 6.0 m
    L2 = 4.0 + 0.1 * np.random.randint(1, 21)  # 4.1 to 6.0 m
    room_dim = np.array([L1, L2, config.room_height])
    
    # --- Random SNR and reverberation ---
    SNR_diffuse = 10 + np.random.randint(0, 11)  # 10 to 20 dB
    beta = 0.3 + 0.001 * np.random.randint(0, 251)  # 0.3 to 0.55 s (T60)
    
    # --- Generate speaker and mic positions ---
    pos_and_rir_time = time.time()
    (s_first, label_first, s_second, label_second, 
     s_noise, mic_positions) = create_locations_18_dynamic(
        room_dim=room_dim.tolist(),
        speaker_radius=config.R,
        radius_noise=config.noise_R,
        num_jumps=config.num_jumps
    )
    
    # --- Generate RIR for noise source ---
    h_noise = generate_rir(
        c=config.c,
        fs=config.fs,
        receiver_positions=mic_positions,
        source_position=s_noise,
        room_dim=room_dim,
        reverberation_time=beta,
        n_samples=config.n_rir_samples
    )
    if verbose: print(f"Position and RIR generation took {time.time() - pos_and_rir_time:.2f} seconds.")
    
    # --- Select speakers ---
    # Speaker 1
    if np.random.rand() < 0.5:
        speaker1_dir = male_speakers[np.random.randint(len(male_speakers))]
    else:
        speaker1_dir = female_speakers[np.random.randint(len(female_speakers))]
    
    # Speaker 2
    if np.random.rand() < 0.5:
        speaker2_dir = male_speakers[np.random.randint(len(male_speakers))]
    else:
        speaker2_dir = female_speakers[np.random.randint(len(female_speakers))]
    
    # --- Initialize storage ---
    Receivers_first_total = None
    Receivers_second_total = None
    label_first_total = None
    label_second_total = None
    
    jump_time = time.time()
    # --- Process each trajectory segment ---
    for j in range(config.num_jumps):
        # Load random speech files
        speech_1 = load_speech(get_random_speech_file(speaker1_dir), config.fs)
        speech_2 = load_speech(get_random_speech_file(speaker2_dir), config.fs)
        
        # Generate RIR for speaker 1 at position j
        h_first = generate_rir(
            c=config.c,
            fs=config.fs,
            receiver_positions=mic_positions,
            source_position=s_first[:, j],
            room_dim=room_dim,
            reverberation_time=beta,
            n_samples=config.n_rir_samples
        )
        
        # Convolve speech with RIR
        Receivers_first = convolve_with_rir(speech_1, h_first)

        # Create sample-level labels
        label_first_temp = np.ones(len(Receivers_first)) * label_first[j]
        
        # Random padding (silence between segments)
        rand_padding = np.random.randint(1, 6)
        pad_samples = int(config.fs / 2 * rand_padding)
        pad_zeros_noise = np.zeros((pad_samples, config.M))
        pad_zeros_label = np.zeros(pad_samples)
        
        # Concatenate with padding
        if j == 0:
            # Initial padding + segment
            label_first_total = np.concatenate([pad_zeros_label, pad_zeros_label, label_first_temp])
            Receivers_first_total = np.vstack([pad_zeros_noise, pad_zeros_noise, Receivers_first])
        else:
            label_first_total = np.concatenate([label_first_total, pad_zeros_label, label_first_temp])
            Receivers_first_total = np.vstack([Receivers_first_total, pad_zeros_noise, Receivers_first])
        
        # Same for speaker 2
        h_second = generate_rir(
            c=config.c,
            fs=config.fs,
            receiver_positions=mic_positions,
            source_position=s_second[:, j],
            room_dim=room_dim,
            reverberation_time=beta,
            n_samples=config.n_rir_samples
        )
        
        Receivers_second = convolve_with_rir(speech_2, h_second)
        label_second_temp = np.ones(len(Receivers_second)) * label_second[j]
        
        rand_padding = np.random.randint(1, 6)
        pad_samples = int(config.fs / 2 * rand_padding)
        pad_zeros_noise = np.zeros((pad_samples, config.M))
        pad_zeros_label = np.zeros(pad_samples)
        
        if j == 0:
            label_second_total = np.concatenate([pad_zeros_label, pad_zeros_label, label_second_temp])
            Receivers_second_total = np.vstack([pad_zeros_noise, pad_zeros_noise, Receivers_second])
        else:
            label_second_total = np.concatenate([label_second_total, pad_zeros_label, label_second_temp])
            Receivers_second_total = np.vstack([Receivers_second_total, pad_zeros_noise, Receivers_second])
    
    # --- Pad to equal length ---
    maxlen = max(len(Receivers_first_total), len(Receivers_second_total))
    
    if len(Receivers_first_total) < maxlen:
        pad_len = maxlen - len(Receivers_first_total)
        Receivers_first_total = np.vstack([Receivers_first_total, np.zeros((pad_len, config.M))])
        label_first_total = np.concatenate([label_first_total, np.zeros(pad_len)])
    
    if len(Receivers_second_total) < maxlen:
        pad_len = maxlen - len(Receivers_second_total)
        Receivers_second_total = np.vstack([Receivers_second_total, np.zeros((pad_len, config.M))])
        label_second_total = np.concatenate([label_second_total, np.zeros(pad_len)])
    
    if verbose: print(f"Trajectory processing took {time.time() - jump_time:.2f} seconds.")
    # --- Generate point-source noise (using Gaussian as placeholder) ---
    noise_len = maxlen - config.n_rir_samples + 1
    noise_temp = np.random.randn(noise_len)
    
    # Convolve with noise RIR
    Receivers_noise = convolve_with_rir(noise_temp, h_noise)
    
    # --- Normalize signals ---
    Receivers_first_total = normalize_signal(Receivers_first_total)
    Receivers_second_total = normalize_signal(Receivers_second_total)
    Receivers_noise = normalize_signal(Receivers_noise)
    
    # --- Combine speakers ---
    receivers = Receivers_first_total + Receivers_second_total
    
    M = receivers.shape[1]
    length_receives = receivers.shape[0]
    
    # --- Calculate noise amplitudes for target SNRs ---
    A_x = np.mean(np.std(receivers, axis=0))
    A_n_diffuse = A_x / (10 ** (SNR_diffuse / 20))
    A_n_direction = A_x / (10 ** (config.SNR_direction / 20))
    A_n_mic = A_x / (10 ** (config.SNR_mic / 20))
    
    # --- Create microphone noise ---
    mic_noise = A_n_mic * np.random.randn(length_receives, M)
    
    # --- Create diffuse noise ---
    # Get microphone positions in 2D for diffuse noise generation
    mic_pos_2d = mic_positions[:, :2]
    
    diff_noise = time.time()
    try:
        diffuse_noise = fun_create_diffuse_noise(
            mic_positions=mic_pos_2d,
            fs=config.fs,
            L=length_receives
        )
        diffuse_noise = normalize_signal(diffuse_noise)
    except Exception as e:
        print(f"  Warning: Diffuse noise generation failed ({e}), using Gaussian noise")
        diffuse_noise = np.random.randn(length_receives, M)
    
    # Ensure diffuse noise matches signal length
    if len(diffuse_noise) < length_receives:
        repeat_times = int(np.ceil(length_receives / len(diffuse_noise)))
        diffuse_noise = np.tile(diffuse_noise, (repeat_times, 1))
    diffuse_noise = diffuse_noise[:length_receives, :M]
    
    # Ensure Receivers_noise matches length
    if len(Receivers_noise) < length_receives:
        pad_len = length_receives - len(Receivers_noise)
        Receivers_noise = np.vstack([Receivers_noise, np.zeros((pad_len, M))])
    Receivers_noise = Receivers_noise[:length_receives, :]
    if verbose: print(f"Diffuse noise generation took {time.time() - diff_noise:.2f} seconds.")

    # --- Combine all noise sources and create mixture ---
    noise_total = mic_noise + A_n_diffuse * diffuse_noise + A_n_direction * Receivers_noise
    receivers = receivers + noise_total
    
    # --- Normalize to [-1, 1] range ---
    noise_total = noise_total / np.max(np.abs(noise_total))
    receivers = receivers / np.max(np.abs(receivers))
    Receivers_first_total = Receivers_first_total / np.max(np.abs(Receivers_first_total))
    Receivers_second_total = Receivers_second_total / np.max(np.abs(Receivers_second_total))
    
    # --- Create frame-level VAD labels ---
    vad_first_speaker = create_vad_dynamic(label_first_total, config.hop, config.nfft)
    vad_second_speaker = create_vad_dynamic(label_second_total, config.hop, config.nfft)
    
    return {
        'receivers': receivers,
        'first_speaker': Receivers_first_total,
        'second_speaker': Receivers_second_total,
        'noise': noise_total,
        'vad_first': vad_first_speaker,
        'vad_second': vad_second_speaker,
        'label_first_samples': label_first_total,
        'label_second_samples': label_second_total,
        'room_dim': room_dim,
        'T60': beta,
        'SNR_diffuse': SNR_diffuse,
        'mic_positions': mic_positions
    }


def create_database(
    num_samples: int = 500,
    timit_path: str = None,
    output_path: str = None,
    start_idx: int = 1
):
    """
    Create the full database.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    timit_path : str
        Path to TIMIT dataset root
    output_path : str
        Path to save output files
    start_idx : int
        Starting index for file naming
    """
    
    # Set default paths
    if timit_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timit_path = os.path.join(os.path.dirname(script_dir), 'TIMIT')
    
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'data')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get speaker lists
    print("Scanning TIMIT database...")
    male_speakers, female_speakers = get_timit_speakers(timit_path)
    print(f"  Found {len(male_speakers)} male speakers")
    print(f"  Found {len(female_speakers)} female speakers")
    
    if len(male_speakers) == 0 or len(female_speakers) == 0:
        raise ValueError(f"No speakers found in {timit_path}")
    
    # Configuration
    config = Config()
    config.timit_base_path = timit_path
    config.output_path = output_path
    
    print(f"\nGenerating {num_samples} samples...")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    for i in range(start_idx, start_idx + num_samples):
        print(f"\rProcessing sample {i}/{start_idx + num_samples - 1}...", end="", flush=True)
        
        try:
            # Generate sample
            result = create_database_sample(i, config, male_speakers, female_speakers)
            
            # Save audio files
            sf.write(
                os.path.join(output_path, f'first_{i}.wav'),
                result['first_speaker'],
                config.fs
            )
            sf.write(
                os.path.join(output_path, f'second_{i}.wav'),
                result['second_speaker'],
                config.fs
            )
            sf.write(
                os.path.join(output_path, f'together_{i}.wav'),
                result['receivers'],
                config.fs
            )
            
            # Save labels as numpy files
            np.save(
                os.path.join(output_path, f'label_location_first_{i}.npy'),
                result['vad_first']
            )
            np.save(
                os.path.join(output_path, f'label_location_second_{i}.npy'),
                result['vad_second']
            )
            
            # Save metadata
            np.savez(
                os.path.join(output_path, f'metadata_{i}.npz'),
                room_dim=result['room_dim'],
                T60=result['T60'],
                SNR_diffuse=result['SNR_diffuse'],
                mic_positions=result['mic_positions']
            )
            
        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            continue
    
    print(f"\n\nDatabase generation complete!")
    print(f"Files saved to: {output_path}")


num_samples = 2
create_database(num_samples=num_samples)