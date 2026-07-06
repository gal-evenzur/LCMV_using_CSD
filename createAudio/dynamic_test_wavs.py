import os
import time
import argparse
from typing import List, Dict
import numpy as np
import soundfile as sf
import das_generator as generator

# Import from your existing utility modules
from create_data_base import (
    get_timit_speakers,
    get_random_speech_file,
    load_speech,
    normalize_signal
)
from dataset_funcs import (
    AcousticTrajectorySimulator,
    fun_create_diffuse_noise,
    create_vad_dynamic
)


class Config:
    """Configuration parameters for dynamic dataset generation."""
    seed = 15
    
    # Acoustic & Environment parameters
    c = 340                     # Sound velocity (m/s)
    fs = 16000                  # Sample frequency (Hz)
    room_height = 3.0           # Room height fixed at 3m
    
    # Analysis parameters
    nfft = 2048                 # FFT length
    hop = 512                   # Hop size for STFT and VAD
    path_hop = 1024               # Update interval for dynamic trajectory
    
    # Array configuration
    M = 4                       # Number of microphones
    array_radius = 0.1          # Radius of the mic array
    
    # Paper-specific Dynamic Parameters
    speaker_radius = 1.3        # Distance of arc from array center (m)
    linear_velocity = np.random.choice([1, 2, 3])         # Speaker movement speed (m/s) EXACTLY as per paper
    angle_resolution = 10       # DOA resolution in degrees
    repeatMode = 'bounce'         # stop or bounce  
    
    # Missing parameters (Assumed, please update based on the PDF)
    start_angle_deg = 0         # Starting angle of the arc
    end_angle_deg = 180         # Ending angle of the arc
    
    # SNR parameters
    SNR_mic = np.random.choice([20, 30])                # Microphone noise SNR (dB)
    SNR_diffuse = np.random.choice([10, 20, 30])            # Diffuse noise SNR (dB)
    
    # Initialization
    initial_noise_pad_sec = 2.0 # Seconds of pure noise before speech starts
    
    # Paths
    timit_base_path = None      # Set at runtime
    output_path = None          # Set at runtime
    
    num_samples = 6
    start_idx = 3
    dataset_title = 'test/dynamic'

    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')


def ensure_audio_length(speech: np.ndarray, target_length: int) -> np.ndarray:
    """Loops or truncates audio to match the exact time required to complete the arc."""
    if len(speech) < target_length:
        repeats = int(np.ceil(target_length / len(speech)))
        speech = np.tile(speech, repeats)
    return speech[:target_length]


def create_test_sample_dynamic(
    sample_idx: int,
    config: Config,
    male_speakers: List[str],
    female_speakers: List[str],
    verbose: bool = True
) -> Dict:
    
    # --- 1. Parameterize Environment (Reverberation & SNR) ---
    # The paper tests several reverberation times and SNRs. 
    # Here we randomly sample from a realistic range for each generated file.
    L1 = 4.0 + 0.1 * np.random.randint(1, 21)
    L2 = 4.0 + 0.1 * np.random.randint(1, 21)
    room_dim = np.array([L1, L2, config.room_height])
    
    T60 = np.random.choice([0.2, 0.4, 0.6])      # Parameterized reverberation times
    SNR_diffuse = config.SNR_diffuse

    # --- 2 & 3 & 4. Unified Geometry & Trajectory Setup ---
    # Calculate target samples
    start_rad = np.deg2rad(config.start_angle_deg)
    end_rad = np.deg2rad(config.end_angle_deg)
    arc_length = abs(end_rad - start_rad) * config.speaker_radius
    duration_sec = max(arc_length / config.linear_velocity, 6.0)
    target_samples = int(duration_sec * config.fs)
    
    # Initialize the Simulator (matching static params)
    plot_name = f"dynamic_sample_{sample_idx}_T60_{T60:.1f}_SNR_{SNR_diffuse:.1f}"
    plot_name = os.path.join(config.plot_dir, plot_name)
    simulator = AcousticTrajectorySimulator(
        room_dim=room_dim.tolist(), 
        speaker_radius=config.speaker_radius, 
        radius_noise=0.2, # Matching static default
        num_jumps=0,       # Not used for continuous
        plot_name=plot_name
    )
    
    # Generate continuous paths for both speakers (we'll just use speaker 1 for now)
    s_first_path, labels_s1, s_second_path, labels_s2, s_noise, mic_positions = simulator.generate_continuous(
        fs=config.fs, 
        update_interval=config.path_hop, 
        len_s1=target_samples, v_s1=config.linear_velocity, 
        start_s1=start_rad, end_s1=end_rad, mode_s1=config.repeatMode,
        # Dummy params for S2 just to satisfy the function
        len_s2=target_samples, v_s2=0.0, 
        start_s2=0.0, end_s2=0.0, mode_s2='stop'
    )
    
    sp_path = s_first_path
    sample_labels = labels_s1
    
    # Load speaker audio
    speaker_list = male_speakers if np.random.rand() < 0.5 else female_speakers
    speaker_dir = speaker_list[np.random.randint(len(speaker_list))]
    raw_speech = load_speech(get_random_speech_file(speaker_dir), config.fs)
    source_signal = ensure_audio_length(raw_speech, target_samples)

    
    # --- 5. Dynamic Acoustic Simulation ---
    if verbose:
        start_time = time.time()
        print(f"  Running das_generator (T60={T60}s)...", end="", flush=True)
        
    receiver_signals = generator.generate(
        source_signal,
        c=config.c,
        fs=config.fs,
        rp_path=mic_positions,
        sp_path=sp_path,
        L=room_dim,
        reverberation_time=T60,
        nRIR=4096,
        mtypes=generator.mic_type.omnidirectional,
        orientation=[0, 0]
    )
    
    if verbose: print(f" done in {time.time() - start_time:.2f}s.")
    
    # --- 6. Noise Generation & Injection ---
    receiver_signals = normalize_signal(receiver_signals)
    
    # Calculate noise amplitudes strictly on the ACTIVE speech segment
    # (Doing this before padding ensures the 2 seconds of silence don't skew the SNR)
    A_x = np.mean(np.std(receiver_signals, axis=0))
    A_n_diffuse = A_x / (10 ** (SNR_diffuse / 20))
    A_n_mic = A_x / (10 ** (config.SNR_mic / 20))
    
    # Inject 2 Seconds of Silence to the speaker signals and labels
    pad_samples = int(config.initial_noise_pad_sec * config.fs)
    receiver_signals = np.vstack((np.zeros((pad_samples, config.M)), receiver_signals))
    sample_labels = np.concatenate((np.zeros(pad_samples), sample_labels))
    
    length_receives = receiver_signals.shape[0]
    
    # Generate Noise for the entirely new length
    mic_noise = A_n_mic * np.random.randn(length_receives, config.M)
    
    mic_pos_2d = mic_positions[:, :2]
    try:
        diffuse_noise = fun_create_diffuse_noise(
            mic_positions=mic_pos_2d,
            fs=config.fs,
            L=length_receives
        )
        diffuse_noise = normalize_signal(diffuse_noise)
    except Exception as e:
        if verbose: print(f"  Warning: Diffuse noise failed ({e}), falling back to Gaussian.")
        diffuse_noise = np.random.randn(length_receives, config.M)
        
    if len(diffuse_noise) < length_receives:
        repeat_times = int(np.ceil(length_receives / len(diffuse_noise)))
        diffuse_noise = np.tile(diffuse_noise, (repeat_times, 1))
    diffuse_noise = diffuse_noise[:length_receives, :config.M]
    
    # Mix
    noise_total = mic_noise + A_n_diffuse * diffuse_noise
    mixture = receiver_signals + noise_total
    
    # Normalize outputs to [-1, 1]
    noise_total = noise_total / np.max(np.abs(noise_total))
    mixture = mixture / np.max(np.abs(mixture))
    clean_speech_multi = receiver_signals / np.max(np.abs(receiver_signals))
    
    # --- 7. Downsample Labels for Classifier ---
    # Convert sample-by-sample angles to frame-level classes
    vad_labels = create_vad_dynamic(sample_labels, config.hop, config.nfft)
    
    return {
        'mixture': mixture,
        'clean_speech': clean_speech_multi,
        'noise': noise_total,
        'labels': vad_labels,
        'raw_sample_labels': sample_labels,
        'room_dim': room_dim,
        'T60': T60,
        'SNR_diffuse': SNR_diffuse,
        'mic_positions': mic_positions
    }


if __name__ == "__main__":
    config = Config()
    
    parser = argparse.ArgumentParser(description="Generate dynamic test samples (Arc trajectory)")
    parser.add_argument("--num_samples", type=int, default=config.num_samples, help="Number of files to generate")
    parser.add_argument("--start_idx", type=int, default=config.start_idx)
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_path = os.path.dirname(script_dir)
    
    timit_path = os.path.join(workspace_path, 'data', 'TIMIT')
    sim_audio_path = os.path.join(workspace_path, 'data', 'simulated_audio')
    output_path = os.path.join(sim_audio_path, config.dataset_title)
    
    os.makedirs(output_path, exist_ok=True)
    config.timit_base_path = timit_path
    config.output_path = output_path
    
    print("Scanning TIMIT database...")
    male_speakers, female_speakers = get_timit_speakers(timit_path)
    
    print(f"\nGenerating {args.num_samples} dynamic samples...")
    print("-" * 60)
    
    for i in range(args.start_idx, args.start_idx + args.num_samples):
        print(f"\nSample {i}/{args.start_idx + args.num_samples - 1}:")
        
        result = create_test_sample_dynamic(i, config, male_speakers, female_speakers)
        
        # Save Audio
        sf.write(os.path.join(output_path, f'together_{i}.wav'), result['mixture'], config.fs)
        sf.write(os.path.join(output_path, f'first_{i}.wav'), result['clean_speech'], config.fs)
        second_wav = np.zeros_like(result['clean_speech'])  
        sf.write(os.path.join(output_path, f'second_{i}.wav'), second_wav, config.fs)
        
        second_labels = np.zeros_like(result['labels'])

        # Save Labels & Metadata
        np.save(os.path.join(output_path, f'label_location_first_{i}.npy'), result['labels'])
        np.save(os.path.join(output_path, f'label_location_second_{i}.npy'), second_labels)
        np.savez(
            os.path.join(output_path, f'metadata_{i}.npz'),
            room_dim=result['room_dim'],
            T60=result['T60'],
            SNR_diffuse=result['SNR_diffuse'],
            mic_positions=result['mic_positions']
        )
        
    print(f"\nGeneration complete! Saved to {output_path}")