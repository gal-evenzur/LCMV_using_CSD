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
    """Base configuration for static/structural parameters."""
    # Acoustic & Environment parameters
    c = 340                     
    fs = 16000                  
    room_height = 3.0           
    
    # Analysis parameters
    nfft = 2048                 
    hop = 512                   
    path_hop = 1024               
    
    # Array configuration
    M = 4                       
    array_radius = 0.1          
    
    # Initialization
    initial_noise_pad_sec = 1.0
    min_speech_duration = initial_noise_pad_sec + 1.0
    
    # Dynamic settings mapped from args
    seed = None
    T60 = None
    SNR_diffuse = None
    SNR_mic = None
    linear_velocity = None
    start_angle_deg = None
    end_angle_deg = None
    speaker_radius = None
    repeatMode = None
    
    # Paths
    timit_base_path = None      
    output_path = None          
    dataset_title = 'test/dynamic'
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')


def ensure_audio_length(speech: np.ndarray, target_length: int) -> np.ndarray:
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
    
    # --- 1. Parameterize Environment ---
    # Randomized room dimensions but FIXED T60 and SNR from config
    L1 = 4.0 + 0.1 * np.random.randint(1, 21)
    L2 = 4.0 + 0.1 * np.random.randint(1, 21)
    room_dim = np.array([L1, L2, config.room_height])
    
    # --- 2. Unified Geometry & Trajectory Setup ---
    start_rad = np.deg2rad(config.start_angle_deg)
    end_rad = np.deg2rad(config.end_angle_deg)
    arc_length = abs(end_rad - start_rad) * config.speaker_radius
    
    # Prevent division by zero if stationary
    v = config.linear_velocity if config.linear_velocity > 0 else 0.01
    duration_sec = max(arc_length / v, config.min_speech_duration)  # Ensure at least some active speech
    target_samples = int(duration_sec * config.fs)
    
    os.makedirs(config.plot_dir, exist_ok=True)
    plot_name = os.path.join(config.plot_dir, f"dynamic_sample_{sample_idx}_T60_{config.T60:.2f}_SNR_{config.SNR_diffuse:.1f}")
    
    simulator = AcousticTrajectorySimulator(
        room_dim=room_dim.tolist(), 
        speaker_radius=config.speaker_radius, 
        radius_noise=0.2, # Matching static default
        num_jumps=0,       # Not used for continuous
        plot_name=plot_name
    )
    
    s_first_path, labels_s1, s_second_path, labels_s2, s_noise, mic_positions = simulator.generate_continuous(
        fs=config.fs, 
        update_interval=config.path_hop, 
        len_s1=target_samples, v_s1=config.linear_velocity, 
        start_s1=start_rad, end_s1=end_rad, mode_s1=config.repeatMode,
        # Dummy params for S2 just to satisfy the function
        len_s2=target_samples, v_s2=0.0, 
        start_s2=0.0, end_s2=0.0, mode_s2='stop'
    )
    
    # Load speaker audio
    speaker_list = male_speakers if np.random.rand() < 0.5 else female_speakers
    speaker_dir = speaker_list[np.random.randint(len(speaker_list))]
    raw_speech = load_speech(get_random_speech_file(speaker_dir), config.fs)
    source_signal = ensure_audio_length(raw_speech, target_samples)

    # --- 3. Dynamic Acoustic Simulation ---
    if verbose:
        start_time = time.time()
        print(f"  Running das_generator (T60={config.T60}s)...", end="", flush=True)
        
    receiver_signals = generator.generate(
        source_signal,
        c=config.c,
        fs=config.fs,
        rp_path=mic_positions,
        sp_path=s_first_path,
        L=room_dim,
        reverberation_time=config.T60,
        nRIR=4096,
        mtypes=generator.mic_type.omnidirectional,
        orientation=[0, 0]
    )
    if verbose: print(f" done in {time.time() - start_time:.2f}s.")
    
    # --- 4. Noise Generation & Injection ---
    receiver_signals = normalize_signal(receiver_signals)
    
    # Calculate noise amplitudes strictly on the ACTIVE speech segment
    # (Doing this before padding ensures the 2 seconds of silence don't skew the SNR)
    A_x = np.mean(np.std(receiver_signals, axis=0))
    A_n_diffuse = A_x / (10 ** (config.SNR_diffuse / 20))
    A_n_mic = A_x / (10 ** (config.SNR_mic / 20))
    
    # Inject 2 Seconds of Silence to the speaker signals and labels
    pad_samples = int(config.initial_noise_pad_sec * config.fs)
    receiver_signals = np.vstack((np.zeros((pad_samples, config.M)), receiver_signals))
    sample_labels = np.concatenate((np.zeros(pad_samples), labels_s1))
    
    length_receives = receiver_signals.shape[0]
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
        'T60': config.T60,
        'SNR_diffuse': config.SNR_diffuse,
        'mic_positions': mic_positions
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parameterized dynamic acoustic trajectory samples.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of files to generate")
    parser.add_argument("--start_idx", type=int, default=1, help="Starting index for file naming")
    parser.add_argument("--seed", type=int, default=420, help="Random seed for reproducibility")
    parser.add_argument("--T60", type=float, default=0.4, help="Reverberation time in seconds")
    parser.add_argument("--SNR_diffuse", type=float, default=20.0, help="Diffuse noise SNR (dB)")
    parser.add_argument("--SNR_mic", type=float, default=30.0, help="Microphone noise SNR (dB)")
    parser.add_argument("--linear_velocity", type=float, default=1.0, help="Speaker velocity (m/s)")
    parser.add_argument("--start_angle_deg", type=float, default=0.0, help="Start angle of trajectory")
    parser.add_argument("--end_angle_deg", type=float, default=180.0, help="End angle of trajectory")
    parser.add_argument("--speaker_radius", type=float, default=1.3, help="Radius of speaker arc (m)")
    parser.add_argument("--repeatMode", type=str, default="bounce", choices=["stop", "bounce"], help="Trajectory completion mode")
    
    args = parser.parse_args()
    
    config = Config()
    config.seed = args.seed
    config.T60 = args.T60
    config.SNR_diffuse = args.SNR_diffuse
    config.SNR_mic = args.SNR_mic
    config.linear_velocity = args.linear_velocity
    config.start_angle_deg = args.start_angle_deg
    config.end_angle_deg = args.end_angle_deg
    config.speaker_radius = args.speaker_radius
    config.repeatMode = args.repeatMode
    
    np.random.seed(config.seed)
    
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
    
    print(f"\nGenerating {args.num_samples} dynamic samples for T60={config.T60}...")
    print("-" * 60)
    
    for i in range(args.start_idx, args.start_idx + args.num_samples):
        print(f"Sample {i}: Seed={config.seed + i}")
        
        # Advance seed slightly per sample to ensure variety within the same batch
        np.random.seed(config.seed + i)
        
        result = create_test_sample_dynamic(i, config, male_speakers, female_speakers)
        
        sf.write(os.path.join(output_path, f'together_{i}.wav'), result['mixture'], config.fs)
        sf.write(os.path.join(output_path, f'first_{i}.wav'), result['clean_speech'], config.fs)
        sf.write(os.path.join(output_path, f'second_{i}.wav'), np.zeros_like(result['clean_speech']), config.fs)
        
        np.save(os.path.join(output_path, f'label_location_first_{i}.npy'), result['labels'])
        np.save(os.path.join(output_path, f'label_location_second_{i}.npy'), np.zeros_like(result['labels']))
        np.savez(
            os.path.join(output_path, f'metadata_{i}.npz'),
            room_dim=result['room_dim'],
            T60=result['T60'],
            SNR_diffuse=result['SNR_diffuse'],
            mic_positions=result['mic_positions']
        )