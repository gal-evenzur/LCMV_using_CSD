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
    create_semicircular_mic_array,
    generate_source_path,
    fun_create_diffuse_noise,
    create_vad_dynamic
)


class ConcurrentConfig:
    """Configuration parameters for the 40s concurrent dynamic dataset."""
    seed = 100
    
    # Acoustic & Environment parameters
    c = 340                     
    fs = 16000                  
    room_height = 3.0           
    
    # Analysis parameters
    nfft = 2048                 
    hop = 512                   
    path_hop = 32               
    
    # Array configuration
    M = 4                       
    array_radius = 0.1          
    
    # Paper-specific Parameters (Section 7.1.3)
    num_samples = 4            # "Ten 40-s-long signals"
    total_duration = 40.0       # Seconds
    static_duration = 10.0      # Time each speaker gets alone to facilitate RTF
    concurrent_duration = 20.0  # Time both speakers are active and moving
    
    speaker_radius = 1.3        
    linear_velocity = 1.0 / 3.0 # "One meter every 3 s" (approx 0.333 m/s)
    angle_resolution = 10       
    
    # Trajectory Angles (Degrees)
    spk1_start_deg = 0.0
    spk1_end_deg = 140.0
    
    spk2_start_deg = 160.0
    spk2_end_deg = 180.0
    
    # SNR parameters
    SNR_diffuse = 10            
    SNR_mic = 30                
    
    dataset_title = 'test/concurrent_dynamic'
    start_idx = 1


def build_40s_source_signal(speaker_dir: str, fs: int, active_segments: list) -> np.ndarray:
    """
    Constructs a 40-second audio array, filling specified active time segments 
    with concatenated TIMIT speech, and the rest with silence.
    active_segments: list of tuples (start_time_sec, end_time_sec)
    """
    total_samples = int(40.0 * fs)
    full_signal = np.zeros(total_samples)
    
    for start_sec, end_sec in active_segments:
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        target_len = end_idx - start_idx
        
        # Load and concatenate speech until we fill the segment
        segment_audio = np.array([])
        while len(segment_audio) < target_len:
            speech = load_speech(get_random_speech_file(speaker_dir), fs)
            segment_audio = np.concatenate([segment_audio, speech])
            
        full_signal[start_idx:end_idx] = segment_audio[:target_len]
        
    return full_signal


def generate_composite_trajectory(
    fs: int, duration_sec: float, path_hop: int, center: np.ndarray, radius: float, 
    velocity: float, static_deg: float, dyn_start_deg: float, dyn_end_deg: float, 
    angle_classes: np.ndarray
):
    """Generates a trajectory that is static for the first half, and moving for the second half."""
    total_samples = int(duration_sec * fs)
    half_samples = total_samples // 2
    
    # 1. Static Portion (First 20 seconds)
    static_rad = np.deg2rad(static_deg)
    x_stat = center[0] + radius * np.cos(static_rad)
    y_stat = center[1] + radius * np.sin(static_rad)
    
    sp_path_static = np.zeros((3, half_samples))
    sp_path_static[0, :] = x_stat
    sp_path_static[1, :] = y_stat
    sp_path_static[2, :] = center[2]
    
    label_idx = np.argmin(np.abs(angle_classes - static_deg)) + 1
    labels_static = np.full(half_samples, label_idx, dtype=int)
    
    # 2. Dynamic Portion (Last 20 seconds)
    sp_path_dyn, labels_dyn = generate_source_path(
        len_source_signal=total_samples - half_samples,
        update_interval=path_hop,
        center=center,
        radius=radius,
        angle_classes=angle_classes,
        fs=fs,
        linear_velocity=velocity,
        mode='bounce',
        start_angle=np.deg2rad(dyn_start_deg),
        end_angle=np.deg2rad(dyn_end_deg)
    )
    
    # Concatenate
    sp_path = np.hstack((sp_path_static, sp_path_dyn))
    labels = np.concatenate((labels_static, labels_dyn))
    
    return sp_path, labels


def create_concurrent_sample(
    sample_idx: int,
    config: ConcurrentConfig,
    male_speakers: List[str],
    female_speakers: List[str],
    verbose: bool = True
) -> Dict:
    
    # --- 1. Environment & Geometry Setup ---
    L1 = 4.0 + 0.1 * np.random.randint(1, 21)
    L2 = 4.0 + 0.1 * np.random.randint(1, 21)
    room_dim = np.array([L1, L2, config.room_height])
    T60 = np.random.choice([0.2, 0.4, 0.6])
    
    center = np.array([room_dim[0] / 2, room_dim[1] / 2])
    center_3d = np.array([center[0], center[1], 1.0])
    mic_positions, _, _, _ = create_semicircular_mic_array(
        center, config.array_radius, height=1.0, 
        angle_resolution=360, selected_indices=[0, 54, 119, 179]
    )
    
    angle_classes = np.arange(0, 181, config.angle_resolution)
    
    # Pick independent speakers
    spk1_dir = male_speakers[np.random.randint(len(male_speakers))]
    spk2_dir = female_speakers[np.random.randint(len(female_speakers))]
    
    # --- 2. Construct Source Signals ---
    # Timeline strategy to satisfy RTF isolation requirement:
    # 00s-10s: Spk1 Alone (Static)
    # 10s-20s: Spk2 Alone (Static)
    # 20s-40s: Spk1 & Spk2 Concurrent (Moving)
    
    source_spk1 = build_40s_source_signal(spk1_dir, config.fs, [(0.0, 10.0), (20.0, 40.0)])
    source_spk2 = build_40s_source_signal(spk2_dir, config.fs, [(10.0, 20.0), (20.0, 40.0)])
    
    # --- 3. Construct Trajectories ---
    if verbose: print(f"  Generating paths & acoustics for 40s duration...")
    
    path_spk1, labels_spk1 = generate_composite_trajectory(
        config.fs, config.total_duration, config.path_hop, center_3d, config.speaker_radius,
        config.linear_velocity, config.spk1_start_deg, config.spk1_start_deg, config.spk1_end_deg, angle_classes
    )
    
    path_spk2, labels_spk2 = generate_composite_trajectory(
        config.fs, config.total_duration, config.path_hop, center_3d, config.speaker_radius,
        config.linear_velocity, config.spk2_start_deg, config.spk2_start_deg, config.spk2_end_deg, angle_classes
    )
    
    # --- 4. Acoustic Simulation (Superposition) ---
    # Simulate Speaker 1
    mic_signals_spk1 = generator.generate(
        source_spk1, c=config.c, fs=config.fs, rp_path=mic_positions,
        sp_path=path_spk1, L=room_dim, reverberation_time=T60, nRIR=1024,
        mtypes=generator.mic_type.omnidirectional, orientation=[0, 0]
    )
    
    # Simulate Speaker 2
    mic_signals_spk2 = generator.generate(
        source_spk2, c=config.c, fs=config.fs, rp_path=mic_positions,
        sp_path=path_spk2, L=room_dim, reverberation_time=T60, nRIR=1024,
        mtypes=generator.mic_type.omnidirectional, orientation=[0, 0]
    )
    
    # Combine signals at the microphones
    clean_mixture = mic_signals_spk1 + mic_signals_spk2
    clean_mixture = normalize_signal(clean_mixture)
    
    # --- 5. Noise Injection ---
    length_receives = clean_mixture.shape[0]
    A_x = np.mean(np.std(clean_mixture, axis=0))
    A_n_diffuse = A_x / (10 ** (config.SNR_diffuse / 20))
    A_n_mic = A_x / (10 ** (config.SNR_mic / 20))
    
    mic_noise = A_n_mic * np.random.randn(length_receives, config.M)
    
    try:
        diffuse_noise = fun_create_diffuse_noise(mic_positions[:, :2], fs=config.fs, L=length_receives)
        diffuse_noise = normalize_signal(diffuse_noise)
    except Exception as e:
        diffuse_noise = np.random.randn(length_receives, config.M)
        
    if len(diffuse_noise) < length_receives:
        diffuse_noise = np.tile(diffuse_noise, (int(np.ceil(length_receives / len(diffuse_noise))), 1))
    diffuse_noise = diffuse_noise[:length_receives, :config.M]
    
    noise_total = mic_noise + A_n_diffuse * diffuse_noise
    mixture = clean_mixture + noise_total
    
    # Normalize final outputs
    mixture = mixture / np.max(np.abs(mixture))
    mic_signals_spk1 = mic_signals_spk1 / (np.max(np.abs(mic_signals_spk1)) + 1e-8)
    mic_signals_spk2 = mic_signals_spk2 / (np.max(np.abs(mic_signals_spk2)) + 1e-8)
    
    # --- 6. Frame-level VAD Labels ---
    # Apply VAD logic to zero-out labels during silence periods
    # source_spk1 is silent from 10s-20s. source_spk2 is silent from 0-10s.
    frame_len = int(10.0 * config.fs)
    labels_spk1[frame_len:2*frame_len] = 0  # Zero out Spk1 label when silent
    labels_spk2[0:frame_len] = 0            # Zero out Spk2 label when silent
    
    vad_spk1 = create_vad_dynamic(labels_spk1, config.hop, config.nfft)
    vad_spk2 = create_vad_dynamic(labels_spk2, config.hop, config.nfft)
    
    return {
        'mixture': mixture,
        'spk1_clean': mic_signals_spk1,
        'spk2_clean': mic_signals_spk2,
        'vad_spk1': vad_spk1,
        'vad_spk2': vad_spk2,
        'room_dim': room_dim,
        'T60': T60,
        'mic_positions': mic_positions
    }


if __name__ == "__main__":
    config = ConcurrentConfig()
    np.random.seed(config.seed)
    
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_path = os.path.dirname(script_dir)
    timit_path = os.path.join(workspace_path, 'data', 'TIMIT')
    output_path = os.path.join(workspace_path, 'data', 'simulated_audio', config.dataset_title)
    os.makedirs(output_path, exist_ok=True)
    
    male_speakers, female_speakers = get_timit_speakers(timit_path)
    
    print(f"\nGenerating {config.num_samples} concurrent dynamic samples (40s each)...")
    print("-" * 60)
    
    for i in range(config.start_idx, config.start_idx + config.num_samples):
        print(f"\nSample {i}/{config.start_idx + config.num_samples - 1}:")
        
        result = create_concurrent_sample(i, config, male_speakers, female_speakers)
        
        sf.write(os.path.join(output_path, f'together_{i}.wav'), result['mixture'], config.fs)
        sf.write(os.path.join(output_path, f'first_{i}.wav'), result['spk1_clean'], config.fs)
        sf.write(os.path.join(output_path, f'second_{i}.wav'), result['spk2_clean'], config.fs)
        
        np.save(os.path.join(output_path, f'label_location_first_{i}.npy'), result['vad_spk1'])
        np.save(os.path.join(output_path, f'label_location_second_{i}.npy'), result['vad_spk2'])
        np.savez(
            os.path.join(output_path, f'metadata_{i}.npz'),
            room_dim=result['room_dim'], T60=result['T60'], mic_positions=result['mic_positions']
        )
        
    print(f"\nGeneration complete! Files saved to {output_path}")