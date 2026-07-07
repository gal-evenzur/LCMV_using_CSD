"""
Generates a dataset of 40-second simulated dynamic speakers.
- Phase 1 (0s-10s): S1 static at angle_s1_start (S2 silent).
- Phase 2 (10s-20s): S2 static at angle_s2_start (S1 silent).
- Phase 3 (20s-40s): Both moving concurrently using 'bounce' mode.
"""

from logging import config
import os
import time
import argparse
from typing import List, Dict
import numpy as np
import soundfile as sf
import das_generator as generator
import matplotlib.pyplot as plt

# Imports from your existing modules
from create_data_base import (
    get_timit_speakers,
    get_random_speech_file,
    load_speech,
    normalize_signal,
    generate_rir,
    convolve_with_rir
)
from dataset_funcs import (
    AcousticTrajectorySimulator,
    fun_create_diffuse_noise,
    create_vad_dynamic
)

class Config:
    """Configuration parameters for the 40-second paper replication."""
    seed = 42
    
    # Acoustic parameters
    c = 340                     
    fs = 16000                  
    n_rir_samples = 4096        
    room_height = 3.0           
    
    # Analysis parameters
    nfft = 2048                 
    hop = 512                   
    path_hop = 1024               
    M = 4                       
    
    # Speaker parameters
    radius_s1 = 1.3
    radius_s2 = 1.3
    linear_velocity = 0.333     # 1 meter every 3 seconds
    
    angle_s1_start = 0.0
    angle_s1_end = 140.0
    angle_s2_start = 160.0
    angle_s2_end = 180.0
    
    # Timeline blocks (in seconds)
    t_phase1 = 10.0             # S1 static, S2 silent
    t_phase2 = 10.0             # S2 static, S1 silent
    t_phase3 = 20.0             # Concurrent dynamic
    t_total = t_phase1 + t_phase2 + t_phase3
    initial_noise_pad_sec = 1.0   # Silence before the first speaker starts
    
    # SNR parameters
    SNR_direction = 20
    SNR_mic = 30
    SNR_diffuse = 15 # will be randomized per sample          
    
    # Output configurations
    num_samples = 10
    start_idx = 1
    dataset_title = 'test/paperlike'
    timit_base_path = None
    output_path = None
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')


class PaperReplicationSimulator(AcousticTrajectorySimulator):
    """
    Subclass that encapsulates the rigid 40-second timeline requirements
    for this specific paper replication, leaving the base math untouched.
    """
    def generate_40s_scenario(self, fs: int, path_hop: int, config: Config):
        # 1. Setup the physical environment
        self._place_array_center()
        self._calculate_geometry()
        center_3d = np.array([self.array_center[0], self.array_center[1], self.mic_height])
        s_noise = self._generate_noise_source()
        
        static_samples = int((config.t_phase1 + config.t_phase2) * fs)
        dynamic_samples = int(config.t_phase3 * fs)
        
        # 2. Generate Speaker 1 Trajectory
        s1_stat_path, s1_stat_labels = self._generate_source_path(
            len_source_signal=static_samples, update_interval=path_hop, center=center_3d,
            radius=config.radius_s1, angle_classes=self.angle_classes, fs=fs,
            linear_velocity=0.0, start_angle=np.deg2rad(config.angle_s1_start), end_angle=np.deg2rad(config.angle_s1_start), mode='stop'
        )
        s1_dyn_path, s1_dyn_labels = self._generate_source_path(
            len_source_signal=dynamic_samples, update_interval=path_hop, center=center_3d,
            radius=config.radius_s1, angle_classes=self.angle_classes, fs=fs,
            linear_velocity=config.linear_velocity, start_angle=np.deg2rad(config.angle_s1_start), end_angle=np.deg2rad(config.angle_s1_end), mode='bounce'
        )
        s1_path = np.hstack([s1_stat_path, s1_dyn_path])
        s1_labels = np.concatenate([s1_stat_labels, s1_dyn_labels])

        # 3. Generate Speaker 2 Trajectory
        s2_stat_path, s2_stat_labels = self._generate_source_path(
            len_source_signal=static_samples, update_interval=path_hop, center=center_3d,
            radius=config.radius_s2, angle_classes=self.angle_classes, fs=fs,
            linear_velocity=0.0, start_angle=np.deg2rad(config.angle_s2_start), end_angle=np.deg2rad(config.angle_s2_start), mode='stop'
        )
        s2_dyn_path, s2_dyn_labels = self._generate_source_path(
            len_source_signal=dynamic_samples, update_interval=path_hop, center=center_3d,
            radius=config.radius_s2, angle_classes=self.angle_classes, fs=fs,
            linear_velocity=config.linear_velocity, start_angle=np.deg2rad(config.angle_s2_start), end_angle=np.deg2rad(config.angle_s2_end), mode='bounce'
        )
        s2_path = np.hstack([s2_stat_path, s2_dyn_path])
        s2_labels = np.concatenate([s2_stat_labels, s2_dyn_labels])

            
        # 5. Plot labels if requested
        if self.plot_name is not None:
            # Plot labels over time for both speakers
            plt.figure(figsize=(12, 6))
            # label times are based on the update intervals, so we can create a time axis for better visualization
            
            time_axis_s1 = np.arange(len(s1_labels)) / fs
            time_axis_s2 = np.arange(len(s2_labels)) / fs
            plt.plot(time_axis_s1, s1_labels, label='Speaker 1 Labels', color='blue')
            plt.plot(time_axis_s2, s2_labels, label='Speaker 2 Labels', color='green')
            plt.xlabel('Time (samples)')
            plt.ylabel('Angle Class Label')
            plt.title('Angle Class Labels Over Time')
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_name + '_labels.png')

        return s1_path, s1_labels, s2_path, s2_labels, s_noise, self.mic_array_coords


def accumulate_speech_with_gaps(speaker_dir: str, target_samples: int, fs: int) -> np.ndarray:
    """Concatenates TIMIT sentences with organic silence gaps (0.3s - 0.8s)."""
    accumulated_audio = []
    current_length = 0
    
    while current_length < target_samples:
        wav_path = get_random_speech_file(speaker_dir)
        speech = load_speech(wav_path, fs)
        
        gap_samples = np.random.randint(int(0.3 * fs), int(0.8 * fs))
        gap = np.zeros(gap_samples)
        
        accumulated_audio.extend([speech, gap])
        current_length += len(speech) + len(gap)
        
    full_audio = np.concatenate(accumulated_audio)
    return full_audio[:target_samples]


def create_paper_test_sample(
    sample_idx: int,
    config: Config,
    male_speakers: List[str],
    female_speakers: List[str],
    verbose: bool = True
) -> Dict:
    
    fs = config.fs
    total_samples = int(config.t_total * fs)
    phase1_samp = int(config.t_phase1 * fs)
    phase2_samp = int(config.t_phase2 * fs)
    phase3_samp = int(config.t_phase3 * fs)
    
    # --- 1. Environment ---
    L1 = 4.0 + 0.1 * np.random.randint(1, 21)
    L2 = 4.0 + 0.1 * np.random.randint(1, 21)
    room_dim = np.array([L1, L2, config.room_height])
    
    SNR_diffuse = config.SNR_diffuse + np.random.randint(0, 11)  # Randomized per sample
    T60 = 0.3 + 0.001 * np.random.randint(0, 251)  # T60

    total_SNR = -10 * np.log10((10 ** (-SNR_diffuse / 10)) + (10 ** (-config.SNR_direction / 10)) + (10 ** (-config.SNR_mic / 10)))
    
    plot_name = os.path.join(config.plot_dir, f'paperlike_{sample_idx}_labels_SNR_{total_SNR:.1f}_T60_{T60:.3f}.png')
    print(f"\n--- Generating Sample {sample_idx} --- with plot: {plot_name}")
    # --- 2. Generate Geometry & Trajectories ---
    # We pass the max radius to the base constructor just for safe array clearance calculations
    max_radius = max(config.radius_s1, config.radius_s2)
    simulator = PaperReplicationSimulator(
        room_dim=room_dim.tolist(), speaker_radius=max_radius, 
        radius_noise=0.2, num_jumps=0, plot_name=plot_name
    )
    
    s1_path, s1_labels, s2_path, s2_labels, s_noise, mic_positions = simulator.generate_40s_scenario(
        fs, config.path_hop, config
    )

    # --- 3. Assemble and Mask Audio Timelines ---
    speaker1_dir = male_speakers[np.random.randint(len(male_speakers))] if np.random.rand() < 0.5 else female_speakers[np.random.randint(len(female_speakers))]
    speaker2_dir = male_speakers[np.random.randint(len(male_speakers))] if np.random.rand() < 0.5 else female_speakers[np.random.randint(len(female_speakers))]

    # Get exactly 30s of active speech for each (10s static + 20s dynamic)
    speech_s1_30s = accumulate_speech_with_gaps(speaker1_dir, phase1_samp + phase3_samp, fs)
    speech_s2_30s = accumulate_speech_with_gaps(speaker2_dir, phase2_samp + phase3_samp, fs)
    
    # Mask S1 (silent from 10s-20s)
    audio_s1_40s = np.concatenate([
        speech_s1_30s[:phase1_samp], 
        np.zeros(phase2_samp), 
        speech_s1_30s[phase1_samp:]
    ])
    s1_labels[phase1_samp : phase1_samp + phase2_samp] = 0
    
    # Mask S2 (silent from 0s-10s)
    audio_s2_40s = np.concatenate([
        np.zeros(phase1_samp),
        speech_s2_30s[:phase2_samp],
        speech_s2_30s[phase2_samp:]
    ])
    s2_labels[:phase1_samp] = 0

    # --- 4. Acoustic Simulation (das_generator) ---
    if verbose: print(f"\n  Running S1 das_generator simulation...", end="", flush=True)
    Receivers_first_total = generator.generate(
        audio_s1_40s, c=config.c, fs=fs, rp_path=mic_positions,
        sp_path=s1_path, L=room_dim, reverberation_time=T60,
        nRIR=config.n_rir_samples, mtypes=generator.mic_type.omnidirectional, orientation=[0,0]
    )
    
    if verbose: print(f"\n  Running S2 das_generator simulation...", end="", flush=True)
    Receivers_second_total = generator.generate(
        audio_s2_40s, c=config.c, fs=fs, rp_path=mic_positions,
        sp_path=s2_path, L=room_dim, reverberation_time=T60,
        nRIR=config.n_rir_samples, mtypes=generator.mic_type.omnidirectional, orientation=[0,0]
    )
    
    Receivers_first_total = normalize_signal(Receivers_first_total)
    Receivers_second_total = normalize_signal(Receivers_second_total)
    receivers = Receivers_first_total + Receivers_second_total

    # --- 5. Correct SNR Calculation (Active Phase Only) ---
    # Calculate signal power strictly during Phase 3 where both speakers are active
    active_start_idx = phase1_samp + phase2_samp
    active_receivers = receivers[active_start_idx:, :]
    A_x = np.mean(np.std(active_receivers, axis=0))
    
    A_n_diffuse = A_x / (10 ** (SNR_diffuse / 20))
    A_n_direction = A_x / (10 ** (config.SNR_direction / 20))
    A_n_mic = A_x / (10 ** (config.SNR_mic / 20))


    # Inject Initial Silence to ALL audio signals and labels
    pad_samples = int(config.initial_noise_pad_sec * config.fs)
    pad_matrix = np.zeros((pad_samples, config.M))
    
    # Pad the main mixture and isolated speaker arrays
    receivers = np.vstack((pad_matrix, receivers))
    Receivers_first_total = np.vstack((pad_matrix, Receivers_first_total))
    Receivers_second_total = np.vstack((pad_matrix, Receivers_second_total))
    
    # Pad the labels (0 represents silence/no active angle)
    s1_labels = np.concatenate((np.zeros(pad_samples), s1_labels))
    s2_labels = np.concatenate((np.zeros(pad_samples), s2_labels))    
    total_samples = receivers.shape[0]

    # --- 6. Noise Generation ---
    # Point-source directional noise (Full 40s)

    noise_temp = np.random.randn(total_samples + config.n_rir_samples)
    h_noise = generate_rir(
        c=config.c, fs=fs, receiver_positions=mic_positions,
        source_position=s_noise, room_dim=room_dim, 
        reverberation_time=T60, n_samples=config.n_rir_samples
    )
    Receivers_noise = convolve_with_rir(noise_temp, h_noise)[:total_samples, :]
    Receivers_noise = normalize_signal(Receivers_noise)

    # Mic noise
    mic_noise = A_n_mic * np.random.randn(total_samples, config.M)
    
    # Diffuse noise
    try:
        diffuse_noise = fun_create_diffuse_noise(
            mic_positions=mic_positions[:, :2], fs=fs, L=total_samples
        )
        diffuse_noise = normalize_signal(diffuse_noise)
    except Exception as e:
        if verbose: print(f"\n  Warning: Diffuse noise failed ({e}), using Gaussian.")
        diffuse_noise = np.random.randn(total_samples, config.M)
        
    if len(diffuse_noise) < total_samples:
        diffuse_noise = np.tile(diffuse_noise, (int(np.ceil(total_samples / len(diffuse_noise))), 1))
    diffuse_noise = diffuse_noise[:total_samples, :config.M]

    # Combine all noises exactly as in create_test_wavs
    noise_total = mic_noise + (A_n_diffuse * diffuse_noise) + (A_n_direction * Receivers_noise)
    mixture = receivers + noise_total

    # Global normalizations
    noise_total = noise_total / np.max(np.abs(noise_total))
    mixture = mixture / np.max(np.abs(mixture))
    Receivers_first_total = Receivers_first_total / np.max(np.abs(Receivers_first_total))
    Receivers_second_total = Receivers_second_total / np.max(np.abs(Receivers_second_total))

    # --- 7. VAD Labels ---
    vad_first = create_vad_dynamic(s1_labels, config.hop, config.nfft)
    vad_second = create_vad_dynamic(s2_labels, config.hop, config.nfft)

    return {
        'mixture': mixture,
        'first_speaker': Receivers_first_total,
        'second_speaker': Receivers_second_total,
        'noise': noise_total,
        'vad_first': vad_first,
        'vad_second': vad_second,
        'room_dim': room_dim,
        'T60': T60,
        'SNR_diffuse': SNR_diffuse,
        'mic_positions': mic_positions
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 40s Paper Replication Dataset.")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of files to generate")
    parser.add_argument("--start_idx", type=int, default=1, help="Starting file index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Exposing the paper parameters
    parser.add_argument("--radius_s1", type=float, default=1.3, help="Radius for S1 (m)")
    parser.add_argument("--radius_s2", type=float, default=1.3, help="Radius for S2 (m)")
    parser.add_argument("--angle_s1_start", type=float, default=0.0, help="S1 starting/static angle")
    parser.add_argument("--angle_s1_end", type=float, default=140.0, help="S1 movement target angle")
    parser.add_argument("--angle_s2_start", type=float, default=160.0, help="S2 starting/static angle")
    parser.add_argument("--angle_s2_end", type=float, default=180.0, help="S2 movement target angle")
    
    args = parser.parse_args()
    
    config = Config()
    config.num_samples = args.num_samples
    config.start_idx = args.start_idx
    config.seed = args.seed
    config.radius_s1 = args.radius_s1
    config.radius_s2 = args.radius_s2
    config.angle_s1_start = args.angle_s1_start
    config.angle_s1_end = args.angle_s1_end
    config.angle_s2_start = args.angle_s2_start
    config.angle_s2_end = args.angle_s2_end
    
    np.random.seed(config.seed)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_path = os.path.dirname(script_dir)
    
    timit_path = os.path.join(workspace_path, 'data', 'TIMIT')
    output_path = os.path.join(workspace_path, 'data', 'simulated_audio', config.dataset_title)
    config.timit_base_path = timit_path
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    
    male_speakers, female_speakers = get_timit_speakers(timit_path)
    
    print(f"\nGenerating {config.num_samples} paper replication samples...")
    print("-" * 60)
    
    for i in range(config.start_idx, config.start_idx + config.num_samples):
        print(f"\rProcessing sample {i}/{config.start_idx + config.num_samples - 1}...", end="", flush=True)
        
        np.random.seed(config.seed + i)
        
        result = create_paper_test_sample(i, config, male_speakers, female_speakers, verbose=True)
        
        sf.write(os.path.join(output_path, f'first_{i}.wav'), result['first_speaker'], config.fs)
        sf.write(os.path.join(output_path, f'second_{i}.wav'), result['second_speaker'], config.fs)
        sf.write(os.path.join(output_path, f'together_{i}.wav'), result['mixture'], config.fs)
        
        np.save(os.path.join(output_path, f'label_location_first_{i}.npy'), result['vad_first'])
        np.save(os.path.join(output_path, f'label_location_second_{i}.npy'), result['vad_second'])
        
        np.savez(
            os.path.join(output_path, f'metadata_{i}.npz'),
            room_dim=result['room_dim'],
            T60=result['T60'],
            SNR_diffuse=result['SNR_diffuse'],
            mic_positions=result['mic_positions']
        )
    
    print(f"\n\nGeneration complete! Files saved to: {output_path}")