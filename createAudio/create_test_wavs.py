from create_data_base import *


def create_test_sample_static(
    sample_idx: int,
    config: Config,
    male_speakers: List[str],
    female_speakers: List[str],
    ang_S1: float = None,
    ang_S2: float = None,
    verbose: bool = True
) -> dict:
    
    # --- Random room dimensions ---
    L1 = 4.0 + 0.1 * np.random.randint(1, 21)  # 4.1 to 6.0 m
    L2 = 4.0 + 0.1 * np.random.randint(1, 21)  # 4.1 to 6.0 m
    room_dim = np.array([L1, L2, config.room_height])
    
    # --- Random SNR and reverberation ---
    SNR_diffuse = 10 + np.random.randint(0, 11)  # 10 to 20 dB
    beta = 0.3 + 0.001 * np.random.randint(0, 251)  # 0.3 to 0.55 s (T60)
    
    # --- Generate speaker and mic positions ---
    pos_and_rir_time = time.time()
    simulator = AcousticTrajectorySimulator(room_dim.tolist(), config.R, config.noise_R, num_jumps=1)
    s_first, label_first, s_second, label_second, s_noise, mic_positions = simulator.generate()
    # dim(s_first) = (3, num_jumps), label_first = (num_jumps,), etc.

    
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
    
    # First speaker speaking
    # Then silence
    # Then second speaker speaking
    # Then silence
    # Then both speaking together

    speech_1_alone = load_speech(get_random_speech_file(speaker1_dir), config.fs)
    speech_2_alone = load_speech(get_random_speech_file(speaker2_dir), config.fs)
        
    # Another load for the together part (to ensure different content for each segment)
    speech_1_together = load_speech(get_random_speech_file(speaker1_dir), config.fs)
    speech_2_together = load_speech(get_random_speech_file(speaker2_dir), config.fs)
    
    silence_gap = np.zeros((int(config.fs * 1), config.M)) 

    # --- Process first speaker alone ---
    h_first = generate_rir(
        c=config.c,
        fs=config.fs,
        receiver_positions=mic_positions,
        source_position=s_first[:, 0],
        room_dim=room_dim,
        reverberation_time=beta,
        n_samples=config.n_rir_samples
    )
    rec_step1_first = convolve_with_rir(speech_1_alone, h_first)
    rec_step1_second = np.zeros_like(rec_step1_first)  # No second speaker
    label_step1_first = np.ones(len(rec_step1_first)) * label_first[0]
    label_step1_second = np.zeros(len(rec_step1_first))  # No second speaker

    # --- Process second speaker alone ---
    h_second = generate_rir(
        c=config.c,
        fs=config.fs,
        receiver_positions=mic_positions,
        source_position=s_second[:, 0],
        room_dim=room_dim,
        reverberation_time=beta,
        n_samples=config.n_rir_samples
    )
    rec_step2_second = convolve_with_rir(speech_2_alone, h_second)
    rec_step2_first = np.zeros_like(rec_step2_second)  # No first speaker
    label_step2_second = np.ones(len(rec_step2_second)) * label_second[0]
    label_step2_first = np.zeros(len(rec_step2_second))  # No first speaker

    # --- Process both speakers together ---
    rec_step3_first = convolve_with_rir(speech_1_together, h_first)
    rec_step3_second = convolve_with_rir(speech_2_together, h_second)
    label_step3_first = np.ones(len(rec_step3_first)) * label_first[0]
    label_step3_second = np.ones(len(rec_step3_second)) * label_second[0]

    # --- Concatenate segments with silence gaps ---
    # Order: first alone -> silence -> second alone -> silence -> both together
    # Each segment is a 2D array of shape (num_samples, num_channels), and we want to stack them vertically to create a longer time series for each channel.
    # Labels are 1D arrays.
    Receivers_first_total = np.concatenate([rec_step1_first, silence_gap, rec_step2_first, silence_gap, rec_step3_first], axis=0)
    Receivers_second_total = np.concatenate([rec_step1_second, silence_gap, rec_step2_second, silence_gap, rec_step3_second], axis=0)
    label_first_total = np.concatenate([label_step1_first, np.zeros(len(silence_gap)), label_step2_first, np.zeros(len(silence_gap)), label_step3_first])
    label_second_total = np.concatenate([label_step1_second, np.zeros(len(silence_gap)), label_step2_second, np.zeros(len(silence_gap)), label_step3_second]) 
    
    # pad to equal length in a single np command
    maxlen = max(len(Receivers_first_total), len(Receivers_second_total))
    if len(Receivers_first_total) < maxlen:
        pad_len = maxlen - len(Receivers_first_total)
        Receivers_first_total = np.vstack([Receivers_first_total, np.zeros((pad_len, config.M))])
        label_first_total = np.concatenate([label_first_total, np.zeros(pad_len)])
    if len(Receivers_second_total) < maxlen:
        pad_len = maxlen - len(Receivers_second_total)
        Receivers_second_total = np.vstack([Receivers_second_total, np.zeros((pad_len, config.M))])
        label_second_total = np.concatenate([label_second_total, np.zeros(pad_len)])

    
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
        'mic_positions': mic_positions,
        'true_geom_sector_first': label_first[0],
        'true_geom_sector_second': label_second[0]
    }

# Help me run this function to generate a test sample and save the resulting mixture and labels as WAV files for listening and verification.
class Config:
    """Configuration parameters for dataset generation.
    """

    seed = 500
    
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
    num_samples = 20
    start_idx = 1  # Starting index for file naming (e.g., 1 for 'first_1.wav')

    # File naming
    trainORval = 'test/static'  # 'train' or 'val'
    dataset_title = trainORval


if __name__ == "__main__":
    # --- Load speaker directories ---
    
    """
    Create the actual test samples.
    
    Parameters
    ----------

    config : Config
        Configuration object with all parameters
    """
    config = Config()
    np.random.seed(config.seed)

    dataset_title = config.dataset_title
    num_samples = config.num_samples
    start_idx = config.start_idx


    # Set default paths
    if config.timit_base_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_path = os.path.dirname(script_dir)
        timit_path = os.path.join(workspace_path, 'data', 'TIMIT')
    
    if config.output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_path = os.path.dirname(script_dir)
        data_path = os.path.join(workspace_path, 'data')
        sim_audio_path = os.path.join(data_path, 'simulated_audio')
        os.makedirs(sim_audio_path, exist_ok=True)
        output_path = os.path.join(sim_audio_path, dataset_title)
    
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
    config.timit_base_path = timit_path
    config.output_path = output_path
    
    print(f"\nGenerating {num_samples} samples...")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    for i in range(start_idx, start_idx + num_samples):
        print(f"\rProcessing sample {i}/{start_idx + num_samples - 1}...", end="", flush=True)
        
        # Generate sample
        result = create_test_sample_static(i, config, male_speakers, female_speakers)
        
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
        # Save metadata
        np.savez(
            os.path.join(output_path, f'metadata_{i}.npz'),
            room_dim=result['room_dim'],
            T60=result['T60'],
            SNR_diffuse=result['SNR_diffuse'],
            mic_positions=result['mic_positions'],
            true_sector_spk1=result['true_geom_sector_first'],
            true_sector_spk2=result['true_geom_sector_second']
        )
    
    print(f"\n\nDatabase generation complete!")
    print(f"Files saved to: {output_path}")
