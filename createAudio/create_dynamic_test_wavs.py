from create_data_base import *
import scipy
from abc import ABC, abstractmethod
# I FUCKING WASTED MY TIME HEREE
class TrajectoryGenerator(ABC):
    """Base class for all acoustic speaker trajectories."""
    
    def __init__(self, config, room_dim: np.ndarray, s_R: float):
        self.config = config
        self.room_dim = room_dim
        self.s_R = s_R
        # Calculate center once during initialization
        self.center = np.array([room_dim[0] / 2, room_dim[1] / 2])

    def _calculate_num_updates(self, sentence_duration: float = 0, num_updates: int = 0) -> int:
        """Centralized logic for determining trajectory length."""
        if sentence_duration > 0:
            return int(np.ceil(sentence_duration / self.config.spatial_update_rate))
        elif num_updates > 0:
            return num_updates
        else:
            raise ValueError("Either sentence_duration or num_updates must be provided.")

    @abstractmethod
    def generate(self, **kwargs) -> np.ndarray:
        """Must be implemented by subclasses to generate the 3D coordinates."""
        pass




class CircularTrajectory(TrajectoryGenerator):
    def generate(self, start_angle: float, end_angle: float, 
                 sentence_duration: float = 0, num_updates: int = 0, height: float = 1.5) -> np.ndarray:
        
        num_updates = self._calculate_num_updates(sentence_duration, num_updates)
        angles = np.linspace(start_angle, end_angle, num_updates)
        
        trajectory = np.zeros((3, num_updates))
        trajectory[0, :] = self.center[0] + self.s_R * np.cos(np.radians(angles))
        trajectory[1, :] = self.center[1] + self.s_R * np.sin(np.radians(angles))
        trajectory[2, :] = height
        
        return trajectory


class StaticTrajectory(TrajectoryGenerator):
    def generate(self, angle: float, 
                 sentence_duration: float = 0, num_updates: int = 0, height: float = 1.0) -> np.ndarray:
        
        num_updates = self._calculate_num_updates(sentence_duration, num_updates)
        
        trajectory = np.zeros((3, num_updates))
        trajectory[0, :] = self.center[0] + self.s_R * np.cos(np.radians(angle))
        trajectory[1, :] = self.center[1] + self.s_R * np.sin(np.radians(angle))
        trajectory[2, :] = height
        
        return trajectory


class StopWaitGoTrajectory(TrajectoryGenerator):
    def generate(self, start_angle: float, pause_angle: float, end_angle: float, 
                 vel: float, sentence_duration: float, height: float = 1.5) -> tuple:
        
        target_total_updates = self._calculate_num_updates(sentence_duration=sentence_duration)
        
        # Calculate distances and times
        distance_move1 = np.radians(abs(pause_angle - start_angle)) * self.s_R
        distance_move2 = np.radians(abs(end_angle - pause_angle)) * self.s_R
        
        time_move1 = distance_move1 / vel
        time_move2 = distance_move2 / vel
        
        num_updates_move1 = self._calculate_num_updates(sentence_duration=time_move1)
        num_updates_move2 = self._calculate_num_updates(sentence_duration=time_move2)
        num_updates_pause = target_total_updates - num_updates_move1 - num_updates_move2
        
        if num_updates_pause < 0:
            raise ValueError("Sentence duration is too short for the given velocities and angle distances.")

        # Re-use the sibling classes to generate the segments
        circular_gen = CircularTrajectory(self.config, self.room_dim, self.s_R)
        static_gen = StaticTrajectory(self.config, self.room_dim, self.s_R)

        traj_part1 = circular_gen.generate(start_angle, pause_angle, num_updates=num_updates_move1, height=height)
        traj_part2 = static_gen.generate(pause_angle, num_updates=num_updates_pause, height=height)
        traj_part3 = circular_gen.generate(pause_angle, end_angle, num_updates=num_updates_move2, height=height)

        # Stitch them together
        full_trajectory = np.concatenate([traj_part1, traj_part2, traj_part3], axis=1)

        return full_trajectory
    
def plot_trajectory(trajectory: np.ndarray, mic_positions: np.ndarray, room_dim: np.ndarray, config: Config, name: str = 'trajectory_plot.png'):
    plt.figure(figsize=(6, 6))
    plt.plot(trajectory[0, :], trajectory[1, :], label='Speaker Trajectory', marker='o')
    plt.scatter(mic_positions[:, 0], mic_positions[:, 1], label='Microphones', color='red', marker='x')

    # making the color of the trajectory change over time to indicate the direction of movement:
    points = np.array([trajectory[0, :], trajectory[1, :]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, len(trajectory[0, :])))
    lc.set_array(np.arange(len(trajectory[0, :])))
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)
    # Add a colorbar to indicate time progression
    cbar = plt.colorbar(lc, pad=0.01)
    cbar.set_label('Time Step')

    plt.xlim(0, room_dim[0])
    plt.ylim(0, room_dim[1])
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Speaker Trajectory and Microphone Positions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.plot_path, name))




def dynamic_convolve(dry_signal: np.ndarray, trajectory: np.ndarray, mic_positions: np.ndarray,
                      config: Config) -> np.ndarray:
    """
    Applies dynamic RIRs to a moving source using Overlap-Add.
    """
    frame_length = int(config.fs * config.frame_size_sec)
    hop_length = frame_length // 2  # 50% overlap is standard
    
    # Initialize output array (needs to be longer than input to accommodate RIR tail)
    out_length = len(dry_signal) + config.n_rir_samples - 1
    output_signal = np.zeros((out_length, config.M)) # M is num microphones
    
    window = np.hanning(frame_length)
    
    for i, start_idx in enumerate(range(0, len(dry_signal) - frame_length, hop_length)):
        end_idx = start_idx + frame_length
        
        # 1. Extract and window the dry frame
        frame = dry_signal[start_idx:end_idx] * window
        
        # 2. Get the current spatial coordinate
        current_pos = trajectory[i]
        
        # 3. Generate the instantaneous RIR
        h_inst = rir.generate(
            c=config.c,
            fs=config.fs,
            r=mic_positions.tolist(),
            s=current_pos,
            L=config.room_dim,
            beta=config.T60,
            ns=config.n_rir_samples
        )
        
        # 4. Convolve frame with instantaneous RIR (using FFT convolution for speed)
        # h_inst shape: (n_rir_samples, M)
        convolved_frame = scipy.signal.fftconvolve(frame[:, None], h_inst, axes=0)
        
        # 5. Overlap-Add into the master output array
        conv_len = len(convolved_frame)
        output_signal[start_idx:start_idx + conv_len, :] += convolved_frame
        
    return output_signal


def create_test_sample_dynamic(
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
    trajectory_gen = CircularTrajectory(config, room_dim, config.R)
    trajectory = trajectory_gen.generate(
        start_angle=config.start_angle,
        end_angle=config.end_angle,
        sentence_duration=10.0,  # 10 seconds duration for the test sample
        height=1.5
    )
    mic_positions, *_ = create_semicircular_mic_array(center=np.array([room_dim[0] / 2, room_dim[1] / 2]), 
                                                      radius=0.1, height=1.5, angle_resolution=360, selected_indices=[0, 54, 119, 179])

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
        'mic_positions': mic_positions
    }

class Config:
    """Configuration parameters for dataset generation.
    """

    seed = 31
    
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

    plot_path = None            # Will be set at runtime
    
    # Number of samples to generate
    num_samples = 3
    start_idx = 1  # Starting index for file naming (e.g., 1 for 'first_1.wav')

    # File naming
    trainORval = 'test/static'  # 'train' or 'val'
    dataset_title = trainORval

    # DYNAMIC-> 
    start_angle = 10
    pause_angle = 120
    end_angle = 50
    spatial_update_rate = 50e-3 # seconds (=50 ms)
    

    # -> Linear
    vel = 2
    


if __name__ == "__main__":
    config = Config()
    np.random.seed(config.seed)

    
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
        output_path = os.path.join(sim_audio_path, config.dataset_title)

    if config.plot_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_path = os.path.dirname(script_dir)
        plot_path = os.path.join(workspace_path, 'plots')
        plot_path = os.path.join(plot_path, 'test_wavs', 'dynamic')
        os.makedirs(plot_path, exist_ok=True)
        config.plot_path = plot_path

    
    # --- Random room dimensions ---
    L1 = 4.0 + 0.1 * np.random.randint(1, 21)  # 4.1 to 6.0 m
    L2 = 4.0 + 0.1 * np.random.randint(1, 21)  # 4.1 to 6.0 m
    room_dim = np.array([L1, L2, config.room_height])
    
    # --- Random SNR and reverberation ---
    SNR_diffuse = 10 + np.random.randint(0, 11)  # 10 to 20 dB
    beta = 0.3 + 0.001 * np.random.randint(0, 251)  # 0.3 to 0.55 s (T60)

#def create_semicircular_mic_array(center, radius, height, angle_resolution=360, selected_indices=[0, 54, 119, 179]):

    mic_positions, *_ = create_semicircular_mic_array(center=np.array([room_dim[0] / 2, room_dim[1] / 2]), radius=0.1, height=1.5, angle_resolution=360, selected_indices=[0, 54, 119, 179])

    s_R = config.R + config.noise_R * (np.random.rand() - 0.5)  # Add some noise to the radius for variability
    print("Generating circular trajectory for test sample...")
    trajectory_gen = CircularTrajectory(config, room_dim, s_R)
    trajectory = trajectory_gen.generate(
        start_angle=config.start_angle,
        end_angle=config.end_angle,
        sentence_duration=5.0,  # 5 seconds duration for the test sample
        height=1.5
    )
    plot_trajectory(trajectory, mic_positions, room_dim, config, name='circular_trajectory.png')

    trajectory_gen = StopWaitGoTrajectory(config, room_dim, s_R)
    trajectory = trajectory_gen.generate(
            start_angle=config.start_angle,
            pause_angle=config.pause_angle,
            end_angle=config.end_angle,
            vel=config.vel,
            sentence_duration=10.0,  # 10 seconds duration for the test sample
            height=1.5
        )
    plot_trajectory(trajectory, mic_positions, room_dim, config, name='stop_wait_go_trajectory.png')
