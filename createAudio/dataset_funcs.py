import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import os
import glob
import soundfile as sf
from anf_generator import generate_signals
from anf_generator.CoherenceMatrix import Parameters 
import time
from typing import Optional


def fillline(startp, endp, pts):
    """
    Take starting, ending point & number of points in between.
    Make line to connect between 2 coordinates.
    
    Parameters
    ----------
    startp : array-like
        Starting point [x, y]
    endp : array-like
        Ending point [x, y]
    pts : int
        Number of points in the line
    
    Returns
    -------
    xx : numpy.ndarray
        X coordinates of the line
    yy : numpy.ndarray
        Y coordinates of the line
    """
    # Calculate gradient
    m = (endp[1] - startp[1]) / (endp[0] - startp[0]) if (endp[0] - startp[0]) != 0 else np.inf

    if m == np.inf:  # vertical line
        xx = np.full(pts, startp[0])
        yy = np.linspace(startp[1], endp[1], pts)
    elif m == 0:  # horizontal line
        xx = np.linspace(startp[0], endp[0], pts)
        yy = np.full(pts, startp[1])
    else:  # regular line
        xx = np.linspace(startp[0], endp[0], pts)
        yy = m * (xx - startp[0]) + startp[1]

    return xx, yy


def plotcube(edges=None, origin=None, alpha=None, color=None, ax=None):
    """
    PLOTCUBE - Display a 3D-cube in the current axes
    
    Parameters
    ----------
    edges : array-like, optional
        3-elements vector that defines the length of cube edges.
        Default: [10, 56, 100]
    origin : array-like, optional
        3-elements vector that defines the start point of the cube.
        Default: [10, 10, 10]
    alpha : float, optional
        Scalar that defines the transparency of the cube faces (from 0 to 1).
        Default: 0.7
    color : array-like, optional
        3-elements vector that defines the faces color of the cube.
        Default: [1, 0, 0] (red)
    ax : matplotlib 3D axes, optional
        The axes to plot on. If None, uses current axes.
    
    Returns
    -------
    ax : matplotlib 3D axes
        The axes with the plotted cube
    
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> plotcube([5, 5, 5], [2, 2, 2], 0.8, [1, 0, 0], ax)
    >>> plotcube([5, 5, 5], [10, 10, 10], 0.8, [0, 1, 0], ax)
    >>> plotcube([5, 5, 5], [20, 20, 20], 0.8, [0, 0, 1], ax)
    >>> plt.show()
    """
    
    # Default input arguments
    if edges is None:
        edges = [10, 56, 100]
    if origin is None:
        origin = [10, 10, 10]
    if alpha is None:
        alpha = 0.7
    if color is None:
        color = [1, 0, 0]
    
    edges = np.array(edges)
    origin = np.array(origin)
    color = np.array(color)
    
    # Get or create axes
    if ax is None:
        ax = plt.gca()
    
    # Define the 6 faces of the cube
    # Each face is defined by 4 vertices (x, y, z coordinates)
    XYZ = [
        # Face 1 (x = 0)
        [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]],
        # Face 2 (x = 1)
        [[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]],
        # Face 3 (y = 0)
        [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]],
        # Face 4 (y = 1)
        [[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]],
        # Face 5 (z = 0)
        [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
        # Face 6 (z = 1)
        [[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]],
    ]
    
    # Scale and translate each face
    faces = []
    for face in XYZ:
        # Scale by edges and translate by origin
        x = np.array(face[0]) * edges[0] + origin[0]
        y = np.array(face[1]) * edges[1] + origin[1]
        z = np.array(face[2]) * edges[2] + origin[2]
        # Create vertices for this face (4 corners)
        verts = list(zip(x, y, z))
        faces.append(verts)
    
    # Create 3D polygon collection
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='k', linewidth=0.5)
    ax.add_collection3d(poly3d)
    
    return ax


def calculate_angle(v1, v2):
    """Helper to calculate angle in degrees between two vectors."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    # Clip to handle floating point errors slightly outside [-1, 1]
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))


def create_semicircular_mic_array(center, radius, height, angle_resolution=360, selected_indices=[0, 54, 119, 179]):
    """
    Creates a semi-circular microphone array with random rotation.
    
    Parameters
    ----------
    center : array-like
        [x, y] coordinates of the array center.
    radius : float
        Radius of the microphone circle.
    height : float
        Z-coordinate (height) of the microphones.
    angle_resolution : int
        Number of points in a full circle.
    selected_indices : list
        Indices of the microphones to pick from the 180-degree arc.
        
    Returns
    -------
    mic_array_coords : np.ndarray
        M x 3 array of microphone coordinates.
    active_angles : np.ndarray
        The angles used for the semi-circular arc.
    mic_x_all : np.ndarray
        X coordinates for all points in the arc.
    mic_y_all : np.ndarray
        Y coordinates for all points in the arc.
    """
    center_x, center_y = center
    rotation_offset_idx = np.random.randint(1, angle_resolution + 1)
    
    angles = np.linspace(-np.pi, 2 * np.pi, angle_resolution + (angle_resolution // 2))
    active_angles = angles[rotation_offset_idx - 1 : rotation_offset_idx - 1 + (angle_resolution // 2)]
    
    mic_x_all = radius * np.sin(active_angles) + center_x
    mic_y_all = radius * np.cos(active_angles) + center_y
    
    mic_array_coords = np.array([
        [mic_x_all[i], mic_y_all[i], height] for i in selected_indices if i < len(mic_x_all)
    ])
    
    return mic_array_coords, active_angles, mic_x_all, mic_y_all


class AcousticTrajectorySimulator:
    """
    Generates dynamic trajectories for two speakers and a static noise source in a simulated room.
    """
    
    def __init__(self, room_dim: list, speaker_radius: float, radius_noise: float, num_jumps: int, plot_name: str = None):
        # Input parameters
        self.room_dim = room_dim
        self.room_x, self.room_y, self.room_z = room_dim
        self.speaker_radius = speaker_radius
        self.radius_noise = radius_noise
        self.num_jumps = num_jumps
        self.plot_name = plot_name
        
        # Environmental Constants
        self.mic_height = 1.0
        self.mic_radius = 0.1
        self.wall_margin = 0.5
        self.angle_resolution = 360
        
        # State variables populated during generation
        self.array_center = None
        self.mic_array_coords = None
        self.speaker_ref_x = None
        self.speaker_ref_y = None
        self.reference_vec = None
        self.angle_classes = None

    def generate(self):
        """Orchestrator method to run the simulation and return the 6 specific outputs."""
        self._place_array_center()
        self._calculate_geometry()
        
        s_first, labels_s1, s_second, labels_s2 = self._generate_trajectories()
        s_noise = self._generate_noise_source()
        
        if self.plot_name is not None:
            self._plot_simulation(s_first, s_second, s_noise)
            
        return s_first, labels_s1, s_second, labels_s2, s_noise, self.mic_array_coords

    def _place_array_center(self):
        """Calculates safe clearance and randomly places the array center."""
        required_clearance = self.speaker_radius + self.wall_margin + self.radius_noise
        center_x = (self.room_x - 2 * required_clearance) * np.random.rand() + required_clearance
        center_y = (self.room_y - 2 * required_clearance) * np.random.rand() + required_clearance
        self.array_center = np.array([center_x, center_y])

    def _calculate_geometry(self):
        """Sets up microphone positions, rotation, and speaker reference angles."""
        selected_mic_indices = [0, 54, 119, 179]
        self.mic_array_coords, active_angles, self.mic_x_all, self.mic_y_all = create_semicircular_mic_array(
            self.array_center, self.mic_radius, self.mic_height, 
            self.angle_resolution, selected_mic_indices
        )

        # Speakers
        center_x, center_y = self.array_center
        self.speaker_ref_x = self.speaker_radius * np.sin(active_angles) + center_x
        self.speaker_ref_y = self.speaker_radius * np.cos(active_angles) + center_y
        
        # Reference vector for 0 degrees
        self.reference_vec = np.array([self.speaker_ref_x[0], self.speaker_ref_y[0]]) - self.array_center
        self.angle_classes = np.arange(5, 176, 10)

    def _generate_candidate_position(self, class_offset=0):
        """Picks a random perturbed position for a speaker and calculates its AoA class."""
        idx = np.random.randint(0, self.angle_resolution // 2)
        noise_angle = 0.01 * np.random.randint(1, 315)
        
        pos_x = self.speaker_ref_x[idx] + self.radius_noise * np.sin(noise_angle)
        pos_y = self.speaker_ref_y[idx] + self.radius_noise * np.cos(noise_angle)
        
        vec = np.array([pos_x, pos_y]) - self.array_center
        angle_deg = calculate_angle(self.reference_vec, vec) # Assumes calculate_angle is in scope
        label = np.argmin(np.abs(self.angle_classes - angle_deg)) + class_offset
        
        return pos_x, pos_y, label

    def _validate_step(self, pos1, pos2, prev1, prev2, label1, label2, history_labels, step):
        """Checks business logic: distance between speakers, path crossings, and label history."""
        if (label1 in history_labels) or (label2 in history_labels):
            return False
            
        dist_speakers = np.linalg.norm(np.array(pos1) - np.array(pos2))
        if dist_speakers < 0.5:
            return False
            
        if step > 0:
            cross_dist_1 = np.linalg.norm(np.array(pos1) - np.array(prev2))
            cross_dist_2 = np.linalg.norm(np.array(pos2) - np.array(prev1))
            if cross_dist_1 < 0.5 or cross_dist_2 < 0.5:
                return False
                
        return True

    def _generate_trajectories(self):
        """Main loop that generates valid sequences for both speakers."""
        traj_s1_x, traj_s1_y = np.zeros(self.num_jumps), np.zeros(self.num_jumps)
        traj_s2_x, traj_s2_y = np.zeros(self.num_jumps), np.zeros(self.num_jumps)
        labels_s1, labels_s2 = np.zeros(self.num_jumps, dtype=int), np.zeros(self.num_jumps, dtype=int)
        
        step = 0
        history_labels = []
        
        while step < self.num_jumps:
            is_valid = False
            attempt_counter = 0
            
            while not is_valid:
                if attempt_counter > 300:
                    step, attempt_counter, history_labels = 0, 0, []
                attempt_counter += 1

                p1_x, p1_y, l1 = self._generate_candidate_position(class_offset=0)
                p2_x, p2_y, l2 = self._generate_candidate_position(class_offset=1)
                
                pos1 = (p1_x, p1_y)
                pos2 = (p2_x, p2_y)
                prev1 = (traj_s1_x[step-1], traj_s1_y[step-1]) if step > 0 else (0, 0)
                prev2 = (traj_s2_x[step-1], traj_s2_y[step-1]) if step > 0 else (0, 0)

                is_valid = self._validate_step(pos1, pos2, prev1, prev2, l1, l2, history_labels, step)

            # Accept candidate
            traj_s1_x[step], traj_s1_y[step] = pos1
            traj_s2_x[step], traj_s2_y[step] = pos2
            labels_s1[step], labels_s2[step] = l1, l2
            
            history_labels.extend([l1, l2])
            step += 1

        s_first = np.array([traj_s1_x, traj_s1_y, np.ones(self.num_jumps) * self.mic_height])
        s_second = np.array([traj_s2_x, traj_s2_y, np.ones(self.num_jumps) * self.mic_height])
        
        return s_first, labels_s1, s_second, labels_s2

    def _generate_noise_source(self):
        """Finds a noise position at least 2 meters away from the array center."""
        dist_noise = 0.0
        s_noise = np.zeros(3)
        
        while dist_noise < 2.0:
            nx = self.wall_margin + np.random.rand() * (self.room_x - 2 * self.wall_margin)
            ny = self.wall_margin + np.random.rand() * (self.room_y - 2 * self.wall_margin)
            s_noise = np.array([nx, ny, self.mic_height])
            dist_noise = np.linalg.norm(s_noise - np.array([self.array_center[0], self.array_center[1], self.mic_height]))
            
        return s_noise

    def _plot_simulation(self, s_first, s_second, s_noise):
        """Handles all matplotlib visualization logic."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Basic geometries
        ax.plot(self.speaker_ref_x, self.speaker_ref_y, np.ones(len(self.speaker_ref_x)), label='Speaker circle')
        ax.plot(self.mic_x_all, self.mic_y_all, np.ones(len(self.mic_x_all)), label='Mic circle')
        
        # Assumes fillline and plotcube are in scope
        line_x, line_y = fillline(
            [self.speaker_ref_x[0], self.speaker_ref_y[0]], 
            [self.speaker_ref_x[-1], self.speaker_ref_y[-1]], 
            int(self.speaker_radius * 2 * 100)
        )
        ax.plot(line_x, line_y, np.ones(len(line_x)), label='Reference line')
        
        # Microphones
        mic_indices = [0, 24, 54, 79, 119, 149, 179]
        valid_mic_indices = [idx for idx in mic_indices if idx < len(self.mic_x_all)]
        ax.scatter(
            [self.mic_x_all[idx] for idx in valid_mic_indices],
            [self.mic_y_all[idx] for idx in valid_mic_indices],
            [1] * len(valid_mic_indices),
            marker='o', s=50, label='Microphones'
        )
        
        # Plot jump data
        t_noise = np.linspace(0, 2 * np.pi, 100)
        traj_s1_x, traj_s1_y = s_first[0], s_first[1]
        traj_s2_x, traj_s2_y = s_second[0], s_second[1]
        
        for i in range(self.num_jumps):
            for tx, ty, color in [(traj_s1_x[i], traj_s1_y[i], 'b-'), (traj_s2_x[i], traj_s2_y[i], 'g-')]:
                nx = self.radius_noise * np.sin(t_noise) + tx
                ny = self.radius_noise * np.cos(t_noise) + ty
                ax.plot(nx, ny, np.ones_like(t_noise), color, alpha=0.5)
                
            ax.scatter([traj_s1_x[i]], [traj_s1_y[i]], [self.mic_height], marker='o', s=30, c='blue', label='Speaker 1' if i == 0 else None)
            ax.scatter([traj_s2_x[i]], [traj_s2_y[i]], [self.mic_height], marker='o', s=30, c='green', label='Speaker 2' if i == 0 else None)
            ax.scatter([s_noise[0]], [s_noise[1]], [self.mic_height], marker='o', s=30, c='red', label='Noise source' if i == 0 else None)
            
        plotcube(self.room_dim, [0, 0, 0], 0, [1, 1, 1], ax)
        
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Room with Speakers and Microphone Array')
        ax.legend(loc='upper left')
        
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
        os.makedirs(result_path, exist_ok=True)
        plt.savefig(os.path.join(result_path, f'{self.plot_name}.png'), dpi=150)
def fun_create_diffuse_noise(
    mic_positions: np.ndarray,
    noise_folder: str = None,
    fs: int = 16000,
    c: float = 340.0,
    K: int = 256,
    L: int = None,
    type_nf: str = 'spherical'
) -> np.ndarray:
    """
    Generate diffuse noise with desired spatial coherence for microphone array.
    
    This function creates spatially coherent noise signals that simulate
    a diffuse noise field (e.g., cafe ambience, babble noise) as would be
    captured by a microphone array.
    
    Parameters
    ----------
    mic_positions : np.ndarray
        Microphone positions as M x 2 or M x 3 array, where M is the number
        of microphones. Each row is [x, y] or [x, y, z] coordinates in meters.
        If 2D (M x 2), z-coordinates will be set to 0.
    noise_folder : str, optional
        Path to folder containing noise WAV files. If None, uses default path
        relative to this file: 'Diff_noise_srs/'
    fs : int, optional
        Sample frequency in Hz. Default: 16000
    c : float, optional
        Sound velocity in m/s. Default: 340.0
    K : int, optional
        FFT length for coherence matrix computation. Default: 256
    L : int, optional
        Desired output length in samples. Default: 20 * fs (20 seconds)
    type_nf : str, optional
        Type of noise field: 'spherical' or 'cylindrical'. Default: 'spherical'
    
    Returns
    -------
    noise : np.ndarray
        Generated noise signals with spatial coherence, shape (L, M)
        where L is the number of samples and M is the number of microphones.
    
    Raises
    ------
    ValueError
        If sample frequency of input file doesn't match fs.
    FileNotFoundError
        If no WAV files are found in the noise folder.
    
    Example
    -------
    >>> # Define microphone positions (4 mics in semicircular array, radius 0.1m)
    >>> t = np.linspace(0, np.pi, 180)
    >>> circ_mics_x = 0.1 * np.sin(t)
    >>> circ_mics_y = 0.1 * np.cos(t)
    >>> mic_positions = np.array([
    ...     [circ_mics_x[0], circ_mics_y[0]],
    ...     [circ_mics_x[54], circ_mics_y[54]],
    ...     [circ_mics_x[119], circ_mics_y[119]],
    ...     [circ_mics_x[179], circ_mics_y[179]]
    ... ])
    >>> noise = fun_create_diffuse_noise(mic_positions)
    """
    # Set default values
    if L is None:
        L = 20 * fs  # 20 seconds of data
    
    M = mic_positions.shape[0]  # Number of microphones
    
    # Ensure mic_positions is M x 3 (add z=0 if only 2D)
    if mic_positions.shape[1] == 2:
        mic_positions_3d = np.column_stack([mic_positions, np.zeros(M)])
    else:
        mic_positions_3d = mic_positions
    
    # Set default noise folder path
    if noise_folder is None:
        py_path = os.path.dirname(os.path.abspath(__file__))
        workspace_path = os.path.dirname(py_path)
        data_path = os.path.join(workspace_path, 'data')
        noise_folder = os.path.join(data_path, 'Diff_noise_srs')
    
    # --- Find and load noise file ---
    # Get list of all WAV files in the folder
    wav_files = glob.glob(os.path.join(noise_folder, '*.wav'))
    
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No WAV files found in {noise_folder}")
    
    # Pick a random noise file
    noise_file = wav_files[np.random.randint(len(wav_files))]
    # Print the chosen file for debugging
    print(f"Selected noise file: {noise_file}")
    
    # Read audio file
    data, fs_data = sf.read(noise_file)
    
    # Convert to float and normalize if integer format
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128) / 128.0
    else:
        data = data.astype(np.float64)
    
    # Handle stereo files - convert to mono
    if len(data.shape) > 1:
        data = data[:, 0]  # Take first channel
    
    # Check sample frequency
    if fs != fs_data:
        raise ValueError(f'Sample frequency of input file ({fs_data} Hz) does not match required ({fs} Hz).')
    
    # --- Generate M mutually 'independent' babble speech input signals ---
    # Select random starting point ensuring we have enough data for all channels
    if len(data) < M * L:
        print(f'Audio file too short. Need {M * L} samples, got {len(data)}')
        # Loop the data if too short
        repeats = int(np.ceil((M * L) / len(data)))
        data = np.tile(data, repeats)
        print(f'Looped audio to length {len(data)} samples.')
    
    start_signal = np.random.randint(0, len(data) - M * L)
    
    # Remove DC offset
    data = data - np.mean(data)
    data = data[start_signal:]
    
    # Extract M non-overlapping segments as "independent" noise sources
    babble = np.zeros((M, L))  # Shape: [M, L] for anf-generator (channels x samples)
    for m in range(M):
        babble[m, :] = data[m * L : (m + 1) * L]
    
    # --- Setup parameters for anf-generator ---
    params = Parameters(
        mic_positions=mic_positions_3d,  # M x 3 array
        sc_type=type_nf,                 # 'spherical' or 'cylindrical'
        sample_frequency=fs,
        nfft=K,
        c=c
    )
    
    # --- Generate sensor signals with desired spatial coherence ---
    # The generate_signals function applies the coherence matrix to create
    # spatially coherent noise from independent input signals
    # decomposition='chd' is Cholesky (same as 'cholesky' in MATLAB)
    noise, _, _ =  generate_signals(
        input_signals=babble,      # Shape: [M, L]
        params=params,
        decomposition='chd',       # Cholesky decomposition
        processing='standard'
    )
    
    # Transpose to match MATLAB output format: (L, M)
    noise = noise.T
    
    return noise


def create_vad_dynamic(x: np.ndarray, hop: int, nfft: int) -> np.ndarray:
    """
    Create frame-level labels from sample-level labels using majority voting.
    
    This function downsamples sample-level labels to STFT frame-level labels
    by finding the most frequent value (mode) in each frame window. It is used
    to convert sample-level speaker direction labels to frame-level labels
    that align with STFT frames.
    
    Parameters
    ----------
    x : np.ndarray
        Sample-level label signal (1D array). Values are typically:
        - 0: silence/padding (no speech)
        - 1-17: angle class labels indicating speaker direction
    hop : int
        Hop size (number of samples between consecutive frames).
        Typically matches STFT hop size.
    nfft : int
        Window length (number of samples per frame).
        Typically matches STFT window/FFT length.
    
    Returns
    -------
    vad : np.ndarray
        Frame-level labels (1D array of length L), where L is the number
        of frames. Each value is the most frequent label in that frame.
    
    Notes
    -----
    The function uses the statistical mode (most frequent value) to determine
    the label for each frame. This effectively performs majority voting:
    - If a frame is mostly zeros → returns 0 (no speech)
    - If a frame contains mostly label k → returns k (speech from direction k)
    
    Example
    -------
    >>> # Sample-level labels: 1 second at 16kHz, first half silence, second half speech from direction 5
    >>> labels = np.concatenate([np.zeros(8000), np.ones(8000) * 5])
    >>> hop = 512
    >>> nfft = 2048
    >>> frame_labels = create_vad_dynamic(labels, hop, nfft)
    """
    from scipy.stats import mode
    
    x = np.asarray(x).flatten()
    xlen = len(x)
    wlen = nfft
    
    ''' L = Number of frames in the stft. Explanation:
    The first frame: The very first frame consumes exactly wlen samples of the signal.
    The remaining samples (xlen - wlen): After placing the first frame,
    this is how much of the signal is left for the window to slide across. '''
    L = 1 + (xlen - wlen) // hop

    
    vad = np.zeros(L)
    
    for l in range(L):
        # Extract window: MATLAB uses 1-based indexing, Python uses 0-based
        # MATLAB: x(1+l*hop : wlen+l*hop) -> Python: x[l*hop : l*hop + wlen]
        start_idx = l * hop
        end_idx = start_idx + wlen
        x_w = x[start_idx:end_idx]
        
        # Find mode (most frequent value) in the window
        # scipy.stats.mode returns ModeResult(mode=array, count=array)
        mode_result = mode(x_w, keepdims=False)
        vad[l] = mode_result.mode
    
    return vad


def generate_source_path(
    movement_type: str,
    len_source_signal: int,
    hop: int,
    source_position: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    start_angle: float = 0.0,
    end_angle: float = 2 * np.pi,
) -> np.ndarray:
    """
    Generate source path.

    Parameters:
    -----------
    movement_type : str
        Type of movement ('line', 'circle', 'arc', or 'semi_circle')
    len_source_signal : int
        Length of the source signal
    hop : int
        Number of samples between position updates
    source_position : np.ndarray, optional
        Initial source position (Required for 'line' movement)
    center : np.ndarray, optional
        Center position for movements (Required for all movements)
    radius : float, optional
        Radius of the arc (Required for 'circle'/'arc' movements)
    start_angle : float, optional
        Absolute starting angle in radians (Default: 0.0)
    end_angle : float, optional
        Absolute ending angle in radians (Default: 2*pi)

    Returns:
    --------
    sp_path : np.ndarray
        Source path array of shape (3, len_source_signal)
    """

    # Initialize source path array
    sp_path = np.zeros((3, len_source_signal))
    movement = movement_type.lower()

    # --- Phase 1 & 2: Initialization and Validation ---
    if movement == "line":
        if source_position is None or center is None:
            raise ValueError("For 'line' movement, 'source_position' and 'center' must be provided.")
        start_x, start_y, start_z = source_position
        stop_x, stop_y = center[0] - 1.0, center[1] - 1.0
        
    elif movement in ["circle", "arc", "semi_circle"]:
        if center is None or radius is None:
            raise ValueError("For circle/arc movements, 'center' and 'radius' must be provided.")
        # Calculate the total angular sweep
        total_sweep = end_angle - start_angle
        
    else:
        raise ValueError(f"Unsupported movement type: {movement_type}")

    # --- Phase 3 & 4: Path Generation Loop ---
    for ii in range(0, len_source_signal, hop):
        # Calculate the progress fraction (0.0 to almost 1.0)
        progress = ii / len_source_signal

        if movement == "line":
            # Calculate new source position (linear interpolation)
            x_tmp = start_x + (progress * (stop_x - start_x))
            y_tmp = start_y + (progress * (stop_y - start_y))
            z_tmp = start_z  # Assuming flat line movement on Z

            sp_new = np.array([x_tmp, y_tmp, z_tmp])

        elif movement in ["circle", "arc", "semi_circle"]:
            # Interpolate the current angle based on progress
            current_angle = start_angle + (progress * total_sweep)

            # Construct the coordinates directly in global space
            x_tmp = center[0] + radius * np.cos(current_angle)
            y_tmp = center[1] + radius * np.sin(current_angle)
            z_tmp = center[2]  # Rests perfectly flat at the center's Z-height

            sp_new = np.array([x_tmp, y_tmp, z_tmp])

        # Store source path
        end_idx = min(ii + hop, len_source_signal)
        sp_path[:, ii:end_idx] = sp_new[:, np.newaxis]

    return sp_path


