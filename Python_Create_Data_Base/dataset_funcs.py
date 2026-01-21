import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import os
import glob
import soundfile as sf
from anf_generator import generate_signals
from anf_generator.CoherenceMatrix import Parameters 


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

def create_locations_18_dynamic(
    room_dim: list, 
    speaker_radius: float, 
    radius_noise: float, 
    num_jumps: int,
    plot_name: str = None,
):
    """
    Generates dynamic trajectories for two speakers and a static noise source in a simulated room.
    Trajectories = time series of 3D coordinates for each speaker over `num_jumps` time steps.

    The function ensures:
    1. Speakers move along a general circular path around the microphone array.
    2. Speaker positions are perturbed with random noise to avoid perfect circles.
    3. Speakers do not get too close to each other or cross paths abruptly.
    4. Angles of Arrival (AoA) are discretized into 18 labeled classes (intervals of 10 degrees).

    Parameters
    ----------
    room_dim : list
        Dimensions of the room [length, width, height] in meters.
    speaker_radius : float
        Base radius of the speakers from the array center (meters).
    radius_noise : float
        Maximum random deviation added to the speaker's position (meters).
        This is noise to the *POSITION*, not to the angle directly. 
    num_jumps : int
        Number of time steps (positions) to generate for the trajectory.

    Returns
    -------
    s_first : np.ndarray
        Trajectory of Speaker 1 (3 x num_jumps).
    label_first : np.ndarray
        Class labels (0-17) for Speaker 1's angle of arrival.
    s_second : np.ndarray
        Trajectory of Speaker 2 (3 x num_jumps).
    label_second : np.ndarray
        Class labels (0-17) for Speaker 2's angle of arrival.
    s_noise : np.ndarray
        Static position of the point-source noise [x, y, z].
    mic_array_coords : np.ndarray
        Coordinates of the 4 microphones [x, y, z].
    """

    # --- 1. Setup Simulation Environment ---
    room_x, room_y, room_z = room_dim
    mic_height = 1.0
    mic_radius = 0.1
    wall_margin = 0.5  # Minimum distance from walls
    
    # Calculate safe area for placing the array center
    # The center must be far enough from walls to accommodate the speaker radius + noise
    required_clearance = speaker_radius + wall_margin + radius_noise
    
    # Randomly place the center of the array within the safe zone
    # Formula: (max - min) * rand + min
    center_x = (room_x - 2 * required_clearance) * np.random.rand() + required_clearance
    center_y = (room_y - 2 * required_clearance) * np.random.rand() + required_clearance
    array_center = np.array([center_x, center_y])

    # --- 2. Create Microphone Array Geometry ---
    # Randomly rotate the entire setup (microphones + speakers)
    # This ensures the network learns rotation invariance
    angle_resolution = 360
    rotation_offset_idx = np.random.randint(1, angle_resolution + 1)
    
    # Create a full circle of points (for reference and potential positions)
    # We generate more points than needed (1.5 circles) to handle wrapping easily
    angles = np.linspace(-np.pi, 2 * np.pi, angle_resolution + angle_resolution // 2)
    
    # Slice the specific rotation window we want
    active_angles = angles[rotation_offset_idx - 1 : rotation_offset_idx - 1 + angle_resolution // 2]
    
    # Calculate microphone positions on the small circle (radius 0.1m)
    mic_x_all = mic_radius * np.sin(active_angles) + center_x
    mic_y_all = mic_radius * np.cos(active_angles) + center_y
    
    # Select 4 specific microphones to form the array geometry
    # Indices correspond to specific angles: ~0, ~54, ~119, ~179 degrees
    selected_mic_indices = [0, 54, 119, 179]
    mic_array_coords = np.array([
        [mic_x_all[i], mic_y_all[i], mic_height] for i in selected_mic_indices
    ])

    # Reference points for the speaker circle (radius R)
    speaker_ref_x = speaker_radius * np.sin(active_angles) + center_x
    speaker_ref_y = speaker_radius * np.cos(active_angles) + center_y

    # --- 3. Define Reference Vector for Angle Calculation ---
    # This vector represents "0 degrees" for the current rotation
    # It connects the first point of the circle to the center
    reference_vec = np.array([speaker_ref_x[0], speaker_ref_y[0]]) - array_center
    
    # Define the 18 possible angle classes (5, 15, ..., 175 degrees)
    angle_classes = np.arange(5, 176, 10)

    # --- 4. Generate Speaker Trajectories ---
    # Lists to store the valid path found
    traj_s1_x, traj_s1_y = np.zeros(num_jumps), np.zeros(num_jumps)
    traj_s2_x, traj_s2_y = np.zeros(num_jumps), np.zeros(num_jumps)
    labels_s1, labels_s2 = np.zeros(num_jumps, dtype=int), np.zeros(num_jumps, dtype=int)
    
    # Temporary storage for "lookback" logic to smooth trajectories
    prev_s1_x, prev_s1_y = np.zeros(num_jumps), np.zeros(num_jumps)
    prev_s2_x, prev_s2_y = np.zeros(num_jumps), np.zeros(num_jumps)
    
    step = 0
    history_labels = [] # Keep track to avoid immediate repetition if needed
    
    while step < num_jumps:
        is_position_invalid = True
        attempt_counter = 0
        
        while is_position_invalid:
            # Failsafe: If we get stuck finding a valid point, restart the whole trajectory
            if attempt_counter > 300:
                step = 0
                history_labels = []
                attempt_counter = 0
            attempt_counter += 1

            # --- Generate Candidate Position for Speaker 1 ---
            # Pick a random point on the reference circle
            idx1 = np.random.randint(0, angle_resolution // 2)
            # Add random noise to radius (perturb position)
            noise_angle = 0.01 * np.random.randint(1, 315)
            pos1_x = speaker_ref_x[idx1] + radius_noise * np.sin(noise_angle)
            pos1_y = speaker_ref_y[idx1] + radius_noise * np.cos(noise_angle)
            
            # Calculate Angle relative to reference vector
            vec1 = np.array([pos1_x, pos1_y]) - array_center
            angle_deg_1 = calculate_angle(reference_vec, vec1)
            # Find nearest class index (0-17)
            label1 = np.argmin(np.abs(angle_classes - angle_deg_1))

            # --- Generate Candidate Position for Speaker 2 ---
            idx2 = np.random.randint(0, angle_resolution // 2)
            noise_angle = 0.01 * np.random.randint(1, 315)
            pos2_x = speaker_ref_x[idx2] + radius_noise * np.sin(noise_angle)
            pos2_y = speaker_ref_y[idx2] + radius_noise * np.cos(noise_angle)
            
            vec2 = np.array([pos2_x, pos2_y]) - array_center
            angle_deg_2 = calculate_angle(reference_vec, vec2)
            label2 = np.argmin(np.abs(angle_classes - angle_deg_2))

            # --- Validation Checks ---
            
            # 1. Check if these specific angle classes were just used (Optional constraint from original code)
            if (label1 in history_labels) or (label2 in history_labels):
                is_position_invalid = True
                continue # Retry
            
            # 2. spatial Distance Check: Are speakers too close?
            dist_speakers = np.linalg.norm([pos1_x - pos2_x, pos1_y - pos2_y])
            if dist_speakers < 0.5:
                is_position_invalid = True
                continue

            # 3. Trajectory Consistency Check (only if not the first step)
            # Ensure speakers didn't swap places or cross too confusingly
            if step > 0:
                # Distance from S1(current) to S2(previous)
                cross_dist_1 = np.linalg.norm([pos1_x - prev_s2_x[step-1], pos1_y - prev_s2_y[step-1]])
                # Distance from S2(current) to S1(previous)
                cross_dist_2 = np.linalg.norm([pos2_x - prev_s1_x[step-1], pos2_y - prev_s1_y[step-1]])
                
                if cross_dist_1 < 0.5 or cross_dist_2 < 0.5:
                    is_position_invalid = True
                    continue
            
            # If we passed all checks, accept this step
            is_position_invalid = False

        # Save valid positions
        traj_s1_x[step], traj_s1_y[step] = pos1_x, pos1_y
        traj_s2_x[step], traj_s2_y[step] = pos2_x, pos2_y
        labels_s1[step], labels_s2[step] = label1, label2
        
        # Store for next step's comparison
        prev_s1_x[step], prev_s1_y[step] = pos1_x, pos1_y
        prev_s2_x[step], prev_s2_y[step] = pos2_x, pos2_y

        history_labels.append(label1)
        history_labels.append(label2)
        step += 1

    # Format Output Arrays
    s_first = np.array([traj_s1_x, traj_s1_y, np.ones(num_jumps) * mic_height])
    s_second = np.array([traj_s2_x, traj_s2_y, np.ones(num_jumps) * mic_height])

    # --- 5. Generate Point Source Noise Position ---
    # Find a noise position at least 2 meters away from the array center
    s_noise = np.array([center_x, center_y, mic_height])
    dist_noise = 0.0
    
    while dist_noise < 2.0:
        # Pick random point in room (respecting wall margins)
        nx = wall_margin + np.random.rand() * (room_x - 2 * wall_margin)
        ny = wall_margin + np.random.rand() * (room_y - 2 * wall_margin)
        s_noise = np.array([nx, ny, mic_height])
        dist_noise = np.linalg.norm(s_noise - np.array([center_x, center_y, mic_height]))

    if plot_name is not None:
        # --- 6. Plot All ---
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot speaker circle and mic circle
        ax.plot(speaker_ref_x, speaker_ref_y, np.ones(len(speaker_ref_x)), label='Speaker circle')
        ax.plot(mic_x_all, mic_y_all, np.ones(len(mic_x_all)), label='Mic circle')
        
        # Plot reference line (from first to last point of speaker circle)
        line_x, line_y = fillline(
            [speaker_ref_x[0], speaker_ref_y[0]], 
            [speaker_ref_x[-1], speaker_ref_y[-1]], 
            int(speaker_radius * 2 * 100)
        )
        ax.plot(line_x, line_y, np.ones(len(line_x)), label='Reference line')
        
        # Plot microphone positions
        mic_indices = [0, 24, 54, 79, 119, 149, 179]
        valid_mic_indices = [idx for idx in mic_indices if idx < len(mic_x_all)]
        ax.scatter(
            [mic_x_all[idx] for idx in valid_mic_indices],
            [mic_y_all[idx] for idx in valid_mic_indices],
            [1] * len(valid_mic_indices),
            marker='o', s=50, label='Microphones'
        )
        
        t_noise_location = np.linspace(0, 2 * np.pi, 100)
        
        for i in range(num_jumps):
            # Plot noise radius circles around speaker 2 positions (green)
            x_noise_location = radius_noise * np.sin(t_noise_location) + traj_s2_x[i]
            y_noise_location = radius_noise * np.cos(t_noise_location) + traj_s2_y[i]
            z_noise_location = np.ones_like(t_noise_location)
            ax.plot(x_noise_location, y_noise_location, z_noise_location, 'g-', alpha=0.5)
            
            # Plot noise radius circles around speaker 1 positions (blue)
            x_noise_location = radius_noise * np.sin(t_noise_location) + traj_s1_x[i]
            y_noise_location = radius_noise * np.cos(t_noise_location) + traj_s1_y[i]
            z_noise_location = np.ones_like(t_noise_location)
            ax.plot(x_noise_location, y_noise_location, z_noise_location, 'b-', alpha=0.5)
            
            # Plot speaker 1 (blue), speaker 2 (green), noise source (red)
            # Only add labels on first iteration to avoid duplicate legend entries
            ax.scatter([traj_s1_x[i]], [traj_s1_y[i]], [mic_height], marker='o', s=30, c='blue',
                    label='Speaker 1' if i == 0 else None)
            ax.scatter([traj_s2_x[i]], [traj_s2_y[i]], [mic_height], marker='o', s=30, c='green',
                    label='Speaker 2' if i == 0 else None)
            ax.scatter([s_noise[0]], [s_noise[1]], [mic_height], marker='o', s=30, c='red',
                    label='Noise source' if i == 0 else None)
        
        plotcube(room_dim, [0, 0, 0], 0, [1, 1, 1], ax)  # use function plotcube
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Room with Speakers and Microphone Array (Refactored)')
        ax.legend(loc='upper left')
        
        py_path = os.path.dirname(os.path.abspath(__file__))
        result_path = os.path.join(py_path, 'Results')
        os.makedirs(result_path, exist_ok=True)
        plt.savefig(os.path.join(result_path, f'{plot_name}.png'), dpi=150)

    return s_first, labels_s1, s_second, labels_s2, s_noise, mic_array_coords


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
        noise_folder = os.path.join(py_path, 'Diff_noise_srs')
    
    # --- Find and load noise file ---
    # Get list of all WAV files in the folder
    wav_files = glob.glob(os.path.join(noise_folder, '*.wav'))
    
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No WAV files found in {noise_folder}")
    
    # Pick a random noise file
    noise_file = wav_files[np.random.randint(len(wav_files))]
    
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
        raise ValueError(f'Audio file too short. Need {M * L} samples, got {len(data)}')
    
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
    
    # Number of frames (same formula as MATLAB: 1 + fix((xlen-wlen)/hop))
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


