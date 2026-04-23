from pipeline_ofer_funcs import *

py_folder = os.path.dirname(os.path.realpath(__file__))
workspace_folder = py_folder
folder_to_all_data = os.path.join(workspace_folder,'data')
folder_to_test_data = os.path.join(folder_to_all_data, 'simulated_audio', 'test')

num_wavs_for_test = 3

p_stft = {
    'n_fft': 2048,
    'hoplen': 512,
    'wlen': 2048,
    'NUP': 1025,
    'win': np.hanning(2048),
    'silent_frames': 30, # Number of initial frames to consider as noise-only for covariance estimation
}
def calc_n_frames(xlen, hop, wlen):
    '''
    Calculates the number of frames given the length of the signal, hop size, and window length.
    inputs:
        xlen (int): Length of the input signal in samples.
        hop (int): Hop size in samples.
        wlen (int): Window length in samples.
    outputs:
            n_frames (int): The total number of frames that will be generated from the input signal.
    formula_explanation:
    The formula 1 + (xlen - wlen) // hop is derived from the way frames are extracted from the signal:
    - The first frame starts at sample 0 and ends at sample wlen-1.
    - Each subsequent frame starts hop samples after the previous frame's start.
    '''
    return 1 + (xlen - wlen) // hop

M = 4 # number of microphones in the array

for k in range(1,num_wavs_for_test+1):

    ###### 1. Load and preprocess the data for the current experiment    #####
    inputData = load_and_preprocess_experiment(folder_to_test_data, k, mic_indices=[0, 1, 2, 3])
    mixed_audio = inputData['audio']['mixed']
    receiver_first = inputData['audio']['spk1']
    receiver_second = inputData['audio']['spk2']
    y2_first = inputData['angles']['spk1']
    y2_second = inputData['angles']['spk2']

    n_frames = calc_n_frames(inputData['audio']['mixed'].shape[0], p_stft['hoplen'], p_stft['wlen'])


    print(f"Experiment {k} loaded successfully with sample rate {inputData['fs']} Hz and audio shapes: "
          f"Mixed: {inputData['audio']['mixed'].shape}, "
          f"Speaker 1: {inputData['audio']['spk1'].shape}, "
          f"Speaker 2: {inputData['audio']['spk2'].shape}. "
          f"\n Angle data shapes: Speaker 1: {inputData['angles']['spk1'].shape}, Speaker 2: {inputData['angles']['spk2'].shape}")

    # Reshape to ensure they are strictly column vectors (L_frames, 1) 
    # to match the expectations of the downstream code.

    # Critical Sanity Check
    if len(y2_first) != n_frames:
        raise ValueError(f"Dimension mismatch! Audio yielded {n_frames} frames, but label array has {len(y2_first)} frames.")

    # -------------------------------------------------------------------------
    # 3. Time-Frequency Transformation and Noise Covariance Estimation
    # -------------------------------------------------------------------------

    # 1. Transform the main mixed recording into the STFT domain
    z_k = np.zeros((M, p_stft['NUP'], n_frames), dtype=complex)
    for i in range(M):
        z_k[i, :, :] = stft(mixed_audio[:, i], p_stft['win'], p_stft['hoplen'], p_stft['n_fft'])

    # 2. Transform clean references (needed later for Voice Activity Detection)
    z_k_first = np.zeros((M, p_stft['NUP'], n_frames), dtype=complex)
    for i in range(M):
        z_k_first[i, :, :] = stft(receiver_first[:, i], p_stft['win'], p_stft['hoplen'], p_stft['n_fft'])
        
    z_k_second = np.zeros((M, p_stft['NUP'], n_frames), dtype=complex)
    for i in range(M):
        z_k_second[i, :, :] = stft(receiver_second[:, i], p_stft['win'], p_stft['hoplen'], p_stft['n_fft'])

    # 3. Estimate Noise Spatial Covariance (Qvv) and its Cholesky factor
    cholesky_Qvv = np.zeros((p_stft['NUP'], M, M), dtype=complex)

    for j in range(p_stft['NUP']): 
        # Extract the first 'pad' frames for frequency bin 'j' across all mics
        noise_slice = z_k[:, j, 0:p_stft['silent_frames']] 
        
        # Calculate Cross-Power Spectral Density matrix
        PSD_tmp = noise_slice @ (noise_slice.conj().T) / p_stft['silent_frames']
        
        # Compute Cholesky factorization: Qvv = L * L^H
        cholesky_Qvv[j, :, :] = LA.cholesky(PSD_tmp)


        # Add last step, and maybe convert into class for better organization. But this is the core of the pipeline for loading, preprocessing, and initial transformations.
        # The next steps would involve implementing the Voice Activity Detection (VAD) using the clean references, computing the LCMV beamformer weights, applying the beamformer to the mixed signal, and then evaluating the results against the clean references.
        # Each of those steps can be modularized into separate functions or classes for better readability and maintainability.