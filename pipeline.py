from pipeline_ofer_funcs import *

print("-----------------STARTING LCMV PIPELINE-----------------")
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

class SpatialTrackingPipeline:
    def __init__(self, run_idx, p_stft, folder_to_test_data, M=4):
        """
        Initializes the pipeline configuration.
        """
        self.run_idx = run_idx
        self.p_stft = p_stft
        self.folder_to_test_data = folder_to_test_data
        self.M = M
        self.verbose = 1  # Set verbosity level for debugging and progress updates
        # 0 = no prints, 1 = basic progress, 2 = detailed info for debugging
        
        # Placeholders for data state
        self.input_data = None
        self.n_frames = 0
        
        # Placeholders for STFT and matrices
        self.z_k = None
        self.z_k_first = None
        self.z_k_second = None
        self.cholesky_Qvv = None

    def load_data(self):
        """Loads audio and angle data, calculates frames, and verifies dimensions."""
        if self.verbose:
            print(f"--- Loading data for Experiment {self.run_idx} ---")
        
        self.input_data = load_and_preprocess_experiment(
            self.folder_to_test_data, 
            self.run_idx, 
            mic_indices=list(range(self.M))
        )
        
        # Calculate expected frames
        audio_length = self.input_data['audio']['mixed'].shape[0]
        self.n_frames = calc_n_frames(audio_length, self.p_stft['hoplen'], self.p_stft['wlen'])
    
        # Extract and explicitly reshape angle data to column vectors (L_frames, 1)
        self.y2_first = self.input_data['angles']['spk1'].reshape(-1, 1)
        self.y2_second = self.input_data['angles']['spk2'].reshape(-1, 1)

        if self.verbose > 1:
            print(f"Loaded successfully. Sample rate: {self.input_data['fs']} Hz")
            print(f"Mixed audio shape: {self.input_data['audio']['mixed'].shape}")
        
        # Critical Sanity Check
        if len(self.y2_first) != self.n_frames:
            raise ValueError(f"Dimension mismatch! Audio yielded {self.n_frames} frames, "
                             f"but label array has {len(self.y2_first)} frames.")

    def compute_stft(self):
        """Transforms all necessary audio signals into the time-frequency domain."""
        if self.verbose:
            print("--- Computing STFTs ---")
        
        NUP = self.p_stft['NUP']
        win = self.p_stft['win']
        hoplen = self.p_stft['hoplen']
        n_fft = self.p_stft['n_fft']

        # 1. Main mixed recording
        self.z_k = np.zeros((self.M, NUP, self.n_frames), dtype=complex)
        for i in range(self.M):
            self.z_k[i, :, :] = stft(self.input_data['audio']['mixed'][:, i], win, hoplen, n_fft)

        # 2. Clean references (for later VAD)
        self.z_k_first = np.zeros((self.M, NUP, self.n_frames), dtype=complex)
        for i in range(self.M):
            self.z_k_first[i, :, :] = stft(self.input_data['audio']['spk1'][:, i], win, hoplen, n_fft)
            
        self.z_k_second = np.zeros((self.M, NUP, self.n_frames), dtype=complex)
        for i in range(self.M):
            self.z_k_second[i, :, :] = stft(self.input_data['audio']['spk2'][:, i], win, hoplen, n_fft)

    def estimate_noise_covariance(self):
        """Calculates Qvv and its Cholesky factorization from the silent frames."""
        if self.verbose:
            print("--- Estimating Noise Covariance (Qvv) ---")
        
        NUP = self.p_stft['NUP']
        silent_frames = self.p_stft['silent_frames']
        
        self.cholesky_Qvv = np.zeros((NUP, self.M, self.M), dtype=complex)

        for j in range(NUP): 
            # Extract the initial 'silent' frames for frequency bin 'j' across all mics
            noise_slice = self.z_k[:, j, 0:silent_frames] 
            
            # Calculate Cross-Power Spectral Density matrix
            PSD_tmp = noise_slice @ (noise_slice.conj().T) / silent_frames
            
            # Compute Cholesky factorization: Qvv = L * L^H
            self.cholesky_Qvv[j, :, :] = LA.cholesky(PSD_tmp)

    def run(self):
        """Orchestrates the pipeline steps in the correct order."""
        self.load_data()
        self.compute_stft()
        self.estimate_noise_covariance()
        
        # Future methods will be called here:
        # self.compute_vad()
        # self.extract_gevd_features()
        # self.run_inference()
        
        if self.verbose:
            print("--- Pipeline execution up to Covariance Estimation complete. ---")
        return self

# Add last step, and maybe convert into class for better organization. But this is the core of the pipeline for loading, preprocessing, and initial transformations.
# The next steps would involve implementing the Voice Activity Detection (VAD) using the clean references, computing the LCMV beamformer weights, applying the beamformer to the mixed signal, and then evaluating the results against the clean references.
# Each of those steps can be modularized into separate functions or classes for better readability and maintainability.    

for k in range(1,num_wavs_for_test+1):
    pipeline = SpatialTrackingPipeline(run_idx=k, p_stft=p_stft, folder_to_test_data=folder_to_test_data)
    pipeline.run()