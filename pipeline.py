from pipeline_ofer_funcs import *

print("-----------------STARTING LCMV PIPELINE-----------------")
py_folder = os.path.dirname(os.path.realpath(__file__))
workspace_folder = py_folder
folder_to_all_data = os.path.join(workspace_folder,'data')
folder_to_test_data = os.path.join(folder_to_all_data, 'simulated_audio', 'test')
class SpatialTrackingPipeline:
    def __init__(self, run_idx, p_stft, p_tracking, folder_to_test_data, M=4, verbose=1):
        """
        Initializes the pipeline configuration.
        """
        self.run_idx = run_idx
        self.p_stft = p_stft
        self.p_tracking = p_tracking
        self.folder_to_test_data = folder_to_test_data
        self.M = M
        self.verbose = verbose  # Set verbosity level for debugging and progress updates

        # Step 1: Instantiate scaler exactly as original script did globally
        self.scaler = StandardScaler()

        # Placeholders for data state
        self.input_data = None
        self.n_frames = 0
        
        # Placeholders for STFT and matrices
        self.z_k = None
        self.z_k_first = None
        self.z_k_second = None
        self.cholesky_Qvv = None
        self.X = None

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
        """Step 3: Estimates Qvv from an early noise-dominant segment and computes Cholesky factor."""
        if self.verbose:
            print("--- Estimating Noise Covariance (Cholesky Qvv) ---")
            
        NUP = self.p_stft['NUP']
        pad = self.p_stft['silent_frames']
        
        self.cholesky_Qvv = np.zeros((NUP, self.M, self.M), dtype=complex)
        
        for i in range(0, NUP):
            # Using z_k (mixed signal) with 'pad' frames as in original script
            PSD_tmp = self.z_k[:, i, 0:pad] @ (self.z_k[:, i, 0:pad].conj().T) / pad
            self.cholesky_Qvv[i, :, :] = LA.cholesky(PSD_tmp)

    def extract_gevd_features(self):
        """Step 4: Extracts GEVD-derived spatial features (RTF-like vectors)."""
        if self.verbose:
            print("--- Extracting GEVD Features ---")
            
        NUP = self.p_stft['NUP']
        frame_before = self.p_tracking['frame_before']
        frame_after = self.p_tracking['frame_after']
        win_vad = self.p_tracking['win_vad']
        
        total_frames = self.n_frames - frame_before - frame_after
        if total_frames <= 0:
            raise ValueError("Not enough frames for GEVD feature extraction.")

        # Original script's 'index' variable is equivalent to our 'self.n_frames'
        self.X = np.zeros((total_frames, NUP, self.M), dtype=complex)
        x_index = 0

        def _print_progress(current, total):
            if not self.verbose:
                return
            bar_len = 24
            filled = int(bar_len * current / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\rGEVD extraction: [{bar}] {current:>4}/{total} frames", end="", flush=True)
        
        # Strictly sequential loop to maintain exact original logic
        for l in range(frame_before, self.n_frames - frame_after):
            if self.verbose > 1:
                _print_progress(x_index, total_frames)

            for j in range(NUP):
                chol_j = LA.inv(self.cholesky_Qvv[j, :, :])
                Zvv = 0
                sum_win_vad = 0
                
                for p in range(frame_before + frame_after + 1):
                    # Note: win_vad index 10 is hardcoded just like the original script 
                    # (assuming win_vad is np.hamming(21) where 10 is the center)
                    temp_zvv = chol_j @ (win_vad[10 - frame_before + p] * self.z_k[:, j, l - frame_before + p].reshape(self.M, 1))
                    Zvv = Zvv + temp_zvv @ temp_zvv.conj().T
                    sum_win_vad = sum_win_vad + win_vad[10 - frame_before + p]
                    
                Zvv = Zvv / sum_win_vad

                w, v = LA.eig(Zvv)
                fi = v[:, w.argmax()].reshape(self.M, 1)

                denominator = self.cholesky_Qvv[j, 0, :].reshape(1, self.M) @ fi
                G_cw = np.squeeze(self.cholesky_Qvv[j, :, :]) @ fi / denominator
                self.X[x_index, j, :] = np.squeeze(G_cw)
                
            x_index += 1

        if self.verbose:
            _print_progress(total_frames, total_frames)
            print()

    def run(self):
        """Orchestrates the pipeline steps in the correct order."""
        self.load_data()
        self.compute_stft()
        self.estimate_noise_covariance()
        self.extract_gevd_features()
        
        # Future methods will be called here:
        # self.prepare_model_inputs()
        # self.compute_ground_truth_labels()
        # self.run_inference()
        # self.evaluate_and_save()
        
        if self.verbose:
            print("--- Pipeline execution up to GEVD feature extraction complete. ---")
        return self

# Setup dictionaries required to initialize the new pipeline architecture
p_stft = {
    'n_fft': 2048,
    'hoplen': 512,
    'wlen': 2048,
    'NUP': 1025,
    'win': np.hamming(2048),
    'silent_frames': 30, 
}

p_tracking = {
    'frame_before': 8,
    'frame_after': 5,
    'win_vad': np.hamming(21),
    'threshold': 40,
    'threshold_freq': 0.3
}

# Example instantiation
pipeline = SpatialTrackingPipeline(run_idx=1, p_stft=p_stft, p_tracking=p_tracking,
                                    folder_to_test_data=folder_to_test_data, M=4, verbose=2)
pipeline.run()