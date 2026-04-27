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
        """Step 4: Extracts GEVD-derived spatial features (RTF-like vectors).
        Optimized using NumPy broadcasting across the frequency axis.
        """
        if self.verbose:
            print("--- Extracting GEVD Features (Vectorized) ---")
            
        NUP = self.p_stft['NUP']
        frame_before = self.p_tracking['frame_before']
        frame_after = self.p_tracking['frame_after']
        win_vad = self.p_tracking['win_vad']
        
        # Pre-compute the inverse of the Cholesky matrices for all frequency bins at once
        # Shape: (NUP, M, M)
        chol_inv_batched = LA.inv(self.cholesky_Qvv)
        
        self.X = np.zeros((self.n_frames - frame_before - frame_after, NUP, self.M), dtype=complex)
        x_index = 0
        
        # We must keep the temporal 'l' loop serial for online pipeline compatibility
        for l in range(frame_before, self.n_frames - frame_after):
            # Progress update every 50 frames
            if self.verbose > 1 and l % 50 == 0:
                print(f"Processing frame {l}/{self.n_frames - frame_before - frame_after}...")
            
            # Initialize a batched Zvv tensor for all frequency bins
            # Shape: (NUP, M, M)
            Zvv = np.zeros((NUP, self.M, self.M), dtype=complex)
            sum_win_vad = 0
            
            # Aggregate local context frames
            for p in range(frame_before + frame_after + 1):
                win_val = win_vad[10 - frame_before + p]
                
                # Slice the time frame. Original shape (M, NUP). 
                z_slice = self.z_k[:, :, l - frame_before + p]
                
                # Transpose and add a dummy axis so shape becomes (NUP, M, 1)
                # This aligns it for batched matrix multiplication
                z_slice_batched = z_slice.T[:, :, np.newaxis]
                
                # Multiply batched inverse with batched vectors: (NUP, M, M) @ (NUP, M, 1)
                # temp_zvv shape: (NUP, M, 1)
                temp_zvv = chol_inv_batched @ (win_val * z_slice_batched)
                
                # Outer product across the M dimension for all NUP matrices: (NUP, M, 1) @ (NUP, 1, M)
                # Resulting shape: (NUP, M, M)
                Zvv += temp_zvv @ temp_zvv.conj().transpose(0, 2, 1)
                sum_win_vad += win_val
                
            Zvv /= sum_win_vad

            # Vectorized Eigendecomposition on all NUP matrices simultaneously
            # w shape: (NUP, M) -> Eigenvalues
            # v shape: (NUP, M, M) -> Eigenvectors
            w, v = LA.eig(Zvv)
            
            # Find the index of the maximum eigenvalue for each frequency bin
            # We use .real because eigenvalues of a PSD matrix are strictly real
            max_idx = np.argmax(w.real, axis=1)
            
            # Extract the dominant eigenvectors for all NUP matrices
            # Fancy indexing v[np.arange(NUP), :, max_idx] yields shape (NUP, M)
            # Add an axis to make it (NUP, M, 1) for the final multiplication
            fi = v[np.arange(NUP), :, max_idx][:, :, np.newaxis]

            # Recolor and normalize (numerator shape: NUP, M, 1)
            numerator = self.cholesky_Qvv @ fi
            
            # Denominator: Isolate the first row (reference mic) of cholesky_Qvv
            # self.cholesky_Qvv[:, 0, :] shape: (NUP, M). Add middle axis to make it (NUP, 1, M)
            # Denominator final shape: (NUP, 1, 1)
            denominator = self.cholesky_Qvv[:, 0, :][:, np.newaxis, :] @ fi
            
            # Divide, squeeze out the trailing dummy dimension, and assign to feature matrix
            G_cw = np.squeeze(numerator / denominator)
            self.X[x_index, :, :] = G_cw
            
            x_index += 1
    
    def verify_vectorization(self, max_frames=50):
        """
        Runs both the original sequential and the new vectorized GEVD 
        implementations and asserts that the resulting RTFs are identical.
        max_frames limits the number of frames to compare for faster verification during development.
        """
        print("--- Starting GEVD Verification ---")
        
        NUP = self.p_stft['NUP']
        frame_before = self.p_tracking['frame_before']
        frame_after = self.p_tracking['frame_after']
        win_vad = self.p_tracking['win_vad']
        
        # ---------------------------------------------------------
        # 1. RUN ORIGINAL SEQUENTIAL LOGIC
        # ---------------------------------------------------------
        print("Running sequential extraction...")
        
        X_sequential = np.zeros((max_frames, NUP, self.M), dtype=complex)
        x_index = 0
        
        for l in range(frame_before, min(frame_before + max_frames, self.n_frames - frame_after)):
            for j in range(NUP):
                chol_j = LA.inv(self.cholesky_Qvv[j, :, :])
                Zvv = 0
                sum_win_vad = 0
                for p in range(frame_before + frame_after + 1):
                    temp_zvv = chol_j @ (win_vad[10 - frame_before + p] * self.z_k[:, j, l - frame_before + p].reshape(self.M, 1))
                    Zvv = Zvv + temp_zvv @ temp_zvv.conj().T
                    sum_win_vad = sum_win_vad + win_vad[10 - frame_before + p]
                Zvv = Zvv / sum_win_vad

                w, v = LA.eig(Zvv)
                fi = v[:, w.argmax()].reshape(self.M, 1)

                denominator = self.cholesky_Qvv[j, 0, :].reshape(1, self.M) @ fi
                G_cw = np.squeeze(self.cholesky_Qvv[j, :, :] @ fi / denominator)
                X_sequential[x_index, j, :] = G_cw
            x_index += 1

        # ---------------------------------------------------------
        # 2. RUN NEW VECTORIZED LOGIC
        # ---------------------------------------------------------
        print("Running vectorized extraction...")
        X_vectorized = np.zeros_like(X_sequential)
        chol_inv_batched = LA.inv(self.cholesky_Qvv)
        x_index = 0
        
        for l in range(frame_before, min(frame_before + max_frames, self.n_frames - frame_after)):
            Zvv = np.zeros((NUP, self.M, self.M), dtype=complex)
            sum_win_vad = 0
            for p in range(frame_before + frame_after + 1):
                win_val = win_vad[10 - frame_before + p]
                z_slice_batched = self.z_k[:, :, l - frame_before + p].T[:, :, np.newaxis]
                
                temp_zvv = chol_inv_batched @ (win_val * z_slice_batched)
                Zvv += temp_zvv @ temp_zvv.conj().transpose(0, 2, 1)
                sum_win_vad += win_val
                
            Zvv /= sum_win_vad

            w, v = LA.eig(Zvv)
            max_idx = np.argmax(w.real, axis=1)
            fi = v[np.arange(NUP), :, max_idx][:, :, np.newaxis]

            numerator = self.cholesky_Qvv @ fi
            denominator = self.cholesky_Qvv[:, 0, :][:, np.newaxis, :] @ fi
            X_vectorized[x_index, :, :] = np.squeeze(numerator / denominator)
            
            x_index += 1

        # ---------------------------------------------------------
        # 3. COMPARE RESULTS
        # ---------------------------------------------------------
        # We use np.allclose rather than strict == because batched operations 
        # and serial operations handle memory and rounding slightly differently 
        # at the C-library level, leading to microscopic floating-point variations.
        is_identical = np.allclose(X_sequential, X_vectorized, rtol=1e-5, atol=1e-8)
        
        if is_identical:
            print("SUCCESS: Vectorized RTFs match the sequential RTFs exactly!")
        else:
            print("WARNING: Mismatch detected between implementations.")
            # Calculate Mean Squared Error to see how far off it is
            mse = np.mean(np.abs(X_sequential - X_vectorized)**2)
            print(f"Mean Squared Error: {mse}")
            
        # Assign to class state to keep pipeline intact if you continue the run
        self.X = X_vectorized
    
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