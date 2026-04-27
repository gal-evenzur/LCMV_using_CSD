from pipeline_ofer_funcs import *

print("-----------------STARTING LCMV PIPELINE-----------------")
py_folder = os.path.dirname(os.path.realpath(__file__))
workspace_folder = py_folder
folder_to_all_data = os.path.join(workspace_folder,'data')
folder_to_test_data = os.path.join(folder_to_all_data, 'simulated_audio', 'test')

class SpatialTrackingPipeline:
    def __init__(self, run_idx, p_models, p_stft, p_tracking, folder_to_test_data, M=4, verbose=1):
        """
        Initializes the pipeline configuration and loads Keras models.
        """
        self.run_idx = run_idx
        self.p_models = p_models
        self.p_stft = p_stft
        self.p_tracking = p_tracking
        self.folder_to_test_data = folder_to_test_data
        self.M = M
        self.verbose = verbose  

        # Data states
        self.scaler = StandardScaler()
        self.input_data = None
        self.n_frames = 0
        
        # Matrix placeholders
        self.z_k = None
        self.z_k_first = None
        self.z_k_second = None
        self.cholesky_Qvv = None
        self.X = None
        self.x_test = None
        
        # Model placeholders
        self.model_csd = None
        self.model_doa = None

        # Load models immediately upon instantiation
        self._load_models()

    def _load_models(self):
        """Loads the pre-trained Keras models into memory."""
        if self.verbose:
            print("--- Loading Neural Network Models ---")
        
        # compile=False saves time and avoids warnings if custom loss functions aren't needed for inference
        self.model_csd = load_model(self.p_models['csd_path'], compile=False)
        self.model_doa = load_model(self.p_models['doa_path'], compile=False)
        
        if self.verbose > 1:
            print("Models loaded successfully.")

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

    def prepare_model_inputs(self):
        """
        Formats the GEVD spatial features and reference log-magnitude spectrum 
        into the exact tensor shape expected by the Keras models.
        """
        if self.verbose:
            print("--- Preparing Model Inputs ---")

        NUP = self.p_stft['NUP']
        frame_before = self.p_tracking['frame_before']
        frame_after = self.p_tracking['frame_after']

        # 1. Isolate Non-Reference Spatial Features
        # X shape: (frames, NUP, M). We skip index 0 (reference mic).
        X_non_ref = self.X[:, :, 1:self.M]
        
        # Concatenate real and imaginary components along the feature axis
        X_T = np.concatenate((X_non_ref.real, X_non_ref.imag), axis=2)

        # 2. Per-Frame Spatial Scaling
        # Replicating original logic: fit_transform applied individually to each time frame
        for b in range(len(X_T)):
            X_T[b, :, :] = self.scaler.fit_transform(X_T[b, :, :])

        # 3. Process Reference Microphone (Log-Magnitude)
        # Extract the STFT of the 1st mic, crop temporal padding, and transpose to (frames, NUP)
        z_k_0 = self.z_k[0, :, frame_before:self.n_frames - frame_after].T
        z_k_0_log = np.log(abs(z_k_0))

        # Scale the log-magnitude features across the entire temporal sequence at once
        z_k_0_standart = self.scaler.fit_transform(z_k_0_log)
        
        # Reshape to (frames, NUP, 1) to match X_T dimensions
        z_k_0_standart = np.reshape(z_k_0_standart, (z_k_0_standart.shape[0], NUP, 1))

        # 4. Final Tensor Assembly
        # Combine spatial features (6 channels) with spectral features (1 channel)
        self.x_test = np.concatenate((X_T, z_k_0_standart), axis=2)

        if self.verbose > 1:
            print(f"Final input tensor shape: {self.x_test.shape}")
    
    def compute_ground_truth_labels(self):
        """
        Step 5: Computes VAD masks from clean references, applies temporal smoothing, 
        and maps DOA angles to active frames.
        """
        if self.verbose:
            print("--- Computing Ground Truth Labels ---")

        threshold_freq = self.p_tracking['threshold_freq']
        threshold = self.p_tracking['threshold']
        frame_before = self.p_tracking['frame_before']
        frame_after = self.p_tracking['frame_after']

        # 1. Frequency-Based Voice Activity Detection (VAD)
        def compute_raw_vad(z_k_clean):
            # Calculate absolute magnitude
            vad_temp = abs(z_k_clean)
            # Standardize based on global standard deviation
            vad_temp = vad_temp / vad_temp.std()
            # Mean across the M microphone channels -> shape: (NUP, frames)
            vad_temp = vad_temp.mean(axis=0)
            # Boolean mask for frequencies exceeding threshold
            vad_temp = vad_temp > threshold_freq
            # Sum active frequencies per frame and apply frame-level threshold
            vad_sum = vad_temp.astype(int).sum(axis=0)
            return (vad_sum > threshold).astype(int)

        vad1 = compute_raw_vad(self.z_k_first)
        vad2 = compute_raw_vad(self.z_k_second)

        # 2. Temporal Smoothing (Original For-Loop Implementation)
        check_vad1 = np.zeros(self.n_frames)
        check_vad2 = np.zeros(self.n_frames)

        for l in range(frame_before, self.n_frames - frame_after):
            check_vad1[l] = vad1[l-1:l+2].sum()
            check_vad2[l] = vad2[l-1:l+2].sum()
        
        for l in range(frame_before, self.n_frames - frame_after):
            vad1[l] = 1 if check_vad1[l] == 3 else 0
            vad2[l] = 1 if check_vad2[l] == 3 else 0

        # 3. Angle Alignment and Label Generation
        # Multiply flattened angles by the binary VAD mask
        vad1_location_update = self.y2_first.flatten() * vad1
        vad2_location_update = self.y2_second.flatten() * vad2
        
        # Concurrent Speaker Detection (CSD) labels: 0 (Noise), 1 (One Spk), 2 (Overlap)
        L = vad1 + vad2
        self.y = L[frame_before:self.n_frames - frame_after]

        # DOA labels
        y2_temp = vad1_location_update + vad2_location_update
        self.y2 = y2_temp[frame_before:self.n_frames - frame_after]
        
        # Apply Sentinel Label (19) for frames where both speakers are active
        self.y2 = np.where(self.y != 2, self.y2, 19)

    def run_inference(self):
        """
        Step 6: Executes the forward pass on the loaded models using the formatted inputs.
        """
        if self.verbose:
            print("--- Running Inference ---")

        # Predict CSD (Concurrent Speaker Detection)
        y_pred = self.model_csd.predict(self.x_test)
        # Replaces Pandas idxmax with NumPy argmax (faster, zero overhead)
        self.y_prob_stat = np.argmax(y_pred, axis=1)

        # Predict DOA (Direction of Arrival)
        y2_pred = self.model_doa.predict(self.x_test)
        self.y2_prob_stat = np.argmax(y2_pred, axis=1)

    def evaluate_and_save(self, folder_to_save):
        """
        Step 7: Post-processes predictions via median filtering, saves outputs, and plots confusion matrices.
        """
        if self.verbose:
            print("--- Evaluating, Saving, and Plotting ---")

        # Create save directory if it doesn't exist
        os.makedirs(folder_to_save, exist_ok=True)

        # 1. Median Filtering
        y_mf = ndimage.median_filter(self.y, size=11)
        y_prob_stat_mf = ndimage.median_filter(self.y_prob_stat, size=25)
        y2_prob_stat_mf = ndimage.median_filter(self.y2_prob_stat, size=11)

        # 2. Save Arrays
        np.save(os.path.join(folder_to_save, f'estimate_DOA_{self.run_idx}.npy'), y2_prob_stat_mf)
        np.save(os.path.join(folder_to_save, f'true_DOA_{self.run_idx}.npy'), self.y2)
        np.save(os.path.join(folder_to_save, f'estimate_CSD_{self.run_idx}.npy'), y_prob_stat_mf)
        np.save(os.path.join(folder_to_save, f'true_CSD_{self.run_idx}.npy'), y_mf)

        # 3. Plotting Setup (Formatting labels to match original script)
        annot, cmap, fmt, fz, lw, cbar = True, 'Oranges', '.2f', 9, 0.5, False
        show_null_values, pred_val_axis = 2, 'y'
        figsize = [18, 18]

        # Plot 1: CSD Confusion Matrix
        cm_plot_labels_csd = ['Noise', 'One speaker', '2 speakers']
        if self.verbose > 1:
            print("Plotting CSD Confusion Matrix...")
        plot_confusion_folder = os.path.join(folder_to_save, 'confusion_matrices')
        os.makedirs(plot_confusion_folder, exist_ok=True)
        plot_confusion_matrix_from_data(
            y_mf, y_prob_stat_mf, 3, cm_plot_labels_csd,
            annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis,
            name=f'CSD_Confusion_Matrix_Experiment_{self.run_idx}.png', plot_folder=plot_confusion_folder
        )

        # 4. Filter DOA data for plotting (Remove 0 noise and 19 overlap classes)
        y2_prob_plot = np.delete(y2_prob_stat_mf, np.where(self.y2 == 0)[0])
        y2_true_plot = np.delete(self.y2, np.where(self.y2 == 0))
        y2_prob_plot = np.delete(y2_prob_plot, np.where(y2_true_plot == 19))
        y2_true_plot = np.delete(y2_true_plot, np.where(y2_true_plot == 19))

        # Plot 2: DOA Confusion Matrix
        cm_plot_labels_doa = [f'{i}-{i+9}' for i in range(0, 180, 10)]
        # Offset arrays by -1 because classes are 1-18, but plotting logic expects 0-indexed arrays
        if self.verbose > 1:
            print("Plotting DOA Confusion Matrix...")
        plot_confusion_matrix_from_data(
            y2_true_plot - 1, y2_prob_plot - 1, 18, cm_plot_labels_doa,
            annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis,
            name=f'DOA_Confusion_Matrix_Experiment_{self.run_idx}.png', plot_folder=plot_confusion_folder
        )

    def run(self, folder_to_save):
        """Orchestrates the pipeline steps."""
        self.load_data()
        self.compute_stft()
        self.estimate_noise_covariance()
        self.extract_gevd_features()
        self.prepare_model_inputs()
        self.compute_ground_truth_labels()
        self.run_inference()
        self.evaluate_and_save(folder_to_save)
        
        if self.verbose:
            print(f"--- Pipeline Execution Complete for Experiment {self.run_idx} ---")
        return self
# Configuration setup
models_folder = os.path.join(workspace_folder, 'models')
p_models = { # Need to be updated with actual model filenames
    'csd_path': os.path.join(models_folder, 'model3_GEVD_30_3.h5'),
    'doa_path': os.path.join(models_folder, 'model18_GEVD_30_3.h5')
}

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

plot_dir = os.path.join(workspace_folder, 'plots')

# Example instantiation
pipeline = SpatialTrackingPipeline(run_idx=1, p_stft=p_stft, p_tracking=p_tracking, p_models=p_models,
                                    folder_to_test_data=folder_to_test_data, M=4, verbose=2)
pipeline.run(plot_dir)