from pipeline_ofer_funcs import * # Assumes this is available in your environment

print("-----------------STARTING LCMV PIPELINE-----------------")
py_folder = os.path.dirname(os.path.realpath(__file__))
workspace_folder = py_folder
folder_to_all_data = os.path.join(workspace_folder, 'data')
folder_to_test_data = os.path.join(folder_to_all_data, 'simulated_audio', 'test')

class SpatialTrackingPipeline:
    def __init__(self, run_idx, config, folder_to_test_data, n_mics=4, verbose=1):
        """
        Initializes the pipeline configuration and loads Keras models.
        """
        self.run_idx = run_idx
        self.config = config
        self.folder_to_test_data = folder_to_test_data
        self.n_mics = n_mics
        self.verbose = verbose  

        # Data states
        self.scaler = StandardScaler()
        self.input_data = None
        self.n_frames = 0
        
        # Signal & Feature placeholders
        self.stft_mixed = None
        self.stft_spk1 = None
        self.stft_spk2 = None
        self.angles_spk1 = None
        self.angles_spk2 = None
        
        self.chol_noise_cov = None  # Cholesky decomposition of noise covariance
        self.gevd_features = None   # Spatial features extracted via GEVD
        self.model_inputs = None    # Final tensor fed to models
        
        # Ground Truth & Prediction placeholders
        self.true_csd = None
        self.true_doa = None
        self.pred_csd = None
        self.pred_doa = None

        # Model placeholders
        self.model_csd = None
        self.model_doa = None

        self._load_models()

    def _load_models(self):
        """Loads the pre-trained Keras models into memory."""
        if self.verbose:
            print("--- Loading Neural Network Models ---")
        
        self.model_csd = load_model(self.config['csd_path'], compile=False)
        self.model_doa = load_model(self.config['doa_path'], compile=False)
        
        if self.verbose > 1:
            print("Models loaded successfully.")

    def load_data(self):
        """Loads audio and angle data, calculates frames, and verifies dimensions."""
        if self.verbose:
            print(f"--- Loading data for Experiment {self.run_idx} ---")
        
        self.input_data = load_and_preprocess_experiment(
            self.folder_to_test_data, 
            self.run_idx, 
            mic_indices=list(range(self.n_mics))
        )
        
        audio_length = self.input_data['audio']['mixed'].shape[0]
        self.n_frames = calc_n_frames(audio_length, self.config['hoplen'], self.config['wlen'])
    
        self.angles_spk1 = self.input_data['angles']['spk1'].reshape(-1, 1)
        self.angles_spk2 = self.input_data['angles']['spk2'].reshape(-1, 1)

        if self.verbose > 1:
            print(f"Loaded successfully. Sample rate: {self.input_data['fs']} Hz")
            print(f"Mixed audio shape: {self.input_data['audio']['mixed'].shape}")
        
        if len(self.angles_spk1) != self.n_frames:
            raise ValueError(f"Dimension mismatch! Audio yielded {self.n_frames} frames, "
                             f"but label array has {len(self.angles_spk1)} frames.")

    def compute_stft(self):
        """Transforms all necessary audio signals into the time-frequency domain."""
        if self.verbose:
            print("--- Computing STFTs ---")
        
        n_bins = self.config['n_bins']
        win = self.config['win']
        hoplen = self.config['hoplen']
        n_fft = self.config['n_fft']

        self.stft_mixed = np.zeros((self.n_mics, n_bins, self.n_frames), dtype=complex)
        self.stft_spk1 = np.zeros((self.n_mics, n_bins, self.n_frames), dtype=complex)
        self.stft_spk2 = np.zeros((self.n_mics, n_bins, self.n_frames), dtype=complex)

        for i in range(self.n_mics):
            self.stft_mixed[i, :, :] = stft(self.input_data['audio']['mixed'][:, i], win, hoplen, n_fft)
            self.stft_spk1[i, :, :] = stft(self.input_data['audio']['spk1'][:, i], win, hoplen, n_fft)
            self.stft_spk2[i, :, :] = stft(self.input_data['audio']['spk2'][:, i], win, hoplen, n_fft)

    def estimate_noise_covariance(self):
        """Estimates noise covariance from an early silent segment and computes the Cholesky factor."""
        if self.verbose:
            print("--- Estimating Noise Covariance ---")
            
        n_bins = self.config['n_bins']
        pad = self.config['silent_frames']
        
        self.chol_noise_cov = np.zeros((n_bins, self.n_mics, self.n_mics), dtype=complex)
        
        for i in range(n_bins):
            # Calculate Power Spectral Density (PSD) for the noise profile
            psd_noise = self.stft_mixed[:, i, 0:pad] @ (self.stft_mixed[:, i, 0:pad].conj().T) / pad
            self.chol_noise_cov[i, :, :] = LA.cholesky(psd_noise)

    def extract_gevd_features(self):
        """Extracts GEVD-derived spatial features using NumPy broadcasting across frequency bins."""
        if self.verbose:
            print("--- Extracting GEVD Features (Vectorized) ---")
            
        n_bins = self.config['n_bins']
        frame_before = self.config['frame_before']
        frame_after = self.config['frame_after']
        win_vad = self.config['win_vad']
        
        # Batch inverse of Cholesky matrices for all frequency bins
        chol_inv_batched = LA.inv(self.chol_noise_cov)
        
        n_active_frames = self.n_frames - frame_before - frame_after
        self.gevd_features = np.zeros((n_active_frames, n_bins, self.n_mics), dtype=complex)
        frame_idx = 0
        
        for l in range(frame_before, self.n_frames - frame_after):
            if self.verbose > 1 and l % 50 == 0:
                print(f"Processing frame {l}/{n_active_frames}...")
            
            cov_matrix = np.zeros((n_bins, self.n_mics, self.n_mics), dtype=complex)
            sum_win_vad = 0
            
            for p in range(frame_before + frame_after + 1):
                win_val = win_vad[10 - frame_before + p]
                stft_slice = self.stft_mixed[:, :, l - frame_before + p]
                
                # Align for batched multiplication (n_bins, n_mics, 1)
                stft_batched = stft_slice.T[:, :, np.newaxis]
                temp_cov = chol_inv_batched @ (win_val * stft_batched)
                
                # Outer product across microphones
                cov_matrix += temp_cov @ temp_cov.conj().transpose(0, 2, 1)
                sum_win_vad += win_val
                
            cov_matrix /= sum_win_vad

            # Vectorized Eigendecomposition
            eigenvals, eigenvecs = LA.eig(cov_matrix)
            max_idx = np.argmax(eigenvals.real, axis=1)
            
            # Extract dominant eigenvectors
            dominant_vecs = eigenvecs[np.arange(n_bins), :, max_idx][:, :, np.newaxis]

            # Normalize against the reference microphone (index 0)
            numerator = self.chol_noise_cov @ dominant_vecs
            denominator = self.chol_noise_cov[:, 0, :][:, np.newaxis, :] @ dominant_vecs
            
            self.gevd_features[frame_idx, :, :] = np.squeeze(numerator / denominator)
            frame_idx += 1

    def prepare_model_inputs(self):
        """Formats spatial features and reference log-magnitude into model-ready tensors."""
        if self.verbose:
            print("--- Preparing Model Inputs ---")

        n_bins = self.config['n_bins']
        frame_before = self.config['frame_before']
        frame_after = self.config['frame_after']

        # 1. Non-Reference Spatial Features (real & imaginary parts concatenated)
        spatial_non_ref = self.gevd_features[:, :, 1:self.n_mics]
        features_spatial = np.concatenate((spatial_non_ref.real, spatial_non_ref.imag), axis=2)

        # Per-Frame Scaling
        for b in range(len(features_spatial)):
            features_spatial[b, :, :] = self.scaler.fit_transform(features_spatial[b, :, :])

        # 2. Reference Microphone (Log-Magnitude)
        stft_ref = self.stft_mixed[0, :, frame_before:self.n_frames - frame_after].T
        log_mag_ref = np.log(abs(stft_ref))

        # Scale log-magnitude across the sequence
        log_mag_scaled = self.scaler.fit_transform(log_mag_ref)
        log_mag_scaled = np.reshape(log_mag_scaled, (log_mag_scaled.shape[0], n_bins, 1))

        # Combine spatial features and log-magnitude reference
        self.model_inputs = np.concatenate((features_spatial, log_mag_scaled), axis=2)

        if self.verbose > 1:
            print(f"Final input tensor shape: {self.model_inputs.shape}")
    
    def compute_ground_truth_labels(self):
        """Computes VAD masks, applies temporal smoothing, and maps DOA angles."""
        if self.verbose:
            print("--- Computing Ground Truth Labels ---")

        threshold_freq = self.config['threshold_freq']
        threshold = self.config['threshold']
        frame_before = self.config['frame_before']
        frame_after = self.config['frame_after']

        def compute_raw_vad(stft_clean):
            vad_temp = abs(stft_clean)
            vad_temp = vad_temp / vad_temp.std()
            vad_temp = vad_temp.mean(axis=0)
            vad_temp = vad_temp > threshold_freq
            vad_sum = vad_temp.astype(int).sum(axis=0)
            return (vad_sum > threshold).astype(int)

        vad1 = compute_raw_vad(self.stft_spk1)
        vad2 = compute_raw_vad(self.stft_spk2)

        # Temporal Smoothing
        check_vad1 = np.zeros(self.n_frames)
        check_vad2 = np.zeros(self.n_frames)

        for l in range(frame_before, self.n_frames - frame_after):
            check_vad1[l] = vad1[l-1:l+2].sum()
            check_vad2[l] = vad2[l-1:l+2].sum()
        
        for l in range(frame_before, self.n_frames - frame_after):
            vad1[l] = 1 if check_vad1[l] == 3 else 0
            vad2[l] = 1 if check_vad2[l] == 3 else 0

        # Angle Alignment
        vad1_location = self.angles_spk1.flatten() * vad1
        vad2_location = self.angles_spk2.flatten() * vad2
        
        # Concurrent Speaker Detection (CSD) labels
        active_speakers = vad1 + vad2
        self.true_csd = active_speakers[frame_before:self.n_frames - frame_after]

        # DOA labels
        doa_combined = vad1_location + vad2_location
        self.true_doa = doa_combined[frame_before:self.n_frames - frame_after]
        
        # Sentinel Label (19) for overlap frames
        self.true_doa = np.where(self.true_csd != 2, self.true_doa, 19)

    def run_inference(self):
        """Executes forward passes on the loaded models."""
        if self.verbose:
            print("--- Running Inference ---")

        pred_csd_probs = self.model_csd.predict(self.model_inputs)
        self.pred_csd = np.argmax(pred_csd_probs, axis=1)

        pred_doa_probs = self.model_doa.predict(self.model_inputs)
        self.pred_doa = np.argmax(pred_doa_probs, axis=1)

    def evaluate_and_save(self, folder_to_save):
        """Post-processes predictions, saves outputs, and plots confusion matrices."""
        if self.verbose:
            print("--- Evaluating, Saving, and Plotting ---")

        os.makedirs(folder_to_save, exist_ok=True)

        # Median Filtering
        true_csd_filtered = ndimage.median_filter(self.true_csd, size=11)
        pred_csd_filtered = ndimage.median_filter(self.pred_csd, size=25)
        pred_doa_filtered = ndimage.median_filter(self.pred_doa, size=11)

        # Save Arrays
        np.save(os.path.join(folder_to_save, f'estimate_DOA_{self.run_idx}.npy'), pred_doa_filtered)
        np.save(os.path.join(folder_to_save, f'true_DOA_{self.run_idx}.npy'), self.true_doa)
        np.save(os.path.join(folder_to_save, f'estimate_CSD_{self.run_idx}.npy'), pred_csd_filtered)
        np.save(os.path.join(folder_to_save, f'true_CSD_{self.run_idx}.npy'), true_csd_filtered)

        # Plotting Defaults
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
            true_csd_filtered, pred_csd_filtered, 3, cm_plot_labels_csd,
            annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis,
            name=f'CSD_Confusion_Matrix_Experiment_{self.run_idx}.png', plot_folder=plot_confusion_folder
        )

        # Filter DOA data for plotting (Remove noise and overlap classes)
        valid_indices = np.where((self.true_doa != 0) & (self.true_doa != 19))
        true_doa_plot = self.true_doa[valid_indices]
        pred_doa_plot = pred_doa_filtered[valid_indices]

        # Plot 2: DOA Confusion Matrix
        cm_plot_labels_doa = [f'{i}-{i+9}' for i in range(0, 180, 10)]
        if self.verbose > 1:
            print("Plotting DOA Confusion Matrix...")
            
        plot_confusion_matrix_from_data(
            true_doa_plot - 1, pred_doa_plot - 1, 18, cm_plot_labels_doa,
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


# ==========================================
# CONFIGURATION
# ==========================================
models_folder = os.path.join(workspace_folder, 'models')

pipeline_config = {
    # --- Models Config ---
    'csd_path': os.path.join(models_folder, 'model3_GEVD_30_3.h5'),
    'doa_path': os.path.join(models_folder, 'model18_GEVD_30_3.h5'),

    # --- STFT Config ---
    'n_fft': 2048,
    'hoplen': 512,
    'wlen': 2048,
    'n_bins': 1025,             # Number of frequency bins (= n_fft//2 + 1)
    'win': np.hamming(2048),
    'silent_frames': 30, 

    # --- Tracking Config ---
    'frame_before': 8,
    'frame_after': 5,
    'win_vad': np.hamming(21),
    'threshold': 40,
    'threshold_freq': 0.3
}

plot_dir = os.path.join(workspace_folder, 'plots')

# Example instantiation
pipeline = SpatialTrackingPipeline(
    run_idx=1, 
    config=pipeline_config, 
    folder_to_test_data=folder_to_test_data, 
    n_mics=4, 
    verbose=2
)
pipeline.run(plot_dir)