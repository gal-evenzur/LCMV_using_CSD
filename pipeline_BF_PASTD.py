from pipeline_ofer_funcs import *
import time
import os
import numpy as np
import numpy.linalg as LA
from scipy.io import wavfile
import mir_eval
from librosa.core import stft, istft

"""
=========================================================================================
SPATIAL AUDIO SEPARATION PIPELINE
=========================================================================================
This module implements an offline, multi-microphone spatial audio separation system.
It is designed to take a raw mixed audio recording and a set of pre-calculated tracking 
labels (CSD: Concurrent Speaker Detection, and DOA: Direction of Arrival) to isolate 
individual speakers from background noise.

Algorithm Overview:
1. Domain Shift: Converts multi-channel time-domain audio into the frequency domain (STFT).
2. Noise Estimation: Uses periods of silence (CSD=0) to build a spatial noise covariance 
   matrix (Qvv).
3. Speaker Fingerprinting: Uses single-speaker periods (CSD=1) and their DOA labels to 
   estimate the Relative Transfer Function (RTF) via Generalized Eigenvalue Decomposition 
   (GEVD) or Subspace Tracking (PASTd). This creates a spatial "fingerprint" (G) for each speaker.
4. Beamforming: Applies Minimum Variance Distortionless Response (MVDR) for single speakers, 
   or Linearly Constrained Minimum Variance (LCMV) beamforming for overlapping speakers, 
   actively suppressing noise and cross-talk.
5. Synthesis: Converts the separated frequency frames back into listenable audio (ISTFT).
=========================================================================================
"""

class SpatialSeparationPipeline:
    def __init__(self, run_idx, p_stft, p_tracking, p_beamforming, folder_to_test_data, folder_to_results, M=4, num_speech=2, verbose=1, rtf_method='gevd'):
        """
        Initializes the spatial separation pipeline.
        Args:
            run_idx (int): The ID number of the experiment (e.g., 3).
            p_stft (dict): Configuration for STFT parameters.
            p_tracking (dict): Configuration for tracking thresholds and context windows.
            p_beamforming (dict): Configuration for beamforming parameters and smoothing factors.
            folder_to_test_data (str): Path to the directory containing the raw test data.
            folder_to_results (str): Path to the directory where results will be saved.
            M (int): Number of microphone channels to process.
            num_speech (int): Number of simultaneous speakers to separate.
            verbose (int): Level of logging detail (0: none, 1: key steps, 2: detailed).
            rtf_method (str): 'gevd' or 'pastd' to choose the RTF estimation algorithm.
        """
        self.run_idx = run_idx
        self.p_stft = p_stft
        self.p_tracking = p_tracking
        self.p_beamforming = p_beamforming
        self.folder_to_test_data = folder_to_test_data
        self.folder_to_results = folder_to_results
        self.M = M
        self.num_speech = num_speech
        self.verbose = verbose
        
        self.rtf_method = rtf_method.lower()
        if self.rtf_method not in ['gevd', 'pastd']:
            raise ValueError("rtf_method must be either 'gevd' or 'pastd'")

        # Audio & Labels
        self.fs = None
        self.z_k = None
        self.z_k_first = None
        self.z_k_second = None
        self.y_prob_stat_mf = None
        self.y2_prob_stat_mf = None
        self.y_mf = None  # True CSD for overlap evaluation bounds
        self.evaluate = False # Flag set to True if reference files are successfully loaded

        # State Variables
        self.NUP = self.p_stft['NUP']
        self.buffer_size = self.p_beamforming['buffer_size']

        self.Qvv = np.zeros((self.NUP, self.M, self.M), dtype=complex)
        for j in range(self.NUP):
            self.Qvv[j, :, :] = np.eye(self.M)
        self.Qvv_temp = np.zeros((self.NUP, self.M, self.M), dtype=complex)

        self.G = np.ones((self.NUP, self.M, self.num_speech), dtype=complex)
        self.W = None # Instantiated dynamically once frame count is known
        self.s_hat_total = None

        self.PSD_matrix_per_DOA = np.zeros((18, self.NUP, self.M, self.M), dtype=complex)
        self.total_frame_per_DOA = np.zeros(18)
        self.stand_z = np.empty((0, self.NUP, self.M), dtype=complex)

        # PASTd State Initialization (Only if selected)
        if self.rtf_method == 'pastd':
            # Shape: 18 DOAs, NUP frequencies, M mics, 1 vector
            self.w_pastd = np.random.randn(18, self.NUP, self.M, 1) + 1j * np.random.randn(18, self.NUP, self.M, 1)
            self.w_pastd /= LA.norm(self.w_pastd, axis=2, keepdims=True)
            self.d_pastd = np.full((18, self.NUP, 1, 1), 1e-3, dtype=float)
            self.beta_pastd = self.p_beamforming.get('beta_pastd', 0.95)

        # Frame classification system layout:
        # row 0 -> [active_DOA_slot_0, active_DOA_slot_1, candidate_DOA]
        # row 1 -> [frame_count_slot_0, frame_count_slot_1, frame_count_candidate]
        self.Frame_classification_system = np.zeros((2, 3))

        # Rolling context windows
        self.save_last_frames_first = np.zeros((self.buffer_size, self.NUP, self.M), dtype=complex)
        self.save_last_frames_doa_first = np.zeros(self.buffer_size)
        self.save_last_frames_second = np.zeros((self.buffer_size, self.NUP, self.M), dtype=complex)
        self.save_last_frames_doa_second = np.zeros(self.buffer_size)

        # Trackers
        self.time_first = 0
        self.time_second = 0
        self.first_speaker_active = 0
        self.second_speaker_active = 0
        self.flag_first_noise = 0
        self.sum_Qvv = 0
        self.flag_start_lcmv = 1
        self.alfa_Qvv = self.p_beamforming['alfa_Qvv_init']

    def load_data(self):
        """Loads mixed audio, labels, and optionally clean references for evaluation."""
        if self.verbose:
            print(f"--- Loading data for Experiment {self.run_idx} ({self.rtf_method.upper()}) ---")

        signal_file = os.path.join(self.folder_to_test_data, f'together_{self.run_idx}.wav')
        self.fs, receivers = wavfile.read(signal_file)
        self.receivers = receivers[:, :self.M] / np.max(np.abs(receivers))

        # Load tracked outputs from the results folder
        self.y2_prob_stat_mf = np.load(os.path.join(self.folder_to_results, f'estimate_DOA_{self.run_idx}.npy'))
        self.y_prob_stat_mf = np.load(os.path.join(self.folder_to_results, f'estimate_CSD_{self.run_idx}.npy'))

        # Optional: Load reference signals for evaluation
        try:
            _, ref_first = wavfile.read(os.path.join(self.folder_to_test_data, f'first_{self.run_idx}.wav'))
            _, ref_second = wavfile.read(os.path.join(self.folder_to_test_data, f'second_{self.run_idx}.wav'))

            self.receiver_first = ref_first[:, :self.M] / np.max(np.abs(ref_first))
            self.receiver_second = ref_second[:, :self.M] / np.max(np.abs(ref_second))

            # Ground truth CSD is usually saved in the results folder by the tracking pipeline
            self.y_mf = np.load(os.path.join(self.folder_to_results, f'true_CSD_{self.run_idx}.npy'))
            self.evaluate = True
            if self.verbose > 1: print("Reference files found. Evaluation metrics will be computed.")
        except FileNotFoundError:
            self.evaluate = False
            if self.verbose > 1: print("Reference files not found. Skipping evaluation metrics.")

        self.W = np.ones((len(self.y2_prob_stat_mf), self.NUP, self.M, self.num_speech), dtype=complex)

    def compute_stft(self):
        """Transforms signals into the time-frequency domain and applies trimming."""
        if self.verbose:
            print("--- Computing STFTs ---")

        win = self.p_stft['win']
        hop = self.p_stft['hop']
        nfft = self.p_stft['nfft']
        f_before = self.p_tracking['frame_before']
        f_after = self.p_tracking['frame_after']

        index = int(1 + np.fix((len(self.receivers[:, 1]) - self.p_stft['wlen']) / hop))

        def do_stft(audio):
            z = np.zeros((self.M, self.NUP, index), dtype=complex)
            for i in range(self.M):
                z[i, :, :] = stft(audio[:, i], win, hop, nfft)
            z = np.transpose(z, (2, 1, 0)) # -> [frame, freq, mic]
            return z[f_before:index - f_after, :, :]

        self.z_k = do_stft(self.receivers)

        if self.evaluate:
            self.z_k_first = do_stft(self.receiver_first)
            self.z_k_second = do_stft(self.receiver_second)

    def _update_noise_covariance(self, l):
        """Handles CSD=0 (Noise-only frames) using batched matrix operations."""
        self.time_second += 1
        self.time_first += 1

        # 1. Extract the current frame for all frequencies: Shape (NUP, M)
        z_frame = self.z_k[l]

        # 2. Reshape for batched outer product
        # z_col: (NUP, M, 1) | z_row: (NUP, 1, M)
        z_col = z_frame[:, :, np.newaxis]
        z_row = z_frame[:, np.newaxis, :].conj()

        # 3. Calculate instantaneous covariance for all NUP bins at once: (NUP, M, M)
        Qvv_inst = z_col @ z_row

        if self.flag_first_noise == 0:
            self.flag_first_noise = 1
            self.Qvv = Qvv_inst.copy()
            self.sum_Qvv = 1
        else:
            self.sum_Qvv += 1
            self.alfa_Qvv = self.p_beamforming['alfa_Qvv_run']
            self.Qvv = (1 - self.alfa_Qvv) * self.Qvv + (self.alfa_Qvv) * Qvv_inst

    def _update_rtf_and_tracking(self, l):
        """Handles CSD=1 (Single speaker active). Manages slots and GEVD/PASTd."""
        y2_prob = self.y2_prob_stat_mf[l]

        # Push to context buffer if near an active slot
        if abs(y2_prob - self.Frame_classification_system[0, 0]) < 3 and self.Frame_classification_system[0, 0] != 0:
            self.save_last_frames_first = np.roll(self.save_last_frames_first, 1, axis=0)
            self.save_last_frames_first[0] = self.z_k[l]
            self.save_last_frames_doa_first = np.roll(self.save_last_frames_doa_first, 1)
            self.save_last_frames_doa_first[0] = y2_prob

        elif abs(y2_prob - self.Frame_classification_system[0, 1]) < 3 and self.Frame_classification_system[0, 0] != 0:
            self.save_last_frames_second = np.roll(self.save_last_frames_second, 1, axis=0)
            self.save_last_frames_second[0] = self.z_k[l]
            self.save_last_frames_doa_second = np.roll(self.save_last_frames_doa_second, 1)
            self.save_last_frames_doa_second[0] = y2_prob

        # SLOT 0 UPDATE
        if y2_prob == self.Frame_classification_system[0, 0]:
            self._update_slot(0, y2_prob, l, self.save_last_frames_first, self.save_last_frames_doa_first)

        # SLOT 1 UPDATE
        elif y2_prob == self.Frame_classification_system[0, 1]:
            self._update_slot(1, y2_prob, l, self.save_last_frames_second, self.save_last_frames_doa_second)

        # CANDIDATE DOA PERSISTENCE BRANCH
        elif y2_prob == self.Frame_classification_system[0, 2]:
            self._process_candidate_doa(l, y2_prob)

        # START TRACKING NEW CANDIDATE
        else:
            self.stand_z = self.z_k[l, :, :].reshape(1, self.NUP, self.M)
            self.Frame_classification_system[0, 2] = y2_prob
            self.Frame_classification_system[1, 2] = 1
            self.time_second += 1
            self.time_first += 1

    def _update_slot(self, slot_idx, y2_prob, l, save_last_frames, save_last_frames_doa):
        """Helper to compute batched GEVD or PASTd RTF for an active slot."""
        epsilon = self.p_beamforming['epsilon']

        self.Frame_classification_system[1, slot_idx] += 1
        self.Frame_classification_system[1, 2] = 0
        self.Frame_classification_system[0, 2] = 0

        if slot_idx == 0:
            self.time_first = 0
            self.time_second += 1
        else:
            self.time_second = 0
            self.time_first += 1

        # Extract valid historical frames
        curr_frames = save_last_frames[np.where(save_last_frames_doa > 0)]
        curr_doa = save_last_frames_doa[np.where(save_last_frames_doa > 0)]
        curr_frames = curr_frames[np.where(abs(curr_doa - y2_prob) != 0)]
        curr_doa = save_last_frames_doa[np.where(abs(curr_doa - y2_prob) != 0)]
        curr_frames = curr_frames[np.where(abs(curr_doa - y2_prob) < 3)]

        self.total_frame_per_DOA[y2_prob - 1] += 1
        alfa_G = (1 + len(curr_frames)) / (self.total_frame_per_DOA[y2_prob - 1] + len(curr_frames))

        # 1. Assemble Batched Signal Matrix
        # Current frame: (NUP, M, 1)
        z_frame = self.z_k[l, :, :, np.newaxis]

        # 2. Batched Cholesky and Inversion
        chol_Qvv = LA.cholesky(self.Qvv) # (NUP, M, M)
        norm_chol = LA.norm(chol_Qvv, axis=(1, 2), keepdims=True)
        chol_inv = LA.inv(chol_Qvv + epsilon * norm_chol * np.eye(self.M))

        if self.rtf_method == 'gevd':
            if len(curr_frames) > 0:
                history_trans = curr_frames.transpose(1, 2, 0)
                all_frames = np.concatenate((z_frame, history_trans), axis=2)
            else:
                all_frames = z_frame

            # 3. Batched Whitening and Covariance
            a = chol_inv @ all_frames
            Zvv_temp = a @ a.conj().transpose(0, 2, 1)

            # Smooth PSD matrix
            self.PSD_matrix_per_DOA[y2_prob - 1] = (1 - alfa_G) * self.PSD_matrix_per_DOA[y2_prob - 1] + (alfa_G) * Zvv_temp

            # 4. Batched Eigendecomposition
            w, v = LA.eig(self.PSD_matrix_per_DOA[y2_prob - 1])
            max_idx = np.argmax(w.real, axis=1)
            phi = v[np.arange(self.NUP), :, max_idx][:, :, np.newaxis]

        else:
            # --- PASTd METHOD ---
            doa_idx = y2_prob - 1
            x_t = chol_inv @ z_frame  # Whiten current frame: (NUP, M, 1)
            
            w_p = self.w_pastd[doa_idx]
            d_p = self.d_pastd[doa_idx]
            
            # Subspace tracking update
            y_proj = w_p.conj().transpose(0, 2, 1) @ x_t     # (NUP, 1, 1)
            d_p = self.beta_pastd * d_p + np.abs(y_proj)**2    # (NUP, 1, 1)
            gain = y_proj.conj() / d_p                       # (NUP, 1, 1)
            residual = x_t - w_p * y_proj                    # (NUP, M, 1)
            w_p = w_p + gain * residual
            w_p = w_p / LA.norm(w_p, axis=1, keepdims=True)      # Normalize
            
            # Save state
            self.w_pastd[doa_idx] = w_p
            self.d_pastd[doa_idx] = d_p
            
            phi = w_p  # The tracked principal eigenvector

        # 5. Batched Recolor and Normalize
        numerator = chol_Qvv @ phi  # (NUP, M, 1)
        denominator = chol_Qvv[:, 0:1, :] @ phi

        # Divide, squeeze out dummy dimension, and save to G
        self.G[:, :, slot_idx] = np.squeeze(numerator / denominator, axis=2)

    def _process_candidate_doa(self, l, y2_prob):
        """Helper to promote a candidate DOA to an active slot using batched ops."""
        epsilon = self.p_beamforming['epsilon']
        thresh = self.p_tracking['threshold_chage_location']

        # Append current frame to candidate buffer
        self.stand_z = np.concatenate((self.stand_z, self.z_k[l, :, :].reshape(1, self.NUP, self.M)))
        self.Frame_classification_system[1, 2] += 1
        self.time_second += 1
        self.time_first += 1

        # If the candidate has persisted long enough to be promoted:
        if self.Frame_classification_system[1, 2] > (thresh - 1):

            # 1. Batched Cholesky and Inversion
            chol_Qvv = LA.cholesky(self.Qvv) # (NUP, M, M)
            norm_chol = LA.norm(chol_Qvv, axis=(1, 2), keepdims=True)
            chol_inv = LA.inv(chol_Qvv + epsilon * norm_chol * np.eye(self.M))

            # 2. Transpose stand_z buffer -> (NUP, M, thresh)
            stand_z_trans = self.stand_z.transpose(1, 2, 0)

            if self.rtf_method == 'gevd':
                # 3. Batched Whitening and Covariance
                a = chol_inv @ stand_z_trans
                Zvv_temp = (a @ a.conj().transpose(0, 2, 1)) / thresh

                # Update PSD
                temp_alfa = self.total_frame_per_DOA[y2_prob - 1] + thresh
                self.PSD_matrix_per_DOA[y2_prob - 1] = (self.total_frame_per_DOA[y2_prob - 1] / temp_alfa) * self.PSD_matrix_per_DOA[y2_prob - 1] + (thresh / temp_alfa) * Zvv_temp
            else:
                # --- PASTd METHOD ---
                doa_idx = y2_prob - 1
                w_p = self.w_pastd[doa_idx]
                d_p = self.d_pastd[doa_idx]
                
                for t in range(thresh):
                    x_t = chol_inv @ stand_z_trans[:, :, t:t+1]
                    y_proj = w_p.conj().transpose(0, 2, 1) @ x_t
                    d_p = self.beta_pastd * d_p + np.abs(y_proj)**2
                    gain = y_proj.conj() / d_p
                    w_p = w_p + gain * (x_t - w_p * y_proj)
                    w_p = w_p / LA.norm(w_p, axis=1, keepdims=True)
                    
                self.w_pastd[doa_idx] = w_p
                self.d_pastd[doa_idx] = d_p

            # 4. Slot assignment policy
            fc = self.Frame_classification_system
            if fc[1, 0] == 0:
                to_change = 0
            elif (fc[1, 1] == 0) and (abs(fc[0, 0] - y2_prob) < 3):
                to_change = 0
            elif fc[1, 1] == 0:
                to_change = 1
            else:
                to_change = np.argmin(np.abs(np.array((y2_prob, y2_prob)) - fc[0, 0:2]))
                min1, min2 = np.abs(np.array((y2_prob, y2_prob)) - fc[0, 0:2])
                if (min1 > (thresh - 2)) and (min2 > (thresh - 2)):
                    self.first_speaker_active = 1
                    self.second_speaker_active = 1
                    to_change = 0 if (self.time_first - min1 * 30) > (self.time_second - min2 * 30) else 1

            self.Frame_classification_system[0, to_change] = y2_prob
            self.Frame_classification_system[1, to_change] = self.Frame_classification_system[1, 2]
            self.Frame_classification_system[0, 2] = 0
            self.Frame_classification_system[1, 2] = 0
            self.total_frame_per_DOA[y2_prob - 1] += thresh

            # Clear candidate buffer now that it's promoted
            self.stand_z = np.empty((0, self.NUP, self.M), dtype=complex)

    def _compute_spatial_filters(self, l):
        """Applies MVDR or LCMV using fully vectorized numpy broadcasting."""
        e = self.p_beamforming['e']
        epsilon = self.p_beamforming['epsilon']
        fc = self.Frame_classification_system

        # Pre-allocate output for this frame
        s_hat = np.zeros((2, self.NUP), dtype=complex)

        # Extract the signal frame and reshape for batched multiplication: (NUP, M, 1)
        z_frame = self.z_k[l, :, :, np.newaxis]

        # Case A: No slots -> Ref mic placeholder
        if fc[0, 0] == 0 and fc[0, 1] == 0:
            s_hat[0, :] = self.z_k[l, :, 0]
            s_hat[1, :] = 1e-10

        # Case B & C require inverted Qvv. We do this once for all frequencies!
        elif fc[0, 0] != 0 or fc[0, 1] != 0:
            # Diagonal Loading: Calculate norm per frequency bin, shape (NUP, 1, 1)
            norm_Qvv = LA.norm(self.Qvv, axis=(1, 2), keepdims=True)
            reg_matrix = e * norm_Qvv * np.eye(self.M)

            # Batch invert all matrices: (NUP, M, M)
            inv_Qvv = LA.inv(self.Qvv + reg_matrix)

            # Case B: One slot -> MVDR
            if fc[0, 0] != 0 and fc[0, 1] == 0:
                g = self.G[:, :, 0:1]
                g_conj = g.conj().transpose(0, 2, 1) # (NUP, 1, M)

                c = inv_Qvv @ g                        # (NUP, M, 1)
                inv_temp = (g_conj @ c) + epsilon      # (NUP, 1, 1)
                w = c / inv_temp                       # (NUP, M, 1)

                w_flat = np.squeeze(w, axis=2)
                self.W[l, :, :, 0] = w_flat
                self.W[l, :, :, 1] = 0

                s_hat_j = w.conj().transpose(0, 2, 1) @ z_frame
                s_hat_flat = np.squeeze(s_hat_j)

                s_hat[0, :] = s_hat_flat
                s_hat[1, :] = 1e-10

            # Case B2: Only Slot 1 active (Speaker 2 only)
            elif fc[0, 0] == 0 and fc[0, 1] != 0:
                g = self.G[:, :, 1:2]
                g_conj = g.conj().transpose(0, 2, 1) # (NUP, 1, M)

                c = inv_Qvv @ g                        # (NUP, M, 1)
                inv_temp = (g_conj @ c) + epsilon      # (NUP, 1, 1)
                w = c / inv_temp                       # (NUP, M, 1)

                w_flat = np.squeeze(w, axis=2)
                self.W[l, :, :, 0] = 0
                self.W[l, :, :, 1] = w_flat
                
                s_hat_j = w.conj().transpose(0, 2, 1) @ z_frame
                s_hat_flat = np.squeeze(s_hat_j)

                s_hat[0, :] = 1e-10
                s_hat[1, :] = s_hat_flat

            # Case C: Two slots -> LCMV
            elif fc[0, 0] != 0 and fc[0, 1] != 0:
                if self.flag_start_lcmv: self.flag_start_lcmv = 0

                g = self.G
                g_conj = g.conj().transpose(0, 2, 1)   # (NUP, 2, M)

                c = inv_Qvv @ g                        # (NUP, M, 2)
                term = g_conj @ c                      # (NUP, 2, 2)

                norm_term = LA.norm(term, axis=(1, 2), keepdims=True)
                reg_term = e * norm_term * np.eye(self.num_speech)
                inv_temp = LA.inv(term + reg_term)     # (NUP, 2, 2)

                w = c @ inv_temp                       # (NUP, M, 2)
                self.W[l, :, :, :] = w

                s_hat_j = w.conj().transpose(0, 2, 1) @ z_frame
                s_hat_flat = np.squeeze(s_hat_j, axis=2).T

                s_hat[0, :] = s_hat_flat[0, :]
                s_hat[1, :] = s_hat_flat[1, :]

        # Aggregate outputs
        if l == 0:
            self.s_hat_total = s_hat.T.reshape(1, self.NUP, self.num_speech)
        else:
            self.s_hat_total = np.concatenate((self.s_hat_total, s_hat.T.reshape(1, self.NUP, self.num_speech)), axis=0)

    def run_online_separation(self):
        """Executes the main online tracking and separation loop."""
        if self.verbose: print("--- Running Online Separation ---")

        for l in range(len(self.y2_prob_stat_mf)):
            if self.verbose > 1 and l % 100 == 0:
                print(f"Processing frame {l}/{len(self.y2_prob_stat_mf)}")

            y_prob = self.y_prob_stat_mf[l]

            if y_prob == 0:
                self._update_noise_covariance(l)
            elif y_prob == 1:
                self._update_rtf_and_tracking(l)
            elif y_prob == 2:
                self.first_speaker_active = 1
                self.second_speaker_active = 1
                self.time_second += 1
                self.time_first += 1

            self._compute_spatial_filters(l)

    def reconstruct_audio(self):
        """Converts filtered frequency features back to time-domain audio."""
        if self.verbose: print("--- Reconstructing Audio (ISTFT) ---")

        win = self.p_stft['win']
        hop = self.p_stft['hop']
        nfft = self.p_stft['nfft']

        self.speech_out = []
        for p in range(self.num_speech):
            speech, _ = istft(self.s_hat_total[:, :, p].T, win, win, hop, nfft, self.fs)
            self.speech_out.append(speech)

    def evaluate_and_save(self):
        """Saves outputs to disk and calculates metrics if reference files exist."""
        if self.verbose: print("--- Saving Outputs ---")

        os.makedirs(self.folder_to_results, exist_ok=True)

        for p in range(self.num_speech):
            output_path = os.path.join(self.folder_to_results, f'separating_speaker_{p}_{self.run_idx}_{self.rtf_method}.wav')
            wavfile.write(output_path, self.fs, self.speech_out[p])

        if not self.evaluate: return None, None, None

        if self.verbose: print("--- Computing Evaluation Metrics ---")
        win = self.p_stft['win']
        hop = self.p_stft['hop']
        nfft = self.p_stft['nfft']

        start_overlap = np.nonzero(self.y_mf == 2)[0][0]
        finish_overlap = np.nonzero(self.y_mf == 2)[0][-1]

        s_hat_overlap = self.s_hat_total[start_overlap:finish_overlap]
        z_first_overlap = self.z_k_first[start_overlap:finish_overlap]
        z_second_overlap = self.z_k_second[start_overlap:finish_overlap]

        s_hat_1_time, _ = istft(s_hat_overlap[:, :, 0].T, win, win, hop, nfft, self.fs)
        s_hat_2_time, _ = istft(s_hat_overlap[:, :, 1].T, win, win, hop, nfft, self.fs)
        ref_1_time, _ = istft(z_first_overlap[:, :, 0].T, win, win, hop, nfft, self.fs)
        ref_2_time, _ = istft(z_second_overlap[:, :, 0].T, win, win, hop, nfft, self.fs)

        ref_sources = np.concatenate((ref_1_time.reshape(-1, 1), ref_2_time.reshape(-1, 1)), axis=1)
        est_sources = np.concatenate((s_hat_1_time.reshape(-1, 1), s_hat_2_time.reshape(-1, 1)), axis=1)

        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(ref_sources.T + 1e-9, est_sources.T, compute_permutation=True)

        if self.verbose:
            print("\n--- Evaluation Results (During Overlap) ---")
            print(f"Speaker 0 -> SDR: {sdr[0]:.2f} dB | SIR: {sir[0]:.2f} dB | SAR: {sar[0]:.2f} dB")
            print(f"Speaker 1 -> SDR: {sdr[1]:.2f} dB | SIR: {sir[1]:.2f} dB | SAR: {sar[1]:.2f} dB")
            print("-" * 45)
            print(f"Average   -> SDR: {sdr.mean():.2f} dB | SIR: {sir.mean():.2f} dB | SAR: {sar.mean():.2f} dB")

        return sdr.mean(), sir.mean(), sar.mean()

    def investigate_geometry(self):
        """Diagnoses the acoustic geometry of the current audio file."""
        if self.verbose < 1: return
        print("\n" + "="*55)
        print(" ACOUSTIC GEOMETRY INVESTIGATION")
        print("="*55)

        # 1. DOA Indices
        single_frames_idx = np.where(self.y_prob_stat_mf == 1)[0]
        doas_single = self.y2_prob_stat_mf[single_frames_idx]

        power_0 = np.mean(np.abs(self.s_hat_total[single_frames_idx, :, 0])**2, axis=1)
        power_1 = np.mean(np.abs(self.s_hat_total[single_frames_idx, :, 1])**2, axis=1)

        doas_spk0 = doas_single[power_0 > power_1 * 10]
        doas_spk1 = doas_single[power_1 > power_0 * 10]

        unique_doas_0 = np.unique(doas_spk0[doas_spk0 > 0]).astype(int)
        unique_doas_1 = np.unique(doas_spk1[doas_spk1 > 0]).astype(int)

        print(f"    Speaker 0 sectors: {unique_doas_0}")
        print(f"    Speaker 1 sectors: {unique_doas_1}")

        # 2. Spatial Correlation
        print("\n[2] Spatial Overlap (Speaker vs. Background Noise):")
        correlation_scores = np.zeros(self.num_speech)

        for p in range(self.num_speech):
            corr_per_freq = np.zeros(self.NUP)
            for f in range(self.NUP):
                g_f = self.G[f, :, p]
                w, v = np.linalg.eig(self.Qvv[f, :, :])
                noise_vector = v[:, np.argmax(w.real)]
                numerator = np.abs(np.vdot(g_f, noise_vector))**2
                denominator = (np.linalg.norm(g_f)**2) * (np.linalg.norm(noise_vector)**2)
                corr_per_freq[f] = numerator / (denominator + 1e-15)

            correlation_scores[p] = np.mean(corr_per_freq)
            print(f"    Speaker {p} Spatial Overlap: {correlation_scores[p]:.4f}")
        print("="*55 + "\n")

    def compute_noise_reduction(self):
        """Calculates the quantitative Noise Reduction (NR) in dB."""
        if self.verbose: print("\n--- Computing Noise Reduction Metrics ---")

        noise_frames_idx = np.where(self.y_prob_stat_mf == 0)[0]
        if len(noise_frames_idx) == 0:
            if self.verbose: print("No pure noise frames found to evaluate.")
            return None, None

        power_in_frames = np.mean(np.abs(self.z_k[noise_frames_idx, :, 0])**2, axis=1)
        power_out_frames = np.zeros((self.num_speech, len(noise_frames_idx)))
        for p in range(self.num_speech):
            power_out_frames[p, :] = np.mean(np.abs(self.s_hat_total[noise_frames_idx, :, p])**2, axis=1)

        valid_idx_0 = np.where(power_out_frames[0, :] > 1e-18)[0]
        valid_idx_1 = np.where(power_out_frames[1, :] > 1e-18)[0]
        shared_valid_idx = np.intersect1d(valid_idx_0, valid_idx_1)
        target_nr = [0.0, 0.0]

        if len(shared_valid_idx) > 0:
            mean_power_in_shared = np.mean(power_in_frames[shared_valid_idx])
            for p in range(self.num_speech):
                mean_power_out_shared = np.mean(power_out_frames[p, shared_valid_idx])
                target_nr[p] = 10 * np.log10(mean_power_in_shared / (mean_power_out_shared + 1e-15))
                if self.verbose: print(f"    - Speaker {p} Target NR = {target_nr[p]:.2f} dB")
        else:
            return None, None

        return target_nr[0], target_nr[1]

    def run(self):
        """Orchestrates the separation pipeline and returns metrics."""
        print(f"\n{'='*55}")
        print(f" STARTING PIPELINE EXECUTION (Method: {self.rtf_method.upper()})")
        print(f"{'='*55}")

        total_start = time.time()

        # 1. Load Data
        self.load_data()
        
        # 2. Compute STFT
        self.compute_stft()
        
        # 3. Online Separation (Heavy lifting - we time just this for fair comparison)
        step_start = time.time()
        self.run_online_separation()
        t_sep = time.time() - step_start
        print(f"[⏱] 3. Run Online Separation:      {t_sep:>6.2f} seconds")

        # 4. Reconstruct Audio
        self.reconstruct_audio()
        
        # 5. Evaluate and Save
        if self.evaluate:
            sdr_avg, sir_avg, sar_avg = self.evaluate_and_save()
        else:
            self.evaluate_and_save()
            sdr_avg, sir_avg, sar_avg = None, None, None

        total_time = time.time() - total_start
        print(f"{'-'*55}")
        print(f" TOTAL {self.rtf_method.upper()} EXECUTION TIME:      {total_time:>6.2f} seconds")
        print(f"{'='*55}\n")

        self.investigate_geometry()
        nr_0, nr_1 = self.compute_noise_reduction()

        # We return the separation time (t_sep) as the core performance metric
        return sdr_avg, sir_avg, sar_avg, nr_0, nr_1, t_sep


if __name__ == "__main__":
    # =========================================================================
    # USER CONFIGURATION
    # =========================================================================
    # Select mode: 'gevd', 'pastd', or 'both'
    RUN_MODE = 'both'
    # =========================================================================

    py_folder = os.path.dirname(os.path.realpath(__file__))
    workspace_folder = py_folder
    folder_to_all_data = os.path.join(workspace_folder, 'data')
    folder_to_test_data = os.path.join(folder_to_all_data, 'simulated_audio', 'test', 'paperlike')
    folder_to_results = os.path.join(workspace_folder, 'pipeline_results', 'paperlike')

    p_stft = {
        'nfft': 2048,
        'wlen': 2048,
        'hop': 512,
        'NUP': 1025,
        'win': np.hamming(2048)
    }

    p_tracking = {
        'frame_before': 8,
        'frame_after': 5,
        'win_vad': np.hamming(21),
        'threshold': 40,
        'threshold_freq': 0.3,
        'threshold_chage_location': 8
    }

    p_beamforming = {
        'e': 0.01,
        'epsilon': 0.01,
        'alfa_Qvv_init': 0.99,
        'alfa_Qvv_run': 0.05,
        'buffer_size': 32,
        'beta_pastd': 0.95  # Required for PASTd
    }

    methods_to_run = ['gevd', 'pastd'] if RUN_MODE == 'both' else [RUN_MODE]
    results_summary = {}

    for method in methods_to_run:
        pipeline = SpatialSeparationPipeline(
            run_idx=1,
            p_stft=p_stft,
            p_tracking=p_tracking,
            p_beamforming=p_beamforming,
            folder_to_test_data=folder_to_test_data,
            folder_to_results=folder_to_results,
            M=4,
            verbose=1, # Setting to 1 so the console isn't flooded during comparison
            rtf_method=method
        )

        sdr, sir, sar, nr0, nr1, t_sep = pipeline.run()
        results_summary[method.upper()] = {
            'Time (sec)': t_sep,
            'SDR (dB)': sdr,
            'SIR (dB)': sir,
            'SAR (dB)': sar,
            'NR0 (dB)': nr0,
            'NR1 (dB)': nr1
        }

    # Print Comparison Table
    if len(results_summary) > 0:
        print("\n" + "="*70)
        print(" FINAL PERFORMANCE COMPARISON")
        print("="*70)
        print(f"{'Metric':<15} | " + " | ".join([f"{m:<15}" for m in results_summary.keys()]))
        print("-" * 70)
        
        metrics = ['Time (sec)', 'SDR (dB)', 'SIR (dB)', 'SAR (dB)', 'NR0 (dB)', 'NR1 (dB)']
        for metric in metrics:
            row_str = f"{metric:<15} | "
            for method in results_summary.keys():
                val = results_summary[method].get(metric)
                val_str = f"{val:.2f}" if val is not None else "N/A"
                row_str += f"{val_str:<15} | "
            print(row_str)
        print("="*70 + "\n")