from pipeline_ofer_funcs import *
import time
import os
import numpy as np
import numpy.linalg as LA
from scipy.io import wavfile
import mir_eval

"""
=========================================================================================
SPATIAL AUDIO SEPARATION PIPELINE
=========================================================================================
This module implements an offline, multi-microphone spatial audio separation system.
"""

class SpatialSeparationPipeline:
    """
    Orchestrates the entire spatial separation algorithm via a state-machine architecture.
    """
    def __init__(self, run_idx, p_stft, p_tracking, p_beamforming, folder_to_test_data, folder_to_results, M=4, num_speech=2, verbose=1, method='GEVD'):
        """
        Initializes the spatial separation pipeline.
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
        self.method = method

        # Audio & Labels
        self.fs = None
        self.z_k = None
        self.z_k_first = None
        self.z_k_second = None
        self.y_prob_stat_mf = None
        self.y2_prob_stat_mf = None
        self.y_mf = None
        self.evaluate = False

        # State Variables
        self.NUP = self.p_stft['NUP']
        self.buffer_size = self.p_beamforming['buffer_size']

        self.Qvv = np.zeros((self.NUP, self.M, self.M), dtype=complex)
        for j in range(self.NUP):
            self.Qvv[j, :, :] = np.eye(self.M)
        self.Qvv_temp = np.zeros((self.NUP, self.M, self.M), dtype=complex)

        # --- Lazy Caching Setup ---
        self.qvv_cache_dirty = True
        self.inv_Qvv = np.zeros((self.NUP, self.M, self.M), dtype=complex)
        self.chol_Qvv = np.zeros((self.NUP, self.M, self.M), dtype=complex)
        self.chol_inv = np.zeros((self.NUP, self.M, self.M), dtype=complex)
        # --------------------------

        self.G = np.ones((self.NUP, self.M, self.num_speech), dtype=complex)
        self.W = None
        self.s_hat_total = None

        self.PSD_matrix_per_DOA = np.zeros((18, self.NUP, self.M, self.M), dtype=complex)
        self.total_frame_per_DOA = np.zeros(18)
        self.stand_z = np.empty((0, self.NUP, self.M), dtype=complex)

        # PASTd State Initialization
        if self.method == 'PASTd':
            self.w_pastd = np.random.randn(18, self.NUP, self.M, 1) + 1j * np.random.randn(18, self.NUP, self.M, 1)
            self.w_pastd /= LA.norm(self.w_pastd, axis=2, keepdims=True)
            self.d_pastd = np.full((18, self.NUP, 1, 1), 1e-3, dtype=float)
            self.beta_pastd = self.p_beamforming.get('beta_pastd', 0.95)

        self.Frame_classification_system = np.zeros((2, 3))
        self.save_last_frames_first = np.zeros((self.buffer_size, self.NUP, self.M), dtype=complex)
        self.save_last_frames_doa_first = np.zeros(self.buffer_size)
        self.save_last_frames_second = np.zeros((self.buffer_size, self.NUP, self.M), dtype=complex)
        self.save_last_frames_doa_second = np.zeros(self.buffer_size)

        self.time_first = 0
        self.time_second = 0
        self.first_speaker_active = 0
        self.second_speaker_active = 0
        self.flag_first_noise = 0
        self.sum_Qvv = 0
        self.flag_start_lcmv = 1
        self.alfa_Qvv = self.p_beamforming['alfa_Qvv_init']

    def load_data(self):
        if self.verbose:
            print(f"--- Loading data for Experiment {self.run_idx} ({self.method}) ---")

        signal_file = os.path.join(self.folder_to_test_data, f'together_{self.run_idx}.wav')
        self.fs, receivers = wavfile.read(signal_file)
        self.receivers = receivers[:, :self.M] / np.max(np.abs(receivers))

        self.y2_prob_stat_mf = np.load(os.path.join(self.folder_to_results, f'estimate_DOA_{self.run_idx}.npy'))
        self.y_prob_stat_mf = np.load(os.path.join(self.folder_to_results, f'estimate_CSD_{self.run_idx}.npy'))

        try:
            _, ref_first = wavfile.read(os.path.join(self.folder_to_test_data, f'first_{self.run_idx}.wav'))
            _, ref_second = wavfile.read(os.path.join(self.folder_to_test_data, f'second_{self.run_idx}.wav'))

            self.receiver_first = ref_first[:, :self.M] / np.max(np.abs(ref_first))
            self.receiver_second = ref_second[:, :self.M] / np.max(np.abs(ref_second))

            self.y_mf = np.load(os.path.join(self.folder_to_results, f'true_CSD_{self.run_idx}.npy'))
            self.evaluate = True
            if self.verbose > 1: print("Reference files found. Evaluation metrics will be computed.")
        except FileNotFoundError:
            self.evaluate = False
            if self.verbose > 1: print("Reference files not found. Skipping evaluation metrics.")

        self.W = np.ones((len(self.y2_prob_stat_mf), self.NUP, self.M, self.num_speech), dtype=complex)

    def compute_stft(self):
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
            z = np.transpose(z, (2, 1, 0)) 
            return z[f_before:index - f_after, :, :]

        self.z_k = do_stft(self.receivers)

        if self.evaluate:
            self.z_k_first = do_stft(self.receiver_first)
            self.z_k_second = do_stft(self.receiver_second)

    def _ensure_qvv_cache(self):
        """Helper to lazily update cached matrix inversions only when needed."""
        if not self.qvv_cache_dirty:
            return

        e = self.p_beamforming['e']
        epsilon = self.p_beamforming['epsilon']
        
        norm_Qvv = LA.norm(self.Qvv, axis=(1, 2), keepdims=True)
        self.inv_Qvv = LA.inv(self.Qvv + e * norm_Qvv * np.eye(self.M))
        
        # Tiny regularization strictly to guarantee Cholesky doesn't crash on early frames
        safe_Qvv = self.Qvv + 1e-6 * norm_Qvv * np.eye(self.M)
        self.chol_Qvv = LA.cholesky(safe_Qvv)
        
        norm_chol = LA.norm(self.chol_Qvv, axis=(1, 2), keepdims=True)
        self.chol_inv = LA.inv(self.chol_Qvv + epsilon * norm_chol * np.eye(self.M))
        
        self.qvv_cache_dirty = False

    def _update_noise_covariance(self, l):
        self.time_second += 1
        self.time_first += 1

        z_frame = self.z_k[l]
        z_col = z_frame[:, :, np.newaxis]
        z_row = z_frame[:, np.newaxis, :].conj()
        Qvv_inst = z_col @ z_row

        if self.flag_first_noise == 0:
            self.flag_first_noise = 1
            self.Qvv = Qvv_inst.copy()
            self.sum_Qvv = 1
        else:
            self.sum_Qvv += 1
            self.alfa_Qvv = self.p_beamforming['alfa_Qvv_run']
            self.Qvv = (1 - self.alfa_Qvv) * self.Qvv + (self.alfa_Qvv) * Qvv_inst

        # Flag that Qvv changed, but DO NOT invert it yet!
        self.qvv_cache_dirty = True

    def _update_rtf_and_tracking(self, l):
        y2_prob = self.y2_prob_stat_mf[l]
        epsilon = self.p_beamforming['epsilon']

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

        if y2_prob == self.Frame_classification_system[0, 0]:
            self._update_slot(0, y2_prob, l, self.save_last_frames_first, self.save_last_frames_doa_first)
        elif y2_prob == self.Frame_classification_system[0, 1]:
            self._update_slot(1, y2_prob, l, self.save_last_frames_second, self.save_last_frames_doa_second)
        elif y2_prob == self.Frame_classification_system[0, 2]:
            self._process_candidate_doa(l, y2_prob)
        else:
            self.stand_z = self.z_k[l, :, :].reshape(1, self.NUP, self.M)
            self.Frame_classification_system[0, 2] = y2_prob
            self.Frame_classification_system[1, 2] = 1
            self.time_second += 1
            self.time_first += 1

    def _update_slot(self, slot_idx, y2_prob, l, save_last_frames, save_last_frames_doa):
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

        curr_frames = save_last_frames[np.where(save_last_frames_doa > 0)]
        curr_doa = save_last_frames_doa[np.where(save_last_frames_doa > 0)]
        curr_frames = curr_frames[np.where(abs(curr_doa - y2_prob) != 0)]
        curr_doa = save_last_frames_doa[np.where(abs(curr_doa - y2_prob) != 0)]
        curr_frames = curr_frames[np.where(abs(curr_doa - y2_prob) < 3)]

        self.total_frame_per_DOA[y2_prob - 1] += 1
        alfa_G = (1 + len(curr_frames)) / (self.total_frame_per_DOA[y2_prob - 1] + len(curr_frames))

        z_frame = self.z_k[l, :, :, np.newaxis]
        
        # Make sure cache is up-to-date before using it
        self._ensure_qvv_cache()

        if self.method == 'GEVD':
            if len(curr_frames) > 0:
                history_trans = curr_frames.transpose(1, 2, 0)
                all_frames = np.concatenate((z_frame, history_trans), axis=2)
            else:
                all_frames = z_frame

            a = self.chol_inv @ all_frames
            Zvv_temp = a @ a.conj().transpose(0, 2, 1)

            self.PSD_matrix_per_DOA[y2_prob - 1] = (1 - alfa_G) * self.PSD_matrix_per_DOA[y2_prob - 1] + (alfa_G) * Zvv_temp

            w_eig, v_eig = LA.eig(self.PSD_matrix_per_DOA[y2_prob - 1])
            max_idx = np.argmax(w_eig.real, axis=1)
            phi = v_eig[np.arange(self.NUP), :, max_idx][:, :, np.newaxis]
            
        elif self.method == 'PASTd':
            doa_idx = y2_prob - 1
            x_t = self.chol_inv @ z_frame
            
            w_p = self.w_pastd[doa_idx]
            d_p = self.d_pastd[doa_idx]
            
            y_proj = np.sum(w_p.conj() * x_t, axis=1, keepdims=True)
            d_p *= self.beta_pastd 
            d_p += np.abs(y_proj)**2
            gain = y_proj.conj() / d_p
            
            w_p += gain * (x_t - w_p * y_proj)
            w_p /= np.sqrt(np.sum(np.abs(w_p)**2, axis=1, keepdims=True))
            
            self.w_pastd[doa_idx] = w_p
            self.d_pastd[doa_idx] = d_p
            phi = w_p

        numerator = self.chol_Qvv @ phi 
        denominator = self.chol_Qvv[:, 0:1, :] @ phi
        self.G[:, :, slot_idx] = np.squeeze(numerator / denominator, axis=2)

    def _process_candidate_doa(self, l, y2_prob):
        epsilon = self.p_beamforming['epsilon']
        thresh = self.p_tracking['threshold_chage_location']

        self.stand_z = np.concatenate((self.stand_z, self.z_k[l, :, :].reshape(1, self.NUP, self.M)))
        self.Frame_classification_system[1, 2] += 1
        self.time_second += 1
        self.time_first += 1

        if self.Frame_classification_system[1, 2] > (thresh - 1):
            # Make sure cache is up-to-date before using it
            self._ensure_qvv_cache()

            stand_z_trans = self.stand_z.transpose(1, 2, 0)

            if self.method == 'GEVD':
                a = self.chol_inv @ stand_z_trans
                Zvv_temp = (a @ a.conj().transpose(0, 2, 1)) / thresh

                temp_alfa = self.total_frame_per_DOA[y2_prob - 1] + thresh
                self.PSD_matrix_per_DOA[y2_prob - 1] = (self.total_frame_per_DOA[y2_prob - 1] / temp_alfa) * self.PSD_matrix_per_DOA[y2_prob - 1] + (thresh / temp_alfa) * Zvv_temp
            
            elif self.method == 'PASTd':
                doa_idx = y2_prob - 1
                w_p = self.w_pastd[doa_idx]
                d_p = self.d_pastd[doa_idx]
                
                X_whitened = self.chol_inv @ stand_z_trans
                
                for t in range(thresh):
                    x_t = X_whitened[:, :, t:t+1]
                    y_proj = np.sum(w_p.conj() * x_t, axis=1, keepdims=True)
                    
                    d_p *= self.beta_pastd
                    d_p += np.abs(y_proj)**2
                    gain = y_proj.conj() / d_p
                    
                    w_p += gain * (x_t - w_p * y_proj)
                    w_p /= np.sqrt(np.sum(np.abs(w_p)**2, axis=1, keepdims=True))
                    
                self.w_pastd[doa_idx] = w_p
                self.d_pastd[doa_idx] = d_p

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
            self.stand_z = np.empty((0, self.NUP, self.M), dtype=complex)

    def _compute_spatial_filters(self, l):
        e = self.p_beamforming['e']
        epsilon = self.p_beamforming['epsilon']
        fc = self.Frame_classification_system
        s_hat = np.zeros((2, self.NUP), dtype=complex)
        z_frame = self.z_k[l, :, :, np.newaxis]

        if fc[0, 0] == 0 and fc[0, 1] == 0:
            s_hat[0, :] = self.z_k[l, :, 0]
            s_hat[1, :] = 1e-10

        elif fc[0, 0] != 0 or fc[0, 1] != 0:
            # Make sure cache is up-to-date before filtering
            self._ensure_qvv_cache()

            if fc[0, 0] != 0 and fc[0, 1] == 0:
                g = self.G[:, :, 0:1]
                g_conj = g.conj().transpose(0, 2, 1)

                c = self.inv_Qvv @ g 
                inv_temp = (g_conj @ c) + epsilon 
                w = c / inv_temp 
                
                w_flat = np.squeeze(w, axis=2)
                self.W[l, :, :, 0] = w_flat
                self.W[l, :, :, 1] = 0 

                s_hat_j = w.conj().transpose(0, 2, 1) @ z_frame
                s_hat_flat = np.squeeze(s_hat_j)
                s_hat[0, :] = s_hat_flat
                s_hat[1, :] = 1e-10

            elif fc[0, 0] == 0 and fc[0, 1] != 0:
                g = self.G[:, :, 1:2]
                g_conj = g.conj().transpose(0, 2, 1)

                c = self.inv_Qvv @ g 
                inv_temp = (g_conj @ c) + epsilon
                w = c / inv_temp 

                w_flat = np.squeeze(w, axis=2)
                self.W[l, :, :, 0] = 0
                self.W[l, :, :, 1] = w_flat
                
                s_hat_j = w.conj().transpose(0, 2, 1) @ z_frame
                s_hat_flat = np.squeeze(s_hat_j)
                s_hat[0, :] = 1e-10
                s_hat[1, :] = s_hat_flat

            elif fc[0, 0] != 0 and fc[0, 1] != 0:
                if self.flag_start_lcmv: self.flag_start_lcmv = 0

                g = self.G
                g_conj = g.conj().transpose(0, 2, 1)

                c = self.inv_Qvv @ g 
                term = g_conj @ c 

                norm_term = LA.norm(term, axis=(1, 2), keepdims=True)
                reg_term = e * norm_term * np.eye(self.num_speech)
                inv_temp = LA.inv(term + reg_term) 

                w = c @ inv_temp 
                self.W[l, :, :, :] = w

                s_hat_j = w.conj().transpose(0, 2, 1) @ z_frame
                s_hat_flat = np.squeeze(s_hat_j, axis=2).T 
                s_hat[0, :] = s_hat_flat[0, :]
                s_hat[1, :] = s_hat_flat[1, :]

        if l == 0:
            self.s_hat_total = s_hat.T.reshape(1, self.NUP, self.num_speech)
        else:
            self.s_hat_total = np.concatenate((self.s_hat_total, s_hat.T.reshape(1, self.NUP, self.num_speech)), axis=0)

    def run_online_separation(self):
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
        if self.verbose: print("--- Reconstructing Audio (ISTFT) ---")
        win = self.p_stft['win']
        hop = self.p_stft['hop']
        nfft = self.p_stft['nfft']
        self.speech_out = []
        for p in range(self.num_speech):
            speech, _ = istft(self.s_hat_total[:, :, p].T, win, win, hop, nfft, self.fs)
            self.speech_out.append(speech)

    def evaluate_and_save(self):
        if self.verbose: print("--- Saving Outputs ---")
        os.makedirs(self.folder_to_results, exist_ok=True)
        for p in range(self.num_speech):
            output_path = os.path.join(self.folder_to_results, f'separating_speaker_{p}_{self.run_idx}_{self.method}.wav')
            wavfile.write(output_path, self.fs, self.speech_out[p])

        if not self.evaluate: return
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
        pass

    def compute_noise_reduction(self):
        pass

    def run(self):
        print(f"\n{'='*55}")
        print(f" STARTING PIPELINE EXECUTION (Experiment {self.run_idx} | Method: {self.method})")
        print(f"{'='*55}")
        total_start = time.time()

        step_start = time.time()
        self.load_data()
        t_load = time.time() - step_start
        print(f"[⏱] 1. Load Data:                  {t_load:>6.2f} seconds")

        step_start = time.time()
        self.compute_stft()
        t_stft = time.time() - step_start
        print(f"[⏱] 2. Compute STFT:               {t_stft:>6.2f} seconds")

        step_start = time.time()
        self.run_online_separation()
        t_sep = time.time() - step_start
        print(f"[⏱] 3. Run Online Separation:      {t_sep:>6.2f} seconds")

        if self.verbose > 1:
            num_frames = len(self.y2_prob_stat_mf)
            time_per_frame = t_sep / num_frames
            print(f"     - Time per frame: {time_per_frame:.4f} seconds/frame")

        step_start = time.time()
        self.reconstruct_audio()
        t_recon = time.time() - step_start
        print(f"[⏱] 4. Reconstruct Audio:          {t_recon:>6.2f} seconds")

        step_start = time.time()
        if self.evaluate:
            sdr_avg, sir_avg, sar_avg = self.evaluate_and_save()
        else:
            self.evaluate_and_save()
            sdr_avg, sir_avg, sar_avg = None, None, None

        t_eval = time.time() - step_start
        print(f"[⏱] 5. Evaluate and Save:          {t_eval:>6.2f} seconds")

        total_time = time.time() - total_start
        print(f"{'-'*55}")
        print(f" TOTAL EXECUTION TIME:             {total_time:>6.2f} seconds")
        print(f"{'='*55}\n")

        return sdr_avg, sir_avg, sar_avg, 0.0, 0.0, t_sep