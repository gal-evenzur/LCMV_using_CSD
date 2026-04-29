from pipeline_ofer_funcs import *


class SpatialSeparationPipeline:
    def __init__(self, run_idx, p_stft, p_tracking, p_beamforming, folder_to_test_data, folder_to_results, M=4, num_speech=2, verbose=1):
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
            print(f"--- Loading data for Experiment {self.run_idx} ---")

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
        """Handles CSD=0 (Noise-only frames)"""
        self.time_second += 1
        self.time_first += 1

        for j in range(self.NUP):
            self.Qvv_temp[j, :, :] = self.z_k[l, j, :].reshape(self.M, 1) @ (self.z_k[l, j, :].reshape(self.M, 1).conj().T)

        if self.flag_first_noise == 0:
            self.flag_first_noise = 1
            self.Qvv = self.Qvv_temp.copy()
            self.sum_Qvv = 1
        else:
            self.sum_Qvv += 1
            self.alfa_Qvv = self.p_beamforming['alfa_Qvv_run']
            self.Qvv = (1 - self.alfa_Qvv) * self.Qvv + (self.alfa_Qvv) * self.Qvv_temp

    def _update_rtf_and_tracking(self, l):
        """Handles CSD=1 (Single speaker active). Manages slots and GEVD."""
        y2_prob = self.y2_prob_stat_mf[l]
        epsilon = self.p_beamforming['epsilon']

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
        """Helper to compute GEVD RTF for an active slot."""
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

        for j in range(self.NUP):
            if len(curr_frames):
                current_f = np.concatenate((self.z_k[l, j, :].reshape(1, self.M).T, curr_frames[:, j, :].reshape(len(curr_frames), self.M).T), axis=1)
            else:
                current_f = self.z_k[l, j, :].reshape(1, self.M).T

            cholesky_Qvv = LA.cholesky(self.Qvv[j, :, :])
            chol_j = LA.inv(cholesky_Qvv + epsilon * np.eye(self.M) * LA.norm(cholesky_Qvv))
            a = chol_j @ current_f
            Zvv_temp = a @ a.conj().T
            self.PSD_matrix_per_DOA[y2_prob - 1, j, :, :] = (1 - alfa_G) * self.PSD_matrix_per_DOA[y2_prob - 1, j, :, :] + (alfa_G) * Zvv_temp

            w, v = LA.eig(self.PSD_matrix_per_DOA[y2_prob - 1, j, :, :])
            phi = v[:, w.argmax()].reshape(self.M, 1)
            denominator = cholesky_Qvv[0, :].reshape(1, self.M) @ phi
            self.G[j, :, slot_idx] = np.squeeze(cholesky_Qvv @ phi / denominator)

    def _process_candidate_doa(self, l, y2_prob):
        """Helper to promote a candidate DOA to an active slot."""
        epsilon = self.p_beamforming['epsilon']
        thresh = self.p_tracking['threshold_chage_location']

        self.stand_z = np.concatenate((self.stand_z, self.z_k[l, :, :].reshape(1, self.NUP, self.M)))
        self.Frame_classification_system[1, 2] += 1
        self.time_second += 1
        self.time_first += 1

        if self.Frame_classification_system[1, 2] > (thresh - 1):
            for j in range(self.NUP):
                cholesky_Qvv = LA.cholesky(self.Qvv[j, :, :])
                chol_j = LA.inv(cholesky_Qvv + epsilon * np.eye(self.M) * LA.norm(cholesky_Qvv))
                a = chol_j @ self.stand_z[:, j, :].T
                Zvv_temp = a @ a.conj().T / thresh
                temp_alfa = self.total_frame_per_DOA[y2_prob - 1] + thresh
                self.PSD_matrix_per_DOA[y2_prob - 1, j, :, :] = self.total_frame_per_DOA[y2_prob - 1] / temp_alfa * self.PSD_matrix_per_DOA[y2_prob - 1, j, :, :] + thresh / temp_alfa * Zvv_temp

            # Slot assignment policy
            fc = self.Frame_classification_system
            if fc[1, 0] == 0:
                to_change = 0
            elif (fc[1, 1] == 0) and (abs(fc[0, 0] - y2_prob) < 4):
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

    def _compute_spatial_filters(self, l):
        """Applies MVDR or LCMV based on tracked active slots."""
        e = self.p_beamforming['e']
        epsilon = self.p_beamforming['epsilon']
        fc = self.Frame_classification_system

        # Case A: No slots -> Ref mic placeholder
        if fc[0, 0] == 0 and fc[0, 1] == 0:
            s_hat = self.z_k[l, :, 0]
            s_hat = np.concatenate((s_hat.reshape(1, self.NUP), 1e-10 * np.ones((1, self.NUP))))

        # Case B: One slot -> MVDR
        elif fc[0, 0] != 0 and fc[0, 1] == 0:
            s_hat = np.zeros(self.NUP, dtype=complex)
            for j in range(self.NUP):
                g = self.G[j, :, 0]
                inv_Qvv = LA.inv(self.Qvv[j, :, :] + e * LA.norm(self.Qvv[j, :, :]) * np.eye(self.M))
                c = inv_Qvv @ g
                inv_temp = g.conj().T @ c + epsilon
                self.W[l, j, :, 0] = c / inv_temp
                self.W[l, j, :, 1] = c / inv_temp
                s_hat[j] = self.W[l, j, :, 0].conj().T @ self.z_k[l, j, :]
            s_hat = np.concatenate((s_hat.reshape(1, self.NUP), s_hat.reshape(1, self.NUP)))

        # Case C: Two slots -> LCMV
        elif fc[0, 0] != 0 and fc[0, 1] != 0:
            if self.flag_start_lcmv: self.flag_start_lcmv = 0
            s_hat = np.zeros((2, self.NUP), dtype=complex)
            for j in range(self.NUP):
                g = self.G[j, :, :]
                inv_b = LA.inv(self.Qvv[j, :, :] + e * LA.norm(self.Qvv[j, :, :]) * np.eye(self.M))
                c = inv_b @ g
                inv_temp = LA.inv(g.conj().T @ c + e * LA.norm(g.conj().T @ c) * np.eye(self.num_speech))
                self.W[l, j, :, :] = c @ inv_temp
                s_hat[:, j] = self.W[l, j, :, :].conj().T @ self.z_k[l, j, :]
            s_hat[0, :] = s_hat[0, :] * self.first_speaker_active
            s_hat[1, :] = s_hat[1, :] * self.second_speaker_active

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

        # Ensure the results folder exists
        os.makedirs(self.folder_to_results, exist_ok=True)

        # Save separated sources
        for p in range(self.num_speech):
            output_path = os.path.join(self.folder_to_results, f'separating_speaker_{p}_{self.run_idx}.wav')
            wavfile.write(output_path, self.fs, self.speech_out[p])

        if not self.evaluate: return

        if self.verbose: print("--- Computing Evaluation Metrics ---")
        win = self.p_stft['win']
        hop = self.p_stft['hop']
        nfft = self.p_stft['nfft']

        # Isolate overlap interval
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

        # BSS Eval
        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(ref_sources.T + 1e-9, est_sources.T, compute_permutation=True)

        if self.verbose:
            print("\n--- Evaluation Results ---")
            print(f"SDR: {sdr.mean():.2f} dB | SIR: {sir.mean():.2f} dB")

    def run(self):
        """Orchestrates the separation pipeline."""
        self.load_data()
        self.compute_stft()
        self.run_online_separation()
        self.reconstruct_audio()
        self.evaluate_and_save()
        if self.verbose: print(f"--- Pipeline Execution Complete for Experiment {self.run_idx} ---")
        return self

if __name__ == "__main__":
    # Dynamically locate the workspace
    py_folder = os.path.dirname(os.path.realpath(__file__))
    workspace_folder = py_folder
    folder_to_all_data = os.path.join(workspace_folder, 'data')
    folder_to_test_data = os.path.join(folder_to_all_data, 'simulated_audio', 'test')

    # Define where the tracking pipeline saved its labels, and where we will save the separated audio
    folder_to_results = os.path.join(workspace_folder, 'plots')

    # Configuration objects (same as before)
    p_stft = {
        'nfft': 2048,
        'wlen': 2048,
        'hop': 512,  # wlen / 4
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
        'buffer_size': 32
    }

    # Instantiate and run
    pipeline = SpatialSeparationPipeline(
        run_idx=1,
        p_stft=p_stft,
        p_tracking=p_tracking,
        p_beamforming=p_beamforming,
        folder_to_test_data=folder_to_test_data,
        folder_to_results=folder_to_results,
        M=4,
        verbose=2
    )

    pipeline.run()