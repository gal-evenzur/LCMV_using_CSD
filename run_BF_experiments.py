import os
import pandas as pd
import time
import numpy as np

# Import ONLY the beamformer pipeline
from pipeline_beamformer import SpatialSeparationPipeline

def run_beamformer_experiments(num_experiments=20):
    py_folder = os.path.dirname(os.path.realpath(__file__))
    folder_to_test_data = os.path.join(py_folder, 'data', 'simulated_audio', 'test', 'static')
    folder_to_results = os.path.join(py_folder, 'plots')
    
    # Static configurations for the beamformer pipeline
    p_stft = {'nfft': 2048, 'wlen': 2048, 'hop': 512, 'NUP': 1025, 'win': np.hamming(2048)}
    p_tracking = {
        'frame_before': 8, 'frame_after': 5, 'win_vad': np.hamming(21), 
        'threshold': 40, 'threshold_freq': 0.3, 'threshold_chage_location': 8
    }
    p_beamforming = {'e': 0.01, 'epsilon': 0.01, 'alfa_Qvv_init': 0.99, 'alfa_Qvv_run': 0.05, 'buffer_size': 32}

    all_results = []
    start_time = time.time()
    
    SDR_SUCCESS_THRESHOLD = 3.0 
    
    for i in range(1, num_experiments + 1):
        print(f"\n{'='*75}\n=== STARTING BEAMFORMER ONLY: EXPERIMENT {i}/{num_experiments} ===\n{'='*75}")
        
        try:
            # Step 1: Skip Neural Network (Assume estimate_DOA and estimate_CSD already exist in folder_to_results)
            print(">> Neural Network inference skipped. Loading existing tracking arrays...")
            
            # Step 2: Run LCMV/MVDR Beamformer
            print(">> Running Beamformer & Filtering...")
            bf_pipeline = SpatialSeparationPipeline(
                run_idx=i, p_stft=p_stft, p_tracking=p_tracking, p_beamforming=p_beamforming, 
                folder_to_test_data=folder_to_test_data, folder_to_results=folder_to_results, M=4, verbose=0
            )
            
            sdr_avg, sir_avg, sar_avg, nr_0, nr_1 = bf_pipeline.run()
            
            # Step 3: Diagnostics Collection
            diagnosis = "Unknown"
            noise_contam_pct = 0.0
            spatial_contam_pct = 0.0
            doa_error_pct = 0.0
            doa_jitter_pct = 0.0
            overlap_recall_pct = 0.0 
            
            try:
                # Diagnostics calculations
                est_csd = np.load(os.path.join(folder_to_results, f'estimate_CSD_{i}.npy'))
                true_csd = np.load(os.path.join(folder_to_results, f'true_CSD_{i}.npy'))
                total_frames = len(true_csd)
                
                noise_contam_pct = np.sum((est_csd == 0) & (true_csd > 0)) / total_frames * 100
                spatial_contam_pct = np.sum((est_csd > 0) & (true_csd == 0)) / total_frames * 100
                
                true_overlap_idx = np.where(true_csd == 2)[0]
                if len(true_overlap_idx) > 0:
                    correct_overlap_preds = np.sum(est_csd[true_overlap_idx] == 2)
                    overlap_recall_pct = (correct_overlap_preds / len(true_overlap_idx)) * 100
                
                est_doa = np.load(os.path.join(folder_to_results, f'estimate_DOA_{i}.npy'))
                true_doa = np.load(os.path.join(folder_to_results, f'true_DOA_{i}.npy'))
                
                valid_doa_idx = np.where((true_doa > 0) & (true_doa < 19))[0]
                if len(valid_doa_idx) > 0:
                    doa_error_pct = np.sum(est_doa[valid_doa_idx] != true_doa[valid_doa_idx]) / len(valid_doa_idx) * 100
                
                active_est_doa = est_doa[(est_doa > 0) & (est_doa < 19)]
                if len(active_est_doa) > 1:
                    jumps = np.sum(np.diff(active_est_doa) != 0)
                    doa_jitter_pct = (jumps / (len(active_est_doa) - 1)) * 100
                    
            except Exception as e:
                print(f"Error calculating diagnostics: {e}")

            # Formulate Diagnosis
            if sdr_avg is None:
                diagnosis = "Pipeline crashed. No metrics generated."
            elif sdr_avg < 0:
                diagnosis = f"[Cat 1] Severe Failure. Overlap Recall: {overlap_recall_pct:.1f}% | Noise Contam: {noise_contam_pct:.1f}%"
            elif 0 <= sdr_avg < SDR_SUCCESS_THRESHOLD:
                diagnosis = f"[Cat 2] Poor Separation. DOA Error: {doa_error_pct:.1f}% | DOA Jitter (Jumps): {doa_jitter_pct:.1f}%"
            else:
                diagnosis = f"[Cat 3] Success. Overlap Recall: {overlap_recall_pct:.1f}% | DOA Jitter: {doa_jitter_pct:.1f}%"

            all_results.append({
                'Experiment': i,
                'SDR (dB)': sdr_avg, 'SIR (dB)': sir_avg, 'SAR (dB)': sar_avg,
                'NR Spk 0 (dB)': nr_0, 'NR Spk 1 (dB)': nr_1,
                'Noise Contam (%)': noise_contam_pct, 'Spatial Contam (%)': spatial_contam_pct,
                'DOA Error (%)': doa_error_pct, 'DOA Jitter (%)': doa_jitter_pct, 'Overlap Recall (%)': overlap_recall_pct,
                'Diagnosis': diagnosis
            })
            print(f"✅ Experiment {i} Completed.\n   -> {diagnosis}")
            
        except Exception as e:
            print(f"❌ FAILED on Experiment {i}. Error: {e}")
            all_results.append({
                'Experiment': i, 'SDR (dB)': None, 'SIR (dB)': None, 'SAR (dB)': None,
                'NR Spk 0 (dB)': None, 'NR Spk 1 (dB)': None, 
                'Noise Contam (%)': 0, 'Spatial Contam (%)': 0, 
                'DOA Error (%)': 0, 'DOA Jitter (%)': 0, 'Overlap Recall (%)': 0,
                'Diagnosis': 'Pipeline Crash'
            })

    # Summary
    df_results = pd.DataFrame(all_results)
    
    print("\n\n" + "*"*95)
    print(" 🚀 FINAL SUMMARY OF BEAMFORMER RUNS 🚀")
    print("*"*95)
    
    for res in all_results:
        print(f"--- Experiment {res['Experiment']:02d} ---")
        if res['NR Spk 0 (dB)'] is not None and res['NR Spk 1 (dB)'] is not None:
             print(f"  Noise Reduction   : Speaker 0: {res['NR Spk 0 (dB)']:6.2f} dB | Speaker 1: {res['NR Spk 1 (dB)']:6.2f} dB")
        else:
             print("  Noise Reduction   : Metrics unavailable (Shared valid frames not found)")
             
        if res['SDR (dB)'] is not None:
             print(f"  Speech Separation : SDR: {res['SDR (dB)']:6.2f} dB | SIR: {res['SIR (dB)']:6.2f} dB | SAR: {res['SAR (dB)']:6.2f} dB")
        else:
             print("  Speech Separation : Metrics unavailable")
             
        print(f"  Diagnosis         : {res['Diagnosis']}\n")

    print(f"\nTotal time elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_beamformer_experiments(20)