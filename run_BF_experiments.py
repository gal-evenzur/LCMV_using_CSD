import os
import pandas as pd
import time
import numpy as np

# Import ONLY the beamformer pipeline
from pipeline_beamformer import SpatialSeparationPipeline

def run_beamformer_experiments(num_experiments=20, method='BOTH'):
    """
    Args:
        num_experiments (int): Amount of experiments to run.
        method (str): 'GEVD', 'PASTd', or 'BOTH' to compare them side-by-side.
    """
    py_folder = os.path.dirname(os.path.realpath(__file__))
    folder_to_test_data = os.path.join(py_folder, 'data', 'simulated_audio', 'test', 'static')
    folder_to_results = os.path.join(py_folder, 'plots')
    
    # Static configurations for the beamformer pipeline
    p_stft = {'nfft': 2048, 'wlen': 2048, 'hop': 512, 'NUP': 1025, 'win': np.hamming(2048)}
    p_tracking = {
        'frame_before': 8, 'frame_after': 5, 'win_vad': np.hamming(21), 
        'threshold': 40, 'threshold_freq': 0.3, 'threshold_chage_location': 8
    }
    
    # Added 'beta_pastd' for PASTd
    p_beamforming = {
        'e': 0.01, 'epsilon': 0.01, 'alfa_Qvv_init': 0.99, 
        'alfa_Qvv_run': 0.05, 'buffer_size': 32, 'beta_pastd': 0.95
    }

    all_results = []
    start_time = time.time()
    
    SDR_SUCCESS_THRESHOLD = 3.0 
    
    methods_to_run = ['GEVD', 'PASTd'] if method == 'BOTH' else [method]
    
    for i in range(1, num_experiments + 1):
        for m in methods_to_run:
            print(f"\n{'='*75}\n=== STARTING BEAMFORMER ({m}): EXPERIMENT {i}/{num_experiments} ===\n{'='*75}")
            
            try:
                # Step 1: Skip Neural Network (Assume estimate_DOA and estimate_CSD already exist in folder_to_results)
                print(">> Neural Network inference skipped. Loading existing tracking arrays...")
                
                # Step 2: Run LCMV/MVDR Beamformer
                print(f">> Running Beamformer & Filtering with {m}...")
                bf_pipeline = SpatialSeparationPipeline(
                    run_idx=i, p_stft=p_stft, p_tracking=p_tracking, p_beamforming=p_beamforming, 
                    folder_to_test_data=folder_to_test_data, folder_to_results=folder_to_results, 
                    M=4, verbose=0, method=m
                )
                
                # Unpack the new t_sep returned from the updated pipeline
                sdr_avg, sir_avg, sar_avg, nr_0, nr_1, t_sep = bf_pipeline.run()
                
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
                    diagnosis = f"[Cat 1] Severe Failure. Overlap Recall: {overlap_recall_pct:.1f}%"
                elif 0 <= sdr_avg < SDR_SUCCESS_THRESHOLD:
                    diagnosis = f"[Cat 2] Poor Separation. DOA Error: {doa_error_pct:.1f}%"
                else:
                    diagnosis = f"[Cat 3] Success. Overlap Recall: {overlap_recall_pct:.1f}%"

                all_results.append({
                    'Experiment': i,
                    'Method': m,
                    'Sep_Time (s)': t_sep,
                    'SDR (dB)': sdr_avg, 'SIR (dB)': sir_avg, 'SAR (dB)': sar_avg,
                    'NR Spk 0 (dB)': nr_0, 'NR Spk 1 (dB)': nr_1,
                    'Noise Contam (%)': noise_contam_pct, 'Spatial Contam (%)': spatial_contam_pct,
                    'DOA Error (%)': doa_error_pct, 'DOA Jitter (%)': doa_jitter_pct, 'Overlap Recall (%)': overlap_recall_pct,
                    'Diagnosis': diagnosis
                })
                print(f"✅ Experiment {i} ({m}) Completed.\n   -> Time: {t_sep:.2f}s | {diagnosis}")
                
            except Exception as e:
                print(f"❌ FAILED on Experiment {i} ({m}). Error: {e}")
                all_results.append({
                    'Experiment': i, 'Method': m, 'Sep_Time (s)': None,
                    'SDR (dB)': None, 'SIR (dB)': None, 'SAR (dB)': None,
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
    
    for i in range(1, num_experiments + 1):
        print(f"\n--- Experiment {i:02d} ---")
        exp_data = df_results[df_results['Experiment'] == i]
        
        for _, res in exp_data.iterrows():
            m = res['Method']
            t_sep = res['Sep_Time (s)']
            t_str = f"{t_sep:.2f}s" if t_sep is not None else "N/A"
            
            print(f"  [{m}] Runtime: {t_str}")
            
            if res['SDR (dB)'] is not None:
                print(f"         Speech Sep: SDR: {res['SDR (dB)']:6.2f} dB | SIR: {res['SIR (dB)']:6.2f} dB | SAR: {res['SAR (dB)']:6.2f} dB")
            else:
                print("         Speech Sep: Metrics unavailable")
                
            print(f"         Diagnosis : {res['Diagnosis']}\n")

    print(f"\nTotal script time elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    # You can choose 'GEVD', 'PASTd', or 'BOTH'
    run_beamformer_experiments(20, method='BOTH')