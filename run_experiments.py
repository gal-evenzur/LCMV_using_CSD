import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# Import the pipeline classes and configurations
from pipeline import SpatialTrackingPipeline, pipeline_config, plot_dir
from pipeline_beamformer import SpatialSeparationPipeline

def run_all_experiments(num_experiments=20):
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
        print(f"\n{'='*75}\n=== STARTING EXPERIMENT {i}/{num_experiments} ===\n{'='*75}")
        
        try:
            # Step 1: Run Neural Network Inference
            print(">> Running Neural Network Inference...")
            nn_pipeline = SpatialTrackingPipeline(
                run_idx=i, config=pipeline_config, folder_to_test_data=folder_to_test_data, n_mics=4, verbose=0
            )
            nn_pipeline.run(folder_to_results)
            
            # Step 2: Run LCMV Beamformer
            print(">> Running Beamformer & Filtering...")
            bf_pipeline = SpatialSeparationPipeline(
                run_idx=i, p_stft=p_stft, p_tracking=p_tracking, p_beamforming=p_beamforming, 
                folder_to_test_data=folder_to_test_data, folder_to_results=folder_to_results, M=4, verbose=0
            )
            
            sdr_avg, sir_avg, sar_avg, nr_0, nr_1 = bf_pipeline.run()
            
            # ---------------------------------------------------------
            # Step 3: Automated Diagnostics & Data Collection
            # ---------------------------------------------------------
            diagnosis = "Unknown"
            est_silence_count, true_silence_count = 0, 0
            
            # Metrics for graphing
            noise_contam_pct = 0.0
            spatial_contam_pct = 0.0
            doa_error_pct = 0.0
            doa_jitter_pct = 0.0
            overlap_recall_pct = 0.0  # NEW METRIC: Overlap Detection Rate
            
            try:
                # ---------------- CSD Metrics ----------------
                est_csd = np.load(os.path.join(folder_to_results, f'estimate_CSD_{i}.npy'))
                true_csd = np.load(os.path.join(folder_to_results, f'true_CSD_{i}.npy'))
                total_frames = len(true_csd)
                
                noise_contam_pct = np.sum((est_csd == 0) & (true_csd > 0)) / total_frames * 100
                spatial_contam_pct = np.sum((est_csd > 0) & (true_csd == 0)) / total_frames * 100
                
                # Overlap Recall Calculation (Did the network catch the double-talk?)
                true_overlap_idx = np.where(true_csd == 2)[0]
                if len(true_overlap_idx) > 0:
                    correct_overlap_preds = np.sum(est_csd[true_overlap_idx] == 2)
                    overlap_recall_pct = (correct_overlap_preds / len(true_overlap_idx)) * 100
                
                # ---------------- DOA Metrics ----------------
                est_doa = np.load(os.path.join(folder_to_results, f'estimate_DOA_{i}.npy'))
                true_doa = np.load(os.path.join(folder_to_results, f'true_DOA_{i}.npy'))
                
                valid_doa_idx = np.where((true_doa > 0) & (true_doa < 19))[0]
                if len(valid_doa_idx) > 0:
                    doa_error_pct = np.sum(est_doa[valid_doa_idx] != true_doa[valid_doa_idx]) / len(valid_doa_idx) * 100
                
                active_est_doa = est_doa[(est_doa > 0) & (est_doa < 19)]
                if len(active_est_doa) > 1:
                    jumps = np.sum(np.diff(active_est_doa) != 0)
                    doa_jitter_pct = (jumps / (len(active_est_doa) - 1)) * 100
                
                # ---------------- Geometry Check ----------------
                doa_1 = np.load(os.path.join(folder_to_test_data, f'label_location_first_{i}.npy'))
                doa_2 = np.load(os.path.join(folder_to_test_data, f'label_location_second_{i}.npy'))
                spk1_sectors = np.unique(doa_1[doa_1 > 0]).astype(int).tolist()
                spk2_sectors = np.unique(doa_2[doa_2 > 0]).astype(int).tolist()
                
                is_close = False
                if len(spk1_sectors) > 0 and len(spk2_sectors) > 0:
                    for s1 in spk1_sectors:
                        for s2 in spk2_sectors:
                            if abs(s1 - s2) <= 2:
                                is_close = True
                                break
                proximity_str = "Yes" if is_close else "No"
                
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

            # Append results
            all_results.append({
                'Experiment': i,
                'SDR (dB)': sdr_avg,
                'SIR (dB)': sir_avg,
                'SAR (dB)': sar_avg,
                'NR Spk 0 (dB)': nr_0,
                'NR Spk 1 (dB)': nr_1,
                'Noise Contam (%)': noise_contam_pct,
                'Spatial Contam (%)': spatial_contam_pct,
                'DOA Error (%)': doa_error_pct,
                'DOA Jitter (%)': doa_jitter_pct,
                'Overlap Recall (%)': overlap_recall_pct,
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

    # Export to CSV
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(folder_to_results, 'experiments_summary.csv')
    df_results.to_csv(csv_path, index=False)
    
    # ---------------------------------------------------------
    # PRINT CLEAN SUMMARY
    # ---------------------------------------------------------
    print("\n\n")
    print("*"*95)
    print(" 🚀 FINAL SUMMARY AND DIAGNOSTICS OF ALL EXPERIMENTS 🚀")
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

    print("*"*95)
    print(" 📊 AVERAGES ACROSS ALL RUNS (Including Failures) 📊")
    print("*"*95)
    numeric_df = df_results.drop(columns=['Experiment', 'Diagnosis'])
    print(numeric_df.mean(numeric_only=True).to_string())
    
    print("\n" + "*"*95)
    print(" ⭐ AVERAGES ACROSS VALID RUNS ONLY (Category 3) ⭐")
    print("*"*95)
    valid_df = df_results[df_results['SDR (dB)'] >= SDR_SUCCESS_THRESHOLD]
    if not valid_df.empty:
        print(valid_df.drop(columns=['Experiment', 'Diagnosis']).mean(numeric_only=True).to_string())
    else:
        print("No successful runs found.")
    print("*"*95)
    
    print(f"\nTotal time elapsed: {time.time() - start_time:.2f} seconds.")
    print(f"Detailed results saved to: {csv_path}")

    # ---------------------------------------------------------
    # PLOT GENERATION
    # ---------------------------------------------------------
    print("\nGenerating Diagnostic Plots...")
    
    valid_plot_data = [r for r in all_results if r['SDR (dB)'] is not None]
    
    if len(valid_plot_data) > 0:
        sdrs = [r['SDR (dB)'] for r in valid_plot_data]
        noise_contams = [r['Noise Contam (%)'] for r in valid_plot_data]
        spatial_contams = [r['Spatial Contam (%)'] for r in valid_plot_data]
        doa_errors = [r['DOA Error (%)'] for r in valid_plot_data]
        doa_jitters = [r['DOA Jitter (%)'] for r in valid_plot_data]
        overlap_recalls = [r['Overlap Recall (%)'] for r in valid_plot_data]
        
        # Plot 1: Category 1 (CSD Contaminations vs SDR)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.scatter(noise_contams, sdrs, color='red', alpha=0.7)
        ax1.set_title('Noise Matrix Contamination vs SDR')
        ax1.set_xlabel('% Frames (Est=0 while True>0)')
        ax1.set_ylabel('SDR (dB)')
        ax1.grid(True)
        
        ax2.scatter(spatial_contams, sdrs, color='orange', alpha=0.7)
        ax2.set_title('Spatial (RTF) Contamination vs SDR')
        ax2.set_xlabel('% Frames (Est>0 while True=0)')
        ax2.set_ylabel('SDR (dB)')
        ax2.grid(True)
        
        plt.tight_layout()
        plot1_path = os.path.join(folder_to_results, 'category1_csd_contamination.png')
        plt.savefig(plot1_path)
        plt.close()
        print(f" -> Saved Plot: {plot1_path}")
        
        # Plot 2: Category 2 (DOA Absolute Error vs DOA Jitter vs SDR)
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax3.scatter(doa_errors, sdrs, color='blue', alpha=0.7)
        ax3.set_title('DOA Absolute Error Rate vs SDR')
        ax3.set_xlabel('DOA Error Rate (%)')
        ax3.set_ylabel('SDR (dB)')
        ax3.grid(True)

        ax4.scatter(doa_jitters, sdrs, color='purple', alpha=0.7)
        ax4.set_title('DOA Jitter (Chaotic Jumps) vs SDR')
        ax4.set_xlabel('DOA Jitter Rate (%)')
        ax4.set_ylabel('SDR (dB)')
        ax4.grid(True)
        
        plt.tight_layout()
        plot2_path = os.path.join(folder_to_results, 'category2_doa_jitter.png')
        plt.savefig(plot2_path)
        plt.close()
        print(f" -> Saved Plot: {plot2_path}")
        
        # Plot 3: NEW! Overlap Recall vs SDR
        plt.figure(figsize=(8, 6))
        plt.scatter(overlap_recalls, sdrs, color='green', alpha=0.7)
        plt.title('CSD Overlap Recall vs SDR')
        plt.xlabel('Overlap Recall Rate (%) [Est=2 | True=2]')
        plt.ylabel('SDR (dB)')
        plt.grid(True)
        
        plot3_path = os.path.join(folder_to_results, 'category1_overlap_recall.png')
        plt.savefig(plot3_path)
        plt.close()
        print(f" -> Saved Plot: {plot3_path}")
        
    else:
        print("Not enough valid data to generate plots.")

def plot_single_experiment_doa_accuracy(run_idx):
    """
    Runs the pipeline for a single experiment (if results don't exist),
    filters out noise/overlap frames, and plots a 3-panel subplot:
      1. Running DOA Accuracy (Only when true_CSD == 1)
      2. True vs Estimated DOA for Speaker 1 (Active Regions)
      3. True vs Estimated DOA for Speaker 2 (Active Regions)
    """
    py_folder = os.path.dirname(os.path.realpath(__file__))
    folder_to_test_data = os.path.join(py_folder, 'data', 'simulated_audio', 'test', 'dynamic')
    plot_dir = os.path.join(py_folder, 'pipeline_results', 'dynamic')
    
    # 1. Pipeline Verification / Generation
    true_csd_path = os.path.join(plot_dir, f'true_CSD_{run_idx}.npy')
    true_doa_path = os.path.join(plot_dir, f'true_DOA_{run_idx}.npy')
    est_doa_path = os.path.join(plot_dir, f'estimate_DOA_{run_idx}.npy')
    
    if not (os.path.exists(true_csd_path) and os.path.exists(true_doa_path) and os.path.exists(est_doa_path)):
        print(f"--- Files for Experiment {run_idx} not found. Running pipeline... ---")
        pipeline = SpatialTrackingPipeline(
            config=pipeline_config, 
            folder_to_test_data=folder_to_test_data, 
            n_mics=4, 
            verbose=1
        )
        pipeline.process_single_run(run_idx, plot_dir)
    
    # 2. Load Generated Artifacts
    true_csd = np.load(true_csd_path)
    true_doa = np.load(true_doa_path)
    est_doa = np.load(est_doa_path)
    
    n_frames = len(true_csd)
    frames_x = np.arange(n_frames)
    
    # Masks based on Ground Truth CSD
    valid_mask = (true_csd == 1)
    is_correct = (true_doa == est_doa)
    
    # ---------------------------------------------------------
    # Subplot Calculation 1: Running Accuracy Metrics
    # ---------------------------------------------------------
    running_accuracy = np.zeros(n_frames)
    hits, attempts = 0, 0
    for t in range(n_frames):
        if valid_mask[t]:
            attempts += 1
            if is_correct[t]:
                hits += 1
            running_accuracy[t] = (hits / attempts) * 100
        else:
            running_accuracy[t] = np.nan

    # ---------------------------------------------------------
    # Subplot Calculations 2 & 3: Clean up Angles for Plotting
    # ---------------------------------------------------------
    # Mask out non-active segments with NaN to keep lines clean on the plot
    # The true pipeline marks overlapping frames as sentinel label '19'
    spk1_active = (true_csd == 1) & (true_doa != 19) & (true_doa != 0) 
    spk2_active = (true_csd == 1) & (true_doa != 19) & (true_doa != 0) 

    # Note: Because true_doa collapses speaker 1 & 2 together based on VAD, 
    # we copy the true tracking track directly into respective slots.
    true_doa_spk1 = np.where(spk1_active, true_doa, np.nan)
    est_doa_spk1 = np.where(spk1_active, est_doa, np.nan)

    true_doa_spk2 = np.where(spk2_active, true_doa, np.nan)
    est_doa_spk2 = np.where(spk2_active, est_doa, np.nan)

# ---------------------------------------------------------
    # Rendering Phase: 2-Panel Figure
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(f'Uncut DOA Evaluation Tracking — Experiment {run_idx}', fontsize=16, fontweight='bold', y=0.96)

    # Panel 1: Accuracy Percentage Line (Isolated to CSD==1 for analytical validity)
    axs[0].fill_between(frames_x, 0, 100, where=valid_mask, color='green', alpha=0.1, label='Evaluation Domain (CSD == 1)')
    axs[0].plot(frames_x, running_accuracy, color='darkblue', linewidth=2.5, label='Running Tracking Accuracy')
    
    correct_indices = np.where(valid_mask & is_correct)[0]
    incorrect_indices = np.where(valid_mask & ~is_correct)[0]
    axs[0].scatter(correct_indices, np.ones_like(correct_indices) * 100, color='forestgreen', marker='|', s=60, alpha=0.7)
    axs[0].scatter(incorrect_indices, np.zeros_like(incorrect_indices) * 0, color='crimson', marker='|', s=60, alpha=0.7)
    
    axs[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axs[0].set_ylim(-10, 110)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='lower left')
    axs[0].set_title('DOA Tracking Performance Metrics (Filtered Domain)')

    # Panel 2: Uncut Spatial Trajectory vs Estimate
    axs[1].plot(frames_x, true_doa, color='black', linewidth=2, linestyle='-', label='True DOA (Raw Uncut)')
    axs[1].plot(frames_x, est_doa, color='darkorange', linewidth=1.2, linestyle='--', marker='.', alpha=0.6, label='Estimated DOA')
    axs[1].axhline(y=19, color='purple', linestyle=':', alpha=0.5, label='Sentinel Overlap Value (19)')
    axs[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Silence/Noise Value (0)')
    axs[1].set_xlabel('Frame Index', fontweight='bold')
    axs[1].set_ylabel('DOA Index / Bin', fontweight='bold')
    axs[1].set_ylim(-2, 22)  # Bounds adjusted comfortably to display 0 to 19 values
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Raw Spatial Trajectory Analysis: Entire Timeline')

    # Global Formatting adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save Layout
    single_plot_path = os.path.join(plot_dir, f'DOA_Uncut_2Panel_Analysis_Exp_{run_idx}.png')
    plt.savefig(single_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved uncut 2-panel evaluation plot to: {single_plot_path}")

def run_doa_experiments(num_experiments=20, need_to_estimate_doa=False):
    py_folder = os.path.dirname(os.path.realpath(__file__))
    folder_to_test_data = os.path.join(py_folder, 'data', 'simulated_audio', 'test', 'static')
    
    workspace_dir = py_folder
    plot_dir = os.path.join(workspace_dir, 'pipeline_results', 'model_predicts')
    
    # Ensure the output directory exists
    os.makedirs(plot_dir, exist_ok=True)

# ---------------------------------------------------------
    # Phase 1: Run the Pipeline on all N Experiments (UPDATED)
    # ---------------------------------------------------------
    if need_to_estimate_doa:
        print("\n--- ESTIMATING DOA FOR ALL EXPERIMENTS ---")
        print(f"--- STARTING BATCH PROCESSING FOR {num_experiments} EXPERIMENTS ---")
        
        # Initialize the pipeline ONCE (loads models into memory)
        pipeline = SpatialTrackingPipeline(
            config=pipeline_config, 
            folder_to_test_data=folder_to_test_data, 
            n_mics=4, 
            verbose=1  # Set to 0 to mute, or 2 for deep debugging
        )
        
        # Run the batch process for the specified number of experiments
        run_indices = range(1, num_experiments + 1)
        pipeline.run_batch(run_indices=run_indices, folder_to_save=plot_dir)

    
    # ---------------------------------------------------------
    # Phase 2: Analyze DOA Results (Accuracy vs Frame)
    # ---------------------------------------------------------
    print("\n--- ANALYZING DOA RESULTS ---")
    
    # Lists to hold the loaded data from all experiments
    all_true_csd = []
    all_true_doa = []
    all_est_doa = []
    max_frames = 0
    
    # Load the results generated by the pipeline
    for i in range(1, num_experiments + 1):
        true_csd = np.load(os.path.join(plot_dir, f'true_CSD_{i}.npy'))
        true_doa = np.load(os.path.join(plot_dir, f'true_DOA_{i}.npy'))
        est_doa = np.load(os.path.join(plot_dir, f'estimate_DOA_{i}.npy'))
        
        all_true_csd.append(true_csd)
        all_true_doa.append(true_doa)
        all_est_doa.append(est_doa)
        
        if len(true_csd) > max_frames:
            max_frames = len(true_csd)

    # Arrays to accumulate hits and total valid attempts per frame index
    correct_per_frame = np.zeros(max_frames)
    valid_attempts_per_frame = np.zeros(max_frames)
    
    # Calculate accuracy frame-by-frame across all 20 experiments
    for exp_idx in range(num_experiments):
        n_frames_in_exp = len(all_true_csd[exp_idx])
        
        for t in range(n_frames_in_exp):
            # CRITICAL: Only evaluate DOA when exactly 1 speaker is active
            if all_true_csd[exp_idx][t] == 1:
                valid_attempts_per_frame[t] += 1
                
                # Check for an exact match in the DOA bin
                if all_true_doa[exp_idx][t] == all_est_doa[exp_idx][t]:
                    correct_per_frame[t] += 1

    # Calculate final accuracy percentage, protecting against division by zero
    accuracy_per_frame = np.zeros(max_frames)
    valid_indices = valid_attempts_per_frame > 0
    
    accuracy_per_frame[valid_indices] = (
        correct_per_frame[valid_indices] / valid_attempts_per_frame[valid_indices]
    ) * 100

    # ---------------------------------------------------------
    # Phase 3: Plot the Results
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # We only want to plot the line where valid frames actually existed
    frames_x = np.arange(max_frames)[valid_indices]
    acc_y = accuracy_per_frame[valid_indices]
    
    plt.plot(frames_x, acc_y, color='b', marker='.', linestyle='-', alpha=0.7, label='Average Accuracy')
    
    plt.title(f'DOA Estimation Accuracy vs. Frame Index\n(Averaged over {num_experiments} Experiments, Single-Speaker Frames Only)')
    plt.xlabel('Frame Index')
    plt.ylabel('Accuracy (%)')
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save and display
    analysis_plot_path = os.path.join(plot_dir, 'DOA_Accuracy_vs_Frame.png')
    plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated analysis plot to: {analysis_plot_path}")
    

if __name__ == "__main__":
    for run_idx in range(1, 6):
        plot_single_experiment_doa_accuracy(run_idx)
