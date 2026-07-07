import os
import numpy as np

# Set paths based on your standard workspace structure
# Assumes this script is running from the main project directory
workspace_folder = os.path.dirname(os.path.abspath(__file__))
folder_to_results = os.path.join(workspace_folder, 'plots')
folder_to_test_data = os.path.join(workspace_folder, 'data', 'simulated_audio', 'test', 'static')

def analyze_experiment_diagnostics(run_idx):
    print(f"\n=== Diagnostics for Run {run_idx:02d} ===")
    
    # 1. Check true geometry from metadata
    metadata_path = os.path.join(folder_to_test_data, f'metadata_{run_idx}.npz')
    if os.path.exists(metadata_path):
        meta = np.load(metadata_path)
        spk1_sector = int(meta['true_sector_spk1'])
        spk2_sector = int(meta['true_sector_spk2'])
        true_dist = abs(spk1_sector - spk2_sector)
        print(f"  [Geometry] Speaker 1 Sector: {spk1_sector:02d} | Speaker 2 Sector: {spk2_sector:02d}")
        print(f"  [Geometry] True angular distance (sectors): {true_dist}")
    else:
        print(f"  [Geometry] Metadata file not found at: {metadata_path}")

    # 2. Check neural network predictions and state machine behavior
    try:
        true_csd = np.load(os.path.join(folder_to_results, f'true_CSD_{run_idx}.npy'))
        est_csd = np.load(os.path.join(folder_to_results, f'estimate_CSD_{run_idx}.npy'))
        true_doa = np.load(os.path.join(folder_to_results, f'true_DOA_{run_idx}.npy'))
        est_doa = np.load(os.path.join(folder_to_results, f'estimate_DOA_{run_idx}.npy'))
        
        # Check frames where only speaker 1 is active (according to Ground Truth)
        single_spk_frames = np.where(true_csd == 1)[0]
        
        if len(single_spk_frames) > 0:
            # When did the network jitter or fail by >= 2 sectors during single speaker activity?
            errors_above_2 = np.sum(np.abs(est_doa[single_spk_frames] - true_doa[single_spk_frames]) >= 2)
            pct_errors = (errors_above_2 / len(single_spk_frames)) * 100
            
            print(f"  [NN Predict] Single speaker frames: {len(single_spk_frames)}")
            print(f"  [NN Predict] Frames with DOA error >= 2: {errors_above_2} ({pct_errors:.1f}%)")
            
            # Check if the network prematurely predicted 2 speakers (CSD=2)
            premature_csd2 = np.sum(est_csd[single_spk_frames] == 2)
            print(f"  [NN Predict] Frames with premature overlap prediction (CSD=2): {premature_csd2}")
            
    except Exception as e:
        print(f"  [Error] Failed to load DOA/CSD files: {e}")

# Test the problematic runs along with a successful one (run 1)
runs_to_test = range(1, 21)
for r in runs_to_test:
    analyze_experiment_diagnostics(r)