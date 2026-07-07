import os
import numpy as np

# Setup absolute workspace paths
# מזהה שהסקריפט בתוך plots וקופץ רמה אחת למעלה לתיקייה הראשית
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_folder = os.path.dirname(script_dir) 

folder_to_results = os.path.join(workspace_folder, 'plots')
folder_to_test_data = os.path.join(workspace_folder, 'data', 'simulated_audio', 'test', 'static')

run_idx = 15
print(f"=========================================================")
print(f" TIMELINE ANALYSIS FOR RUN {run_idx:02d}")
print(f"=========================================================")

# 1. Load ground truth metadata to isolate Speaker 2's solo segment
metadata_path = os.path.join(folder_to_test_data, f'metadata_{run_idx}.npz')
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file missing at: {metadata_path}")

meta = np.load(metadata_path)
spk1_sector = int(meta['true_sector_spk1'])
spk2_sector = int(meta['true_sector_spk2'])

# 2. Load the network predictions and ground truths
try:
    true_csd = np.load(os.path.join(folder_to_results, f'true_CSD_{run_idx}.npy'))
    est_csd = np.load(os.path.join(folder_to_results, f'estimate_CSD_{run_idx}.npy'))
    true_doa = np.load(os.path.join(folder_to_results, f'true_DOA_{run_idx}.npy'))
    est_doa = np.load(os.path.join(folder_to_results, f'estimate_DOA_{run_idx}.npy'))
except Exception as e:
    raise RuntimeError(f"Failed to load npy arrays: {e}")

# 3. Isolate the exact frames of the "Speaker 2 Alone" segment
# In ground truth, Speaker 2 is alone when CSD == 1 and the true DOA matches Speaker 2's sector
spk2_alone_indices = np.where((true_csd == 1) & (true_doa == spk2_sector))[0]

if len(spk2_alone_indices) == 0:
    print("[Error] Could not find any ground-truth frames for Speaker 2 Solo.")
    exit()

total_solo_frames = len(spk2_alone_indices)
start_frame = spk2_alone_indices[0]
end_frame = spk2_alone_indices[-1]

print(f"Speaker 2 Solo Segment Window: Frame {start_frame} to Frame {end_frame} (Total: {total_solo_frames} frames)")
print(f"True Target Geometry -> Speaker 1: Sector {spk1_sector} | Speaker 2: Sector {spk2_sector}\n")

# 4. Analyze network behavior inside this critical timeline window
csd_0_count = np.sum(est_csd[spk2_alone_indices] == 0)
csd_1_count = np.sum(est_csd[spk2_alone_indices] == 1)
csd_2_count = np.sum(est_csd[spk2_alone_indices] == 2)

print(f"--- Network Predictions During Speaker 2 Solo ---")
print(f"  Predicted as Noise (CSD=0)   : {csd_0_count} frames ({(csd_0_count/total_solo_frames)*100:.1f}%)")
print(f"  Predicted as Single (CSD=1)  : {csd_1_count} frames ({(csd_1_count/total_solo_frames)*100:.1f}%)")
print(f"  Predicted as Overlap (CSD=2) : {csd_2_count} frames ({(csd_2_count/total_solo_frames)*100:.1f}%) [CRITICAL]")

# 5. Simulate the state-machine candidate tracker frame-by-frame
# Let's trace if the candidate counter fc[1, 2] ever hits threshold = 8
candidate_doa_slot = 0
candidate_counter = 0
max_reached_counter = 0
slot0_doa = spk1_sector  # Assume Slot 0 successfully locked onto Speaker 1 earlier
slot1_doa = 0            # Target slot to initialize
slot1_initialized_at = -1

for idx in spk2_alone_indices:
    y_prob = est_csd[idx]
    y2_prob = est_doa[idx]
    
    if y_prob == 2:
        # State machine logic: CSD=2 completely SKIPS tracking updates.
        # The candidate tracking logic freezes or sits idle.
        continue
        
    elif y_prob == 1:
        # Inside _update_rtf_and_tracking:
        if y2_prob == slot0_doa:
            # Matches Slot 0, resets candidate branch
            candidate_doa_slot = 0
            candidate_counter = 0
        elif slot1_doa != 0 and y2_prob == slot1_doa:
            # Already initialized, resets candidate branch
            candidate_doa_slot = 0
            candidate_counter = 0
        elif y2_prob == candidate_doa_slot:
            # Matches existing candidate! Increment persistence counter
            candidate_counter += 1
            max_reached_counter = max(max_reached_counter, candidate_counter)
            
            if candidate_counter >= 8:  # thresh = 8
                slot1_doa = y2_prob
                slot1_initialized_at = idx
                break  # Successfully initialized Slot 1!
        else:
            # Starts tracking a brand new candidate DOA, overwriting the previous one
            candidate_doa_slot = y2_prob
            candidate_counter = 1
            max_reached_counter = max(max_reached_counter, candidate_counter)

print(f"\n--- State-Machine Tracking Simulation Results ---")
print(f"  Peak consecutive frames reached by a single candidate DOA (fc[1,2]): {max_reached_counter} / 8")

if slot1_doa != 0:
    print(f"  STATUS: SUCCESS! Slot 1 initialized at frame {slot1_initialized_at} with Sector {slot1_doa}")
else:
    print(f"  STATUS: FAILED! Slot 1 remained uninitialized (0) at the end of the solo segment.")
    print(f"  CONCLUSION: Deterministic proof of the hypothesis. False CSD=2 or DOA Jitter blocked initialization.")
print(f"=========================================================")