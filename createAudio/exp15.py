import os
import numpy as np

def analyze_all_runs(num_runs=20):
    # Base directory according to the paths provided
    plots_dir = '/home/goldstsb/LCMV_using_CSD/plots'
    
    print("\n" + "="*85)
    print(" BATCH DIAGNOSTIC REPORT FOR ALL EXPERIMENTS")
    print("="*85)
    
    # Table Header
    print(f"{'Run':<5} | {'True Sectors':<15} | {'Dist':<5} | {'Est. Overlap DOAs':<25} | {'Diagnosis'}")
    print("-" * 85)
    
    for run_idx in range(1, num_runs + 1):
        try:
            # Define exact file paths
            true_doa_path = os.path.join(plots_dir, f'true_DOA_{run_idx}.npy')
            est_doa_path = os.path.join(plots_dir, f'estimate_DOA_{run_idx}.npy')
            true_csd_path = os.path.join(plots_dir, f'true_CSD_{run_idx}.npy')
            
            # Check if all required files exist
            if not (os.path.exists(true_doa_path) and os.path.exists(est_doa_path) and os.path.exists(true_csd_path)):
                print(f"{run_idx:<5} | MISSING FILES")
                continue
                
            true_doa = np.load(true_doa_path)
            est_doa = np.load(est_doa_path)
            true_csd = np.load(true_csd_path)
            
            # 1. Geometry Analysis
            valid_true_doas = true_doa[(true_doa > 0) & (true_doa < 19)]
            unique_sectors = np.unique(valid_true_doas)
            
            circular_distance = "N/A"
            if len(unique_sectors) == 2:
                sec1, sec2 = unique_sectors[0], unique_sectors[1]
                distance = abs(sec1 - sec2)
                circular_distance = min(distance, 18 - distance)
            
            # 2. Overlap DOAs Analysis
            overlap_frames = np.where(true_csd == 2)[0]
            active_est_doas = []
            if len(overlap_frames) > 0:
                est_doas_in_overlap = np.unique(est_doa[overlap_frames])
                active_est_doas = [int(d) for d in est_doas_in_overlap if 0 < d < 19]
            
            # 3. Formulate Diagnosis
            diagnosis = "OK"
            if circular_distance != "N/A" and circular_distance <= 1:
                diagnosis = "ALIASING (Dist <= 1)"
            elif len(overlap_frames) == 0:
                diagnosis = "NO OVERLAP FRAMES"
            elif len(active_est_doas) == 0:
                diagnosis = "MISSED OVERLAP"
            elif len(active_est_doas) == 1:
                diagnosis = "STARVATION (Only 1 DOA)"
            elif len(active_est_doas) > 2:
                diagnosis = f"JITTER (>2 DOAs)"
            
            # 4. Format strings for table output
            sectors_str = str(unique_sectors.astype(int).tolist())
            doas_str = str(active_est_doas)
            dist_str = str(circular_distance)
            
            print(f"{run_idx:<5} | {sectors_str:<15} | {dist_str:<5} | {doas_str:<25} | {diagnosis}")
            
        except Exception as e:
            # Print a short version of the error to keep the table clean
            print(f"{run_idx:<5} | ERROR: {str(e)[:30]}")

    print("="*85 + "\n")

if __name__ == "__main__":
    analyze_all_runs(20)