import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Aggregate and plot DOA performance across T60 environments.")
    parser.add_argument("--t60_list", type=float, nargs='+', default=[0.3, 0.4, 0.6], 
                        help="List of T60 values to include in the plot.")
    parser.add_argument("--data_dir", type=str, default="data/simulated_audio/test/dynamic", 
                        help="Directory containing the output arrays and metadata.")
    parser.add_argument("--save_dir", type=str, default="pipeline_results/dynamic", 
                        help="Directory to save the final plot.")
    return parser.parse_args()

def process_metrics(args):
    """Scans the directory, calculates errors, and aggregates counts per T60."""
    print("--- Aggregating Performance Metrics ---")
    
    # Initialize storage for counts
    # Using rounded T60 keys to avoid float precision mismatch
    target_t60s = [round(t, 2) for t in args.t60_list]
    metrics = {t: {'success': 0, 'low': 0, 'high': 0, 'total': 0} for t in target_t60s}
    
    metadata_files = glob.glob(os.path.join(args.data_dir, 'metadata_*.npz'))
    
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {args.data_dir}. Did the pipeline run?")

    for meta_path in metadata_files:
        # Extract the run index from the filename (e.g., metadata_1.npz -> 1)
        base_name = os.path.basename(meta_path)
        run_idx = base_name.split('_')[1].split('.')[0]
        
        # Load metadata to check T60
        meta = np.load(meta_path)
        t60 = round(float(meta['T60']), 2)
        
        if t60 not in metrics:
            continue  # Skip files that aren't in our target T60 list

        # Define paths for the tracking arrays
        true_csd_path = os.path.join(args.save_dir, f'true_CSD_{run_idx}.npy')
        true_doa_path = os.path.join(args.save_dir, f'true_DOA_{run_idx}.npy')
        est_doa_path = os.path.join(args.save_dir, f'estimate_DOA_{run_idx}.npy')

        if not (os.path.exists(true_csd_path) and os.path.exists(true_doa_path) and os.path.exists(est_doa_path)):
            print(f"Warning: Missing tracking arrays for Experiment {run_idx}. Skipping.")
            continue

        # Load arrays
        true_csd = np.load(true_csd_path)
        true_doa = np.load(true_doa_path)
        est_doa = np.load(est_doa_path)

        # 1. Filter: Keep only frames where a single speaker is active (CSD == 1)
        # and ignore the Sentinel overlap value (19) or silence (0)
        valid_mask = (true_csd == 1) & (true_doa != 0) & (true_doa != 19)
        
        valid_true_doa = true_doa[valid_mask]
        valid_est_doa = est_doa[valid_mask]

        if len(valid_true_doa) == 0:
            continue

        # 2. Compute absolute bin differences
        abs_diff = np.abs(valid_true_doa - valid_est_doa)

        # 3. Categorize errors (Resolution = 10 degrees, so 2 bins = 20 degrees)
        success = np.sum(abs_diff == 0)
        low_error = np.sum((abs_diff > 0) & (abs_diff <= 2))
        high_error = np.sum(abs_diff > 2)

        # 4. Aggregate
        metrics[t60]['success'] += success
        metrics[t60]['low'] += low_error
        metrics[t60]['high'] += high_error
        metrics[t60]['total'] += (success + low_error + high_error)

    return metrics, target_t60s

def plot_stacked_bar(metrics, t60_list, save_dir):
    """Renders and saves the exact stacked bar chart requested."""
    print("--- Generating Stacked Bar Chart ---")
    
    # Calculate percentages summing to 100%
    success_pct = []
    low_pct = []
    high_pct = []
    labels = []

    for t60 in sorted(t60_list):
        data = metrics[t60]
        total = data['total']
        
        if total == 0:
            print(f"Warning: No valid frames found for T60 = {t60}")
            success_pct.append(0)
            low_pct.append(0)
            high_pct.append(0)
        else:
            success_pct.append((data['success'] / total) * 100)
            low_pct.append((data['low'] / total) * 100)
            high_pct.append((data['high'] / total) * 100)
        
        labels.append(f"{t60:.2f}")

    # Set up the plot aesthetics
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bar_width = 0.35
    x_pos = np.arange(len(labels))
    
    # DNN Color scheme extracted from the paper
    color_success = '#3b68e0'  # Royal Blue
    color_low = '#6f42c1'      # Dark Purple
    color_high = '#e88295'     # Pink

    # Create the stacked bars
    bars_success = ax.bar(x_pos, success_pct, bar_width, 
                          label='Successful Estimation - DNN model', color=color_success)
    bars_low = ax.bar(x_pos, low_pct, bar_width, bottom=success_pct, 
                      label='Low Estimation Error - DNN model', color=color_low)
    
    # Calculate the bottom for the third stack
    bottom_high = [i + j for i, j in zip(success_pct, low_pct)]
    bars_high = ax.bar(x_pos, high_pct, bar_width, bottom=bottom_high, 
                       label='High Estimation Error - DNN model', color=color_high)

    # Formatting axes and layout to match the paper
    ax.set_xlabel('T60 (sec)', fontsize=12, labelpad=10)
    ax.set_ylabel('Percentage success rate', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    
    # Move legend above the plot, centered
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, 
              frameon=True, fontsize=10)

    # Clean borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.tight_layout()
    
    # Save output
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, 'DOA_Performance_DNN_Comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved successfully to: {output_path}")


if __name__ == "__main__":
    args = get_args()
    
    # Run metric aggregation
    compiled_metrics, sorted_t60s = process_metrics(args)
    
    # Generate the visualization
    plot_stacked_bar(compiled_metrics, sorted_t60s, args.save_dir)

    # Save the metrics for future reference
    metrics_save_path = os.path.join(args.save_dir, 'DOA_Performance_Metrics.npz')
    # metrics is a dictionary of dictionaries, so we need to convert it to a structured array for saving
    structured_metrics = {f"T60_{t60:.2f}": np.array([compiled_metrics[t60]['success'], 
                                                      compiled_metrics[t60]['low'],
                                                      compiled_metrics[t60]['high'],
                                                      compiled_metrics[t60]['total']])
                          for t60 in compiled_metrics}
    np.savez(metrics_save_path, **structured_metrics)
    print(f"Metrics saved successfully to: {metrics_save_path}")