import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pipeline_ofer_funcs import plot_confusion_matrix_from_data


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
    metrics = {t: {'success': 0, 'low': 0, 'high': 0, 'total': 0, 'raw_errors': []} for t in target_t60s}

    # Global CSD tracking
    all_true_csd = []
    all_est_csd = []
    
    metadata_files = glob.glob(os.path.join(args.data_dir, 'metadata_*.npz'))
    
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {args.data_dir}. Did you create the files?")

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
        est_csd_path = os.path.join(args.save_dir, f'estimate_CSD_{run_idx}.npy')

        if not (os.path.exists(true_csd_path) and os.path.exists(true_doa_path) and os.path.exists(est_doa_path)):
            print(f"Warning: Missing tracking arrays for Experiment {run_idx}. Skipping.")
            continue

        # Load arrays
        true_csd = np.load(true_csd_path)
        true_doa = np.load(true_doa_path)
        est_doa = np.load(est_doa_path)
        est_csd = np.load(est_csd_path) 

        csd_accuracy = np.mean(true_csd == est_csd) * 100
        if csd_accuracy < 60.0:
            print(f"  [WARNING] Experiment {run_idx} (T60={t60}s) has poor CSD accuracy: {csd_accuracy:.2f}%")

        # --- CSD Aggregation ---
        all_true_csd.extend(true_csd)
        all_est_csd.extend(est_csd)

        # 1. Filter: Keep only frames where a single speaker is active (CSD == 1)
        # and ignore the Sentinel overlap value (19) or silence (0)
        valid_mask = (true_csd == 1) & (true_doa != 0) & (true_doa != 19)
        valid_true_doa = true_doa[valid_mask]
        valid_est_doa = est_doa[valid_mask]

        if len(valid_true_doa) == 0:
            continue

        # 2. Compute absolute bin differences
        abs_diff = np.abs(valid_true_doa - valid_est_doa)

        errors_in_degrees = abs_diff * 10
        metrics[t60]['raw_errors'].extend(errors_in_degrees)

        # 3. Categorize errors (Resolution = 10 degrees, so 2 bins = 20 degrees)
        success = np.sum(abs_diff == 0)
        low_error = np.sum((abs_diff > 0) & (abs_diff <= 2))
        high_error = np.sum(abs_diff > 2)

        # 4. Aggregate
        metrics[t60]['success'] += success
        metrics[t60]['low'] += low_error
        metrics[t60]['high'] += high_error
        metrics[t60]['total'] += (success + low_error + high_error)

    return metrics, target_t60s, np.array(all_true_csd), np.array(all_est_csd)

def plot_stacked_bar(metrics, t60_list, save_dir):
    """Generates the grouped bar chart and/or the CDF plot based on the metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("--- Generating Grouped Bar Chart ---")
    success_pct, low_pct, high_pct = [], [], []
    success_counts, low_counts, high_counts = [], [], []
    labels = []

    for t60 in sorted(t60_list):
        data = metrics[t60]
        total = data['total']
        
        success_counts.append(data['success'])
        low_counts.append(data['low'])
        high_counts.append(data['high'])
        
        if total == 0:
            success_pct.append(0); low_pct.append(0); high_pct.append(0)
        else:
            success_pct.append((data['success'] / total) * 100)
            low_pct.append((data['low'] / total) * 100)
            high_pct.append((data['high'] / total) * 100)
        
        labels.append(f"{t60:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(labels))
    bar_width = 0.25 
    
    color_success, color_low, color_high = '#5a8cdb', '#9067c6', '#f094a4'

    bars_success = ax.bar(x_pos - bar_width, success_pct, bar_width, label='Exact (0° Error)', color=color_success, edgecolor='white')
    bars_low = ax.bar(x_pos, low_pct, bar_width, label='Low Error (≤ 20°)', color=color_low, edgecolor='white')
    bars_high = ax.bar(x_pos + bar_width, high_pct, bar_width, label='High Error (> 20°)', color=color_high, edgecolor='white')

    def add_labels(bars, counts):
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5, f'{count}', ha='center', va='bottom', fontsize=9, color='black')

    add_labels(bars_success, success_counts)
    add_labels(bars_low, low_counts)
    add_labels(bars_high, high_counts)

    ax.set_xlabel('T60 (sec)', fontsize=12, labelpad=10)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    bar_path = os.path.join(save_dir, 'DOA_Performance_Grouped_Comparison.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # סגירת הגרף כדי לא להעמיס על הזיכרון
    print(f"Bar Graph saved successfully to: {bar_path}")


def plot_global_csd_matrix(true_csd, est_csd, save_dir):
    """Renders the global CSD confusion matrix using the existing utility function."""
    print("--- Generating Global CSD Confusion Matrix ---")
    
    # Plotting Defaults to match pipeline.py
    annot, cmap, fmt, fz, lw, cbar = True, 'Oranges', '.2f', 9, 0.5, False
    show_null_values, pred_val_axis = 2, 'y'
    figsize = [18, 18]  # Adjust this if the global matrix feels too large
    
    cm_plot_labels_csd = ['Noise', 'One speaker', '2 speakers']
    
    os.makedirs(save_dir, exist_ok=True)
    
    plot_confusion_matrix_from_data(
        true_csd, est_csd, 3, cm_plot_labels_csd,
        annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis,
        name='CSD_Global_Confusion_Matrix.png', plot_folder=save_dir
    )

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run metric aggregation
    doa_metrics, sorted_t60s, all_true_csd, all_est_csd = process_metrics(args)
    
    # Generate the visualization
    plot_stacked_bar(doa_metrics, sorted_t60s, args.save_dir)
    
    # Generate CSD matrix if data was found
    if len(all_true_csd) > 0:
        plot_global_csd_matrix(all_true_csd, all_est_csd, args.save_dir)
    else:
        print("No CSD data gathered, skipping confusion matrix.")
        
    print("--- All plots generated successfully ---")