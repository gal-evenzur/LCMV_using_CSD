import os
import matplotlib.pyplot as plt
import pandas as pd

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, 'results.csv')
OUTPUT_LOW_CSV = os.path.join(SCRIPT_DIR, 'results_low_separation.csv')
OUTPUT_REST_CSV = os.path.join(SCRIPT_DIR, 'results_high_separation.csv')

# Separation thresholds
LOW_SEP_THRESH = 40  # <= 40 degrees
HIGH_SEP_THRESH = 50  # >= 50 degrees


def split_and_save_csv(input_path, low_path, rest_path):
    df = pd.read_csv(input_path)

    # Calculate average SDR, SIR, and SAR across speakers
    df['sdr_mean'] = df[['sdr_spk0', 'sdr_spk1']].mean(axis=1)
    df['sir_mean'] = df[['sir_spk0', 'sir_spk1']].mean(axis=1)
    df['sar_mean'] = df[['sar_spk0', 'sar_spk1']].mean(axis=1)

    # Split datasets
    df_low = df[df['mean_separation_deg'] <= LOW_SEP_THRESH].copy()
    df_rest = df[df['mean_separation_deg'] >= HIGH_SEP_THRESH].copy()

    # Save to disk
    df_low.to_csv(low_path, index=False)
    df_rest.to_csv(rest_path, index=False)
    print(f"Saved low separation (<= {LOW_SEP_THRESH}°) to: {low_path}")
    print(f"Saved rest separation (>= {HIGH_SEP_THRESH}°) to: {rest_path}")


def plot_results(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure speaker averages exist
    for metric in ['sdr', 'sir', 'sar']:
        if f'{metric}_mean' not in df.columns:
            df[f'{metric}_mean'] = df[[f'{metric}_spk0', f'{metric}_spk1']].mean(axis=1)

    metrics = ['sir_mean', 'sar_mean', 'sdr_mean']
    metric_labels = ['SIR', 'SAR', 'SDR']
    markers = ['o', 's', '^']

    # Create a figure with 1 row and 2 columns for subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Subplot 1: Performance vs. SNR (Fixed T60 = 0.3s) ---
    df_snr = (
        df[df['T60'] == 0.3]
        .groupby('SNR_diffuse')[metrics]
        .mean()
        .sort_index()
    )

    for metric, label, marker in zip(metrics, metric_labels, markers):
        axes[0].plot(df_snr.index, df_snr[metric], marker=marker, linewidth=2, label=f'Mean {label}')
    axes[0].set_title(r'Average Metrics vs. SNR (T_60 = 0.3 s), Separation >= 50°')
    axes[0].set_xlabel('SNR Diffuse (dB)')
    axes[0].set_ylabel('Score (dB)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # --- Subplot 2: Performance vs. T60 (Fixed SNR = 10 dB) ---
    df_t60 = (
        df[df['SNR_diffuse'] == 10.0]
        .groupby('T60')[metrics]
        .mean()
        .sort_index()
    )

    for metric, label, marker in zip(metrics, metric_labels, markers):
        axes[1].plot(df_t60.index, df_t60[metric], marker=marker, linewidth=2, label=f'Mean {label}')
    axes[1].set_title(r'Average Metrics vs. T_60 (SNR = 10 dB), Separation >= 50°')
    axes[1].set_xlabel(r'$T_{60}$ (s)')
    axes[1].set_ylabel('Score (dB)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'results.png'))
    plt.close()


if __name__ == '__main__':
    split_and_save_csv(FILE_PATH, OUTPUT_LOW_CSV, OUTPUT_REST_CSV)
    plot_results(OUTPUT_REST_CSV)