import pandas as pd
import os
# --- Configuration (Easily Modifiable) ---
FILE_PATH = os.path.join(os.path.dirname(__file__), 'results.csv')
GROUP_BY_COLS = ['T60', 'SNR_diffuse']
SEPARATION_COL = 'mean_separation_deg'

# Define separation degree thresholds
HIGH_SEP_THRESH = 50
LOW_SEP_THRESH = 40

def analyze_results(filepath):
    # 1. Load the dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find file at '{filepath}'")
        return

    # 2. Average the metrics across both speakers (spk0 and spk1)
    # Modify these pairs if your column names change or you add more metrics
    df['sdr_mean'] = df[['sdr_spk0', 'sdr_spk1']].mean(axis=1)
    df['sir_mean'] = df[['sir_spk0', 'sir_spk1']].mean(axis=1)
    df['sar_mean'] = df[['sar_spk0', 'sar_spk1']].mean(axis=1)
    
    # The new columns we want to aggregate
    metrics_to_agg = ['sdr_mean', 'sir_mean', 'sar_mean']
    
    # 3. Filter data by separation degree thresholds
    df_high = df[df[SEPARATION_COL] >= HIGH_SEP_THRESH]
    df_low = df[df[SEPARATION_COL] <= LOW_SEP_THRESH]
    
    # 4. Helper function to group and aggregate
    def get_stats(data):
        # Change ['mean', 'median'] to include 'std', 'min', 'max', etc. if needed
        return data.groupby(GROUP_BY_COLS)[metrics_to_agg].agg(['mean', 'median']).round(2)
    
    # 5. Output the results
    print(f"--- Separation >= {HIGH_SEP_THRESH} Degrees ---")
    print(get_stats(df_high))
    
    print(f"\n--- Separation <= {LOW_SEP_THRESH} Degrees ---")
    print(get_stats(df_low))

    # Let's save to CSV the metrics for both high and low separation cases
    df_metrics_high = get_stats(df_high).reset_index()
    df_metrics_low = get_stats(df_low).reset_index()
    output_dir = os.path.dirname(filepath)
    df_metrics_high.to_csv(os.path.join(output_dir, f'METRICS_high_separation.csv'), index=False)
    df_metrics_low.to_csv(os.path.join(output_dir, f'METRICS_low_separation.csv'), index=False)

if __name__ == "__main__":
    analyze_results(FILE_PATH)