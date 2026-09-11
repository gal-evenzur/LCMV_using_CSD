import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
py_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(py_dir, 'total_beamformer_results_time_graph.csv')
# Read the data
df = pd.read_csv(csv_path)

# Create wav_length groups (round to nearest 10)
df['wav_length_group'] = (df['wav_length_sec'] / 10).round() * 10

# Calculate the median of time_sec for each method and length group
median_times = df.groupby(['method', 'wav_length_group'])['time_sec'].median().reset_index()

# 1. Graph of median time_sec by wav_length for each section
plt.figure(figsize=(10, 5))
sns.lineplot(data=median_times, x='wav_length_group', y='time_sec', hue='method', marker='o')
plt.title('Median Processing Time vs. Wav Length Group by Method')
plt.xlabel('Wav Length Group (seconds)')
plt.ylabel('Median Time (seconds)')
plt.grid(True)
plt.savefig(os.path.join(py_dir, 'median_time_vs_length.png'))
plt.close()

# 2. Percentage improvement graph based on medians
pivot_df = median_times.pivot(index='wav_length_group', columns='method', values='time_sec')
pivot_df['Improvement_%'] = ((pivot_df['GEVD'] - pivot_df['PASTD']) / pivot_df['GEVD']) * 100

plt.figure(figsize=(10, 5))
sns.barplot(x=pivot_df.index, y=pivot_df['Improvement_%'], color='lightgreen')
plt.title('PASTD Median Processing Time Improvement over GEVD (%)')
plt.xlabel('Wav Length Group (seconds)')
plt.ylabel('Improvement in Median Time (%)')
plt.grid(axis='y')
plt.savefig(os.path.join(py_dir, 'median_improvement.png'))
plt.close()

# Print the table data for reference
print(pivot_df.reset_index().to_markdown(index=False))