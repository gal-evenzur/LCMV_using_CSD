"""
Analyze generated database to find time ranges with specific speaker activity.

This script loads the label files and identifies time ranges where:
1. Only noise (both speakers silent)
2. Only first speaker is active
3. Only second speaker is active
"""

import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt


def find_activity_ranges(vad_first: np.ndarray, vad_second: np.ndarray, 
                         hop: int, fs: int) -> dict:
    """
    Find time ranges for different speaker activity patterns.
    
    Parameters
    ----------
    vad_first : np.ndarray
        Frame-level labels for speaker 1 (0 = silent, >0 = active)
    vad_second : np.ndarray
        Frame-level labels for speaker 2 (0 = silent, >0 = active)
    hop : int
        Hop size in samples
    fs : int
        Sample frequency
    
    Returns
    -------
    ranges : dict
        Dictionary with 'noise_only', 'first_only', 'second_only', 'both' keys
        Each contains a list of (start_time, end_time) tuples in seconds
    """
    # Convert to binary activity
    first_active = vad_first > 0
    second_active = vad_second > 0
    
    # Find different activity patterns
    noise_only = ~first_active & ~second_active
    first_only = first_active & ~second_active
    second_only = ~first_active & second_active
    both_active = first_active & second_active
    
    def get_ranges(mask):
        """Find continuous ranges where mask is True."""
        ranges = []
        in_range = False
        start_frame = 0
        
        for i, val in enumerate(mask):
            if val and not in_range:
                start_frame = i
                in_range = True
            elif not val and in_range:
                end_frame = i
                start_time = start_frame * hop / fs
                end_time = end_frame * hop / fs
                ranges.append((start_time, end_time))
                in_range = False
        
        # Handle case where range extends to end
        if in_range:
            end_time = len(mask) * hop / fs
            start_time = start_frame * hop / fs
            ranges.append((start_time, end_time))
        
        return ranges
    
    return {
        'noise_only': get_ranges(noise_only),
        'first_only': get_ranges(first_only),
        'second_only': get_ranges(second_only),
        'both': get_ranges(both_active)
    }


def analyze_sample(sample_idx: int = 1, data_path: str = None):
    """
    Analyze a specific sample from the database.
    
    Parameters
    ----------
    sample_idx : int
        Sample index to analyze
    data_path : str
        Path to data folder
    """
    # Default path
    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data')
    
    # Parameters (must match create_data_base.py)
    fs = 16000
    hop = 512
    nfft = 2048
    
    print("=" * 70)
    print(f"Analyzing sample {sample_idx}")
    print("=" * 70)
    
    # Load files
    audio_file = os.path.join(data_path, f'together_{sample_idx}.wav')
    label_first_file = os.path.join(data_path, f'label_location_first_{sample_idx}.npy')
    label_second_file = os.path.join(data_path, f'label_location_second_{sample_idx}.npy')
    
    audio, audio_fs = sf.read(audio_file)
    vad_first = np.load(label_first_file)
    vad_second = np.load(label_second_file)
    
    print(f"\nAudio shape: {audio.shape} ({audio.shape[0]/fs:.2f} seconds)")
    print(f"Label frames: {len(vad_first)} frames")
    print(f"Frame duration: {hop/fs*1000:.1f} ms")
    
    # Find activity ranges
    ranges = find_activity_ranges(vad_first, vad_second, hop, fs)
    
    # Print results
    print("\n" + "-" * 70)
    print("TIME RANGES BY ACTIVITY TYPE")
    print("-" * 70)
    
    # Noise only
    print("\n🔇 NOISE ONLY (both speakers silent):")
    if ranges['noise_only']:
        # Find longest range
        longest = max(ranges['noise_only'], key=lambda x: x[1] - x[0])
        print(f"   First occurrence: {ranges['noise_only'][0][0]:.3f}s - {ranges['noise_only'][0][1]:.3f}s")
        print(f"   Longest range:    {longest[0]:.3f}s - {longest[1]:.3f}s ({longest[1]-longest[0]:.3f}s)")
        print(f"   Total ranges:     {len(ranges['noise_only'])}")
    else:
        print("   No ranges found")
    
    # First speaker only
    print("\n🔊 FIRST SPEAKER ONLY:")
    if ranges['first_only']:
        longest = max(ranges['first_only'], key=lambda x: x[1] - x[0])
        print(f"   First occurrence: {ranges['first_only'][0][0]:.3f}s - {ranges['first_only'][0][1]:.3f}s")
        print(f"   Longest range:    {longest[0]:.3f}s - {longest[1]:.3f}s ({longest[1]-longest[0]:.3f}s)")
        print(f"   Total ranges:     {len(ranges['first_only'])}")
        # Get angle class for first occurrence
        start_frame = int(ranges['first_only'][0][0] * fs / hop)
        angle_class = int(vad_first[start_frame])
        print(f"   Angle class:      {angle_class}")
    else:
        print("   No ranges found")
    
    # Second speaker only
    print("\n🔊 SECOND SPEAKER ONLY:")
    if ranges['second_only']:
        longest = max(ranges['second_only'], key=lambda x: x[1] - x[0])
        print(f"   First occurrence: {ranges['second_only'][0][0]:.3f}s - {ranges['second_only'][0][1]:.3f}s")
        print(f"   Longest range:    {longest[0]:.3f}s - {longest[1]:.3f}s ({longest[1]-longest[0]:.3f}s)")
        print(f"   Total ranges:     {len(ranges['second_only'])}")
        # Get angle class for first occurrence
        start_frame = int(ranges['second_only'][0][0] * fs / hop)
        angle_class = int(vad_second[start_frame])
        print(f"   Angle class:      {angle_class}")
    else:
        print("   No ranges found")
    
    # Both speakers
    print("\n🔊🔊 BOTH SPEAKERS ACTIVE:")
    if ranges['both']:
        longest = max(ranges['both'], key=lambda x: x[1] - x[0])
        print(f"   First occurrence: {ranges['both'][0][0]:.3f}s - {ranges['both'][0][1]:.3f}s")
        print(f"   Longest range:    {longest[0]:.3f}s - {longest[1]:.3f}s ({longest[1]-longest[0]:.3f}s)")
        print(f"   Total ranges:     {len(ranges['both'])}")
    else:
        print("   No ranges found")
    
    # Summary for easy copy-paste
    print("\n" + "=" * 70)
    print("SUMMARY - Example time ranges:")
    print("=" * 70)
    
    if ranges['noise_only']:
        r = ranges['noise_only'][0]
        print(f"  Noise only:      [{r[0]:.3f}, {r[1]:.3f}] seconds")
    
    if ranges['first_only']:
        r = ranges['first_only'][0]
        print(f"  First speaker:   [{r[0]:.3f}, {r[1]:.3f}] seconds")
    
    if ranges['second_only']:
        r = ranges['second_only'][0]
        print(f"  Second speaker:  [{r[0]:.3f}, {r[1]:.3f}] seconds")
    
    # Create visualization
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    # Time axis for audio
    t_audio = np.arange(len(audio)) / fs
    
    # Time axis for frames (center of each frame)
    t_frames = np.arange(len(vad_first)) * hop / fs + nfft / (2 * fs)
    
    # Plot 1: Audio waveform (first channel)
    ax1 = axes[0]
    ax1.plot(t_audio, audio[:, 0], 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Audio Waveform (Channel 1) - together_{sample_idx}.wav')
    ax1.grid(True, alpha=0.3)
    
    # Highlight regions
    for start, end in ranges['noise_only'][:5]:  # Limit to first 5
        ax1.axvspan(start, end, alpha=0.2, color='gray', label='Noise only' if start == ranges['noise_only'][0][0] else None)
    for start, end in ranges['first_only'][:5]:
        ax1.axvspan(start, end, alpha=0.2, color='blue', label='First only' if start == ranges['first_only'][0][0] else None)
    for start, end in ranges['second_only'][:5]:
        ax1.axvspan(start, end, alpha=0.2, color='green', label='Second only' if start == ranges['second_only'][0][0] else None)
    
    ax1.legend(loc='upper right')
    
    # Plot 2: Speaker 1 labels
    ax2 = axes[1]
    ax2.fill_between(t_frames, vad_first, alpha=0.7, color='blue', label='Speaker 1')
    ax2.set_ylabel('Angle Class')
    ax2.set_title('Speaker 1 - Direction Labels')
    ax2.set_ylim([-0.5, 18])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot 3: Speaker 2 labels
    ax3 = axes[2]
    ax3.fill_between(t_frames, vad_second, alpha=0.7, color='green', label='Speaker 2')
    ax3.set_ylabel('Angle Class')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Speaker 2 - Direction Labels')
    ax3.set_ylim([-0.5, 18])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    result_path = os.path.join(os.path.dirname(data_path), 'Results')
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, f'activity_analysis_{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return ranges


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze speaker activity in generated samples')
    parser.add_argument('--sample', type=int, default=1, help='Sample index to analyze')
    parser.add_argument('--data-path', type=str, default=None, help='Path to data folder')
    
    args = parser.parse_args()
    
    ranges = analyze_sample(sample_idx=args.sample, data_path=args.data_path)
