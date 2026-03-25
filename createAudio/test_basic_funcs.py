"""
Test script to verify fillline and plotcube functions work correctly.
"""
import numpy as np
import matplotlib.pyplot as plt
from dataset_funcs import fillline, plotcube, create_vad_dynamic
import os

tests = {
    'filline': 0,
    'plotcube': 0,
    'create_locations_18_dynamic': 1,
}

# =============================================================================
# Test fillline function
# =============================================================================
print("=" * 50)
print("Testing fillline function")
print("=" * 50)

# Test 1: Regular diagonal line
print("\n1. Diagonal line from (0,0) to (10,5):")
xx, yy = fillline([0, 0], [10, 5], 11)
print(f"   xx = {xx}")
print(f"   yy = {yy}")

# Test 2: Horizontal line (gradient = 0)
print("\n2. Horizontal line from (0,3) to (10,3):")
xx, yy = fillline([0, 3], [10, 3], 6)
print(f"   xx = {xx}")
print(f"   yy = {yy}")

# Test 3: Vertical line (gradient = inf)
print("\n3. Vertical line from (5,0) to (5,10):")
xx, yy = fillline([5, 0], [5, 10], 6)
print(f"   xx = {xx}")
print(f"   yy = {yy}")

# Test 4: Negative slope line
print("\n4. Negative slope line from (0,10) to (10,0):")
xx, yy = fillline([0, 10], [10, 0], 6)
print(f"   xx = {xx}")
print(f"   yy = {yy}")


# =============================================================================
# Visual test: Plot all line types
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 2D lines
ax1 = axes[0]
ax1.set_title("fillline() - Different Line Types")

# Diagonal
xx, yy = fillline([0, 0], [10, 10], 20)
ax1.plot(xx, yy, 'bo-', label='Diagonal', markersize=4)

# Horizontal
xx, yy = fillline([0, 5], [10, 5], 20)
ax1.plot(xx, yy, 'rs-', label='Horizontal', markersize=4)

# Vertical
xx, yy = fillline([5, 0], [5, 10], 20)
ax1.plot(xx, yy, 'g^-', label='Vertical', markersize=4)

# Negative slope
xx, yy = fillline([0, 8], [8, 0], 20)
ax1.plot(xx, yy, 'm*-', label='Negative slope', markersize=4)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal')


# =============================================================================
# Test plotcube function
# =============================================================================
print("\n" + "=" * 50)
print("Testing plotcube function")
print("=" * 50)
print("\nCreating 3D plot with multiple cubes...")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# Remove the flat axes[1] and replace with 3D
axes[1].remove()

# Cube 1: Red cube at origin
plotcube([3, 3, 3], [0, 0, 0], 0.6, [1, 0, 0], ax2)
print("   - Red cube: edges=[3,3,3], origin=[0,0,0]")

# Cube 2: Green cube offset
plotcube([2, 4, 2], [5, 0, 0], 0.6, [0, 1, 0], ax2)
print("   - Green cube: edges=[2,4,2], origin=[5,0,0]")

# Cube 3: Blue cube at different position
plotcube([3, 2, 4], [0, 5, 0], 0.6, [0, 0, 1], ax2)
print("   - Blue cube: edges=[3,2,4], origin=[0,5,0]")

# Cube 4: Yellow cube (semi-transparent room-like)
plotcube([8, 8, 5], [0, 0, 0], 0.1, [1, 1, 0], ax2)
print("   - Yellow cube (room): edges=[8,8,5], origin=[0,0,0], alpha=0.1")

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('plotcube() - Multiple Cubes')

# Set axis limits
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 10])
ax2.set_zlim([0, 6])

plt.tight_layout()
py_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(py_path, 'Results')
os.makedirs(result_path, exist_ok=True)
plt.savefig(os.path.join(result_path, 'test_output.png'), dpi=150)
print("\n✓ Plot saved to 'test_output.png'")



def test_create_vad_dynamic():
    """
    Test script for create_vad_dynamic function.

    This script tests the frame-level label extraction and visualizes:
    1. Sample-level labels (input)
    2. Frame-level labels after majority voting (output)
    3. Comparison showing how labels are downsampled to STFT frames
"""
    print("=" * 60)
    print("Testing create_vad_dynamic")
    print("=" * 60)
    
    # --- Parameters (same as in create_data_base.m) ---
    fs = 16000
    hop = 512
    nfft = 2048
    
    # --- Create synthetic sample-level labels ---
    # Simulate a signal with:
    # - Initial silence (zeros)
    # - Speaker 1 at angle class 5
    # - Silence padding
    # - Speaker 1 at angle class 12
    # - More silence
    
    duration_sec = 3
    total_samples = duration_sec * fs
    
    # Create label signal
    label_signal = np.zeros(total_samples)
    
    # Segment 1: Silence (0 - 0.3s)
    # Already zeros
    
    # Segment 2: Speaker at angle class 5 (0.3s - 0.8s)
    start1 = int(0.3 * fs)
    end1 = int(0.8 * fs)
    label_signal[start1:end1] = 5
    
    # Segment 3: Silence (0.8s - 1.2s)
    # Already zeros
    
    # Segment 4: Speaker at angle class 12 (1.2s - 2.0s)
    start2 = int(1.2 * fs)
    end2 = int(2.0 * fs)
    label_signal[start2:end2] = 12
    
    # Segment 5: Speaker at angle class 3 (2.0s - 2.5s)
    start3 = int(2.0 * fs)
    end3 = int(2.5 * fs)
    label_signal[start3:end3] = 3
    
    # Segment 6: Silence (2.5s - 3.0s)
    # Already zeros
    
    print(f"\nParameters:")
    print(f"  Sample frequency: {fs} Hz")
    print(f"  Hop size: {hop} samples ({hop/fs*1000:.1f} ms)")
    print(f"  Window size (nfft): {nfft} samples ({nfft/fs*1000:.1f} ms)")
    print(f"  Total duration: {duration_sec} seconds ({total_samples} samples)")
    
    # --- Run create_vad_dynamic ---
    print("\nRunning create_vad_dynamic...")
    vad = create_vad_dynamic(label_signal, hop, nfft)
    
    # Calculate expected number of frames
    expected_frames = 1 + (total_samples - nfft) // hop
    
    print(f"\nResults:")
    print(f"  Input samples: {len(label_signal)}")
    print(f"  Output frames: {len(vad)}")
    print(f"  Expected frames: {expected_frames}")
    print(f"  Unique labels in output: {np.unique(vad)}")
    
    # --- Verify output ---
    assert len(vad) == expected_frames, f"Frame count mismatch: {len(vad)} vs {expected_frames}"
    print("\n✓ Frame count matches expected value")
    
    # Check that we get the expected labels
    unique_labels = set(np.unique(vad))
    expected_labels = {0, 3, 5, 12}
    assert unique_labels == expected_labels, f"Label mismatch: {unique_labels} vs {expected_labels}"
    print("✓ All expected labels found in output")
    
    # --- Create visualization ---
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Time axes
    t_samples = np.arange(total_samples) / fs
    t_frames = np.arange(len(vad)) * hop / fs + nfft / (2 * fs)  # Center of each frame
    
    # --- Plot 1: Sample-level labels ---
    ax1 = axes[0]
    ax1.plot(t_samples, label_signal, 'b-', linewidth=0.5)
    ax1.fill_between(t_samples, label_signal, alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Label (angle class)')
    ax1.set_title('Sample-level Labels (Input)')
    ax1.set_ylim([-0.5, 15])
    ax1.grid(True, alpha=0.3)
    
    # Add segment annotations
    ax1.axvspan(0.3, 0.8, alpha=0.2, color='green', label='Class 5')
    ax1.axvspan(1.2, 2.0, alpha=0.2, color='red', label='Class 12')
    ax1.axvspan(2.0, 2.5, alpha=0.2, color='orange', label='Class 3')
    ax1.legend(loc='upper right')
    
    # --- Plot 2: Frame-level labels ---
    ax2 = axes[1]
    ax2.stem(t_frames, vad, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Label (angle class)')
    ax2.set_title(f'Frame-level Labels (Output) - {len(vad)} frames')
    ax2.set_ylim([-0.5, 15])
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Overlay comparison ---
    ax3 = axes[2]
    
    # Plot sample-level as background
    ax3.fill_between(t_samples, label_signal, alpha=0.3, color='blue', label='Sample-level')
    
    # Plot frame-level as step function
    # Create step representation for frames
    frame_starts = np.arange(len(vad)) * hop / fs
    frame_ends = frame_starts + nfft / fs
    
    for i, label in enumerate(vad):
        if label > 0:
            ax3.hlines(y=label, xmin=frame_starts[i], xmax=min(frame_ends[i], duration_sec), 
                      colors='red', linewidth=2)
            ax3.plot(frame_starts[i], label, 'r|', markersize=10)
    
    ax3.scatter(t_frames, vad, c='red', s=30, zorder=5, label='Frame-level (mode)')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Label (angle class)')
    ax3.set_title('Comparison: Sample-level vs Frame-level Labels')
    ax3.set_ylim([-0.5, 15])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    py_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(py_path, 'Results')
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, 'test_vad_dynamic.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # --- Test edge cases ---
    print("\n" + "-" * 40)
    print("Testing edge cases...")
    
    # Test 1: All zeros
    all_zeros = np.zeros(10000)
    vad_zeros = create_vad_dynamic(all_zeros, hop, nfft)
    assert np.all(vad_zeros == 0), "All-zero input should give all-zero output"
    print("✓ All-zeros input: passed")
    
    # Test 2: Single label throughout
    all_sevens = np.ones(10000) * 7
    vad_sevens = create_vad_dynamic(all_sevens, hop, nfft)
    assert np.all(vad_sevens == 7), "Constant-7 input should give all-7 output"
    print("✓ Constant label input: passed")
    
    # Test 3: Mixed window (majority should win)
    # Create a window where 70% is label 5 and 30% is label 8
    mixed = np.zeros(nfft)
    mixed[:int(0.7 * nfft)] = 5
    mixed[int(0.7 * nfft):] = 8
    # Pad to have at least one full frame
    mixed_padded = np.concatenate([mixed, np.zeros(hop)])
    vad_mixed = create_vad_dynamic(mixed_padded, hop, nfft)
    assert vad_mixed[0] == 5, f"Majority vote should give 5, got {vad_mixed[0]}"
    print("✓ Majority voting: passed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    return vad

test_create_vad_dynamic()

print("\n" + "=" * 50)
print("All tests completed successfully!")
print("=" * 50)

