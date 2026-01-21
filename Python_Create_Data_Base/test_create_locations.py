"""
Test script to understand and verify create_locations_18_dynamic function.

This function creates random speaker and microphone array positions within a room
for acoustic simulation purposes.
"""
import numpy as np
from dataset_funcs import create_locations_18_dynamic

# Set random seed for reproducibility (optional - remove for true randomness)
np.random.seed(11)

# =============================================================================
# Define room and simulation parameters
# =============================================================================
print("=" * 70)
print("Testing create_locations_18_dynamic function")
print("=" * 70)

# Room dimensions [x, y, z] in meters
L = [6, 5, 3]  # 6m x 5m x 3m room

# Radius of the speaker circle around the microphone array (meters)
R = 1.5

# Noise radius for position perturbation (meters)
# This adds small random offsets to speaker positions
noise_R = 0.2

# Number of speaker position configurations (jumps) to generate
num_jumps = 3

print(f"\nInput Parameters:")
print(f"  Room dimensions (L):     {L} meters")
print(f"  Speaker circle radius (R): {R} meters")
print(f"  Position noise radius:   {noise_R} meters")
print(f"  Number of jumps:         {num_jumps}")

# =============================================================================
# Run the function
# =============================================================================
print("\n" + "-" * 70)
print("Running create_locations_18_dynamic...")
print("-" * 70)

s_first, label_first, s_second, label_second, s_noise, r = create_locations_18_dynamic(
    L, R, noise_R, num_jumps
)

# =============================================================================
# Analyze and display the outputs
# =============================================================================
print("\n" + "=" * 70)
print("OUTPUT ANALYSIS")
print("=" * 70)

# --- Microphone Array Positions ---
print("\n1. MICROPHONE ARRAY POSITIONS (r)")
print("   Shape:", r.shape, "-> 4 microphones, each with [x, y, z] coordinates")
print("   The microphones are arranged in a small circle (radius=0.1m)")
for i, mic in enumerate(r):
    print(f"   Mic {i+1}: x={mic[0]:.3f}, y={mic[1]:.3f}, z={mic[2]:.3f}")

# --- First Speaker Positions ---
print("\n2. FIRST SPEAKER POSITIONS (s_first)")
print("   Shape:", s_first.shape, f"-> 3 rows [x, y, z] x {num_jumps} positions")
print("   Each column represents a different 'jump' (time segment)")
for j in range(num_jumps):
    print(f"   Jump {j+1}: x={s_first[0,j]:.3f}, y={s_first[1,j]:.3f}, z={s_first[2,j]:.3f}")

# --- Second Speaker Positions ---
print("\n3. SECOND SPEAKER POSITIONS (s_second)")
print("   Shape:", s_second.shape, f"-> 3 rows [x, y, z] x {num_jumps} positions")
for j in range(num_jumps):
    print(f"   Jump {j+1}: x={s_second[0,j]:.3f}, y={s_second[1,j]:.3f}, z={s_second[2,j]:.3f}")

# --- Angular Labels ---
print("\n4. ANGULAR LABELS")
print("   Labels represent discretized angles (0-17) mapping to 5°-175° in 10° steps")
print("   labels_location = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175]")
labels_location = np.arange(5, 176, 10)
print(f"\n   First speaker labels:  {label_first}")
print(f"   -> Approximate angles: {[labels_location[l] for l in label_first]} degrees")
print(f"\n   Second speaker labels: {label_second}")
print(f"   -> Approximate angles: {[labels_location[l] for l in label_second]} degrees")

# --- Noise Source Position ---
print("\n5. NOISE SOURCE POSITION (s_noise)")
print(f"   Position: x={s_noise[0]:.3f}, y={s_noise[1]:.3f}, z={s_noise[2]:.3f}")
print("   (Located at least 2m away from microphone array center)")

# =============================================================================
# Verify constraints
# =============================================================================
print("\n" + "=" * 70)
print("CONSTRAINT VERIFICATION")
print("=" * 70)

# Check speaker separation
print("\n1. Speaker Separation (should be >= 0.5m between speakers):")
from scipy.spatial.distance import pdist
for j in range(num_jumps):
    pos1 = [s_first[0, j], s_first[1, j]]
    pos2 = [s_second[0, j], s_second[1, j]]
    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
    status = "✓" if dist >= 0.5 else "✗"
    print(f"   Jump {j+1}: distance = {dist:.3f}m {status}")

# Check that speakers are within room bounds
print("\n2. Room Boundary Check (speakers should be inside room):")
for j in range(num_jumps):
    x1_in = 0 < s_first[0, j] < L[0]
    y1_in = 0 < s_first[1, j] < L[1]
    x2_in = 0 < s_second[0, j] < L[0]
    y2_in = 0 < s_second[1, j] < L[1]
    status = "✓" if all([x1_in, y1_in, x2_in, y2_in]) else "✗"
    print(f"   Jump {j+1}: Speaker 1 in bounds: {x1_in and y1_in}, Speaker 2 in bounds: {x2_in and y2_in} {status}")

# Check noise source distance from center
print("\n3. Noise Source Distance from Mic Array Center:")
mic_center = np.mean(r[:, :2], axis=0)
noise_dist = np.linalg.norm(s_noise[:2] - mic_center)
status = "✓" if noise_dist >= 2 else "✗"
print(f"   Distance: {noise_dist:.3f}m (should be >= 2m) {status}")

print("\n" + "=" * 70)
print("Test completed! Check the 3D plot for visualization.")
print("=" * 70)
