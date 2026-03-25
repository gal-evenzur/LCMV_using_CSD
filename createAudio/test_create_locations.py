"""
Test script to understand and verify create_locations_18_dynamic function.

This function creates random speaker and microphone array positions within a room
for acoustic simulation purposes.
"""
import numpy as np
from dataset_funcs import create_locations_18_dynamic
import matplotlib.pyplot as plt

# Set random seed for reproducibility (optional - remove for true randomness)
np.random.seed(42)



# =============================================================================
# Run the generation first
print("Generating locations...")
s1, l1, s2, l2, noise, mics = create_locations_18_dynamic(
    room_dim=[6, 6, 3], speaker_radius=1.3, radius_noise=0.5, num_jumps=4, plot_name='test_locations'
)
