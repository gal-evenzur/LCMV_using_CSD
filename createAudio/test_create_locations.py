"""
Test script to understand and verify create_locations_18_dynamic function.

This function creates random speaker and microphone array positions within a room
for acoustic simulation purposes.
"""
import numpy as np
from dataset_funcs import *
import matplotlib.pyplot as plt

# Set random seed for reproducibility (optional - remove for true randomness)
np.random.seed(42)



# =============================================================================
# Run the generation first
print("Generating locations...")

simulator = AcousticTrajectorySimulator(room_dim=[6, 6, 3], speaker_radius=1.3, radius_noise=0.5, num_jumps=4, plot_name='test_locations')
s_first, label_first, s_second, label_second, s_noise, mic_positions = simulator.generate()

# =============================================================================