from pathlib import Path
from typing import Tuple
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.io import wavfile
import matplotlib.pyplot as plt
import das_generator as generator
import soundfile as sf
import os
from dataset_funcs import generate_source_path


# Parse command line arguments

current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, "Results")
os.makedirs(plots_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="DAS Generator Example")
parser.add_argument(
    "--audio", type=str, default="female_speech.wav", help="Path to audio file"
)
parser.add_argument(
    "--repeats", type=int, default=1, help="Number of times to repeat the audio"
)
args = parser.parse_args()

# Load source signal
audio_path = Path(args.audio)
source_signal, fs = sf.read(audio_path)

len_source_signal = len(source_signal)

# Room dimensions
room_dimensions = np.array([5.0, 4.0, 3.5])  # (m)

# Receiver type
receiver_type = generator.mic_type.omnidirectional

# Receiver positions
# receiver_positions = np.array([[2.6, 2.1, 1.5]])
receiver_positions = np.array([[2.6, 2.1, 1.5], [2.6, 2.3, 1.5]])

print(f"Audio loaded: {len_source_signal} samples at {fs}Hz")
print(f"Room dimensions: {room_dimensions}")

# Calculate the center of gravity (centroid) of the receiver positions
center = np.mean(receiver_positions, axis=0)


# Generate source path (moving) receiver paths (static)
hop = 32  # Number of samples between each position update (reduces computation time)



sp_path, labels = generate_source_path(
    len_source_signal=len_source_signal,
    hop=hop,
    center=center,
    radius=0.75,
    angle_classes=np.arange(5, 176, 10),
    fs=fs,
    linear_velocity=2,
    mode='bounce',
    start_angle=np.deg2rad(0),
    end_angle=np.deg2rad(90)
)

# Plot 3D source path and receiver positions
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for mm in range(receiver_positions.shape[0]):
    ax.plot(
        receiver_positions[mm, 0],
        receiver_positions[mm, 1],
        receiver_positions[mm, 2],
        "x",
        label=f"Receiver {mm + 1}",
    )
ax.plot(sp_path[0, :], sp_path[1, :], sp_path[2, :], "r.", label="Source Path")
ax.set_xlim([0, room_dimensions[0]])
ax.set_ylim([0, room_dimensions[1]])
ax.set_zlim([0, room_dimensions[2]])
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.grid(True)
ax.set_box_aspect(
    [room_dimensions[0], room_dimensions[1], room_dimensions[2]]
)  # Aspect ratio according to room dimensions
plt.legend()
plt.title("3D Source Path and Receiver Positions")
plt.savefig(os.path.join(plots_dir, "dynamic_source_path_and_receivers.png"))

# Plot 2D source path and receiver positions
plt.figure()
for mm in range(receiver_positions.shape[0]):
    plt.plot(
        receiver_positions[mm, 0],
        receiver_positions[mm, 1],
        "x",
        label=f"Receiver {mm + 1}",
    )
plt.plot(sp_path[0, :], sp_path[1, :], "r.", label="Source Path")
plt.xlim([0, room_dimensions[0]])
plt.ylim([0, room_dimensions[1]])
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.grid(True)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.title("2D Source Path and Receiver Positions (X-Y Plane)")
plt.savefig(os.path.join(plots_dir, "dynamic_source_path_and_receivers_2d.png"))


# Plot angle labels over time
label_time = np.arange(len(labels)) / fs
plt.figure()
plt.plot(label_time, labels, "b-")
plt.xlabel("Time [Seconds]")
plt.ylabel("Angle Label [Degrees]")
plt.title("Angle Labels Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "dynamic_angle_labels.png"))



receiver_signals = generator.generate(
    source_signal,  # Source signal
    c=340,  # Sound velocity (m/s)
    fs=fs,  # Sample frequency (samples/s)
    rp_path=receiver_positions,  # Static receiver positions
    sp_path=sp_path,  # Source positions for each sample
    L=room_dimensions,  # Room dimensions [x y z] (m)
    reverberation_time=0,  # Reverberation time (s)
    nRIR=1024,  # Number of output samples
    mtypes=receiver_type,  # Receiver type
    orientation=[0, 0],  # Orientation of the receiver
)

# Check dimensions of input and output signals
print("Shape source_signal: ", source_signal.shape)
print("Shape receiver_signals: ", receiver_signals.shape)
print("Labels shape: ", labels.shape)

# Plot input and output signals
t = np.linspace(0, (len(source_signal) - 1) / fs, len(source_signal))

plt.figure()
plt.subplot(211)
plt.plot(t, source_signal)
plt.title("in(n)")
plt.xlabel("Time [Seconds]")
plt.ylabel("Amplitude")

plt.subplot(212)
plt.plot(t, receiver_signals)
plt.title("out(n)")
plt.xlabel("Time [Seconds]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "dynamic_signals.png"))

# Save output signals to a WAV file
wavfile.write(os.path.join(plots_dir, "dynamic_receiver_signals.wav"), fs, receiver_signals)
