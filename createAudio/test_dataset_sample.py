import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def labels_to_angles(labels):
    labels = np.asarray(labels)
    angles = np.full(labels.shape, np.nan, dtype=float)
    active = labels > 0
    angles[active] = labels[active] * 10 + 5
    return angles


def visualize_audio_and_angles(sample_idx, data_dir='data', channel_idx=0, output_dir='Results'):
    """
    Plot the mixed audio waveform and the time-aligned angle of each speaker.

    Parameters
    ----------
    sample_idx : int
        The index of the sample to visualize.
    data_dir : str
        Path to the folder containing the generated .wav and .npy files.
    channel_idx : int
        Microphone channel to plot from the multi-channel mixture.
    """
    mixed_wav = os.path.join(data_dir, f'together_{sample_idx}.wav')
    label1_npy = os.path.join(data_dir, f'label_location_first_{sample_idx}.npy')
    label2_npy = os.path.join(data_dir, f'label_location_second_{sample_idx}.npy')

    for file_path in [mixed_wav, label1_npy, label2_npy]:
        if not os.path.exists(file_path):
            print(f'Error: File not found: {file_path}')
            return

    y_mix, fs = sf.read(mixed_wav)
    l1 = np.load(label1_npy)
    l2 = np.load(label2_npy)

    hop = 512

    if y_mix.ndim == 1:
        waveform = y_mix
        channel_label = 'mono'
    else:
        if channel_idx < 0 or channel_idx >= y_mix.shape[1]:
            raise ValueError(
                f'channel_idx must be between 0 and {y_mix.shape[1] - 1}, got {channel_idx}'
            )
        waveform = y_mix[:, channel_idx]
        channel_label = f'channel {channel_idx}'

    time_audio = np.arange(len(waveform)) / fs
    time_frames = np.arange(len(l1)) * hop / fs

    ang1 = labels_to_angles(l1)
    ang2 = labels_to_angles(l2)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )
    ax_waveform, ax_angles = axes

    ax_waveform.plot(time_audio, waveform, color='0.2', linewidth=0.6)
    ax_waveform.set_title(f'Mixed audio waveform ({channel_label}) for sample {sample_idx}')
    ax_waveform.set_ylabel('Amplitude')
    ax_waveform.grid(True, alpha=0.3)

    ax_angles.step(time_frames, ang1, where='post', linewidth=1.5, color='tab:blue', label='Speaker 1')
    ax_angles.step(time_frames, ang2, where='post', linewidth=1.5, color='tab:orange', label='Speaker 2')
    ax_angles.set_title('Speaker angle over time')
    ax_angles.set_xlabel('Time (s)')
    ax_angles.set_ylabel('Angle (deg)')
    ax_angles.set_ylim(0, 180)
    ax_angles.set_yticks(np.arange(0, 181, 15))
    ax_angles.grid(True, alpha=0.3)
    ax_angles.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_{sample_idx}_visualization.png')


visualize_audio_and_angles(
    2,
    data_dir='/home/evenzug/LCMV_using_CSD/Python_Create_Data_Base/data',
    output_dir='/home/evenzug/LCMV_using_CSD/Python_Create_Data_Base/Results',
)
