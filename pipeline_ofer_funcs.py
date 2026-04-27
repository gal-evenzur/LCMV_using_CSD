from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.io import wavfile
from DataSamples_to_InputVectors.plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
from DataSamples_to_InputVectors.stft import stft
from DataSamples_to_InputVectors.istft import istft
from joblib import delayed
import multiprocessing
from numpy import linalg as LA
from scipy.io.wavfile import write
import scipy.io as sio
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
import logging
from pystoi import stoi
from pesq import pesq
import mir_eval
from scipy import ndimage
import os


def load_and_preprocess_experiment(data_dir, experiment_index, mic_indices):
    """
    Loads and pre-processes audio and location data for a specific experiment.
    
    Args:
        data_dir (str): Path to the directory containing the raw data.
        experiment_index (int): The ID number of the experiment (e.g., 3).
        mic_indices (list): The specific microphone channels to extract (e.g., [0, 1, 2, 3]).
        
    Returns:
        dict: A dictionary containing the normalized audio arrays, sample rate, and raw angle data.
    """
    
    # 1. Construct File Paths
    files = {
        'mixed': os.path.join(data_dir, f'together_{experiment_index}.wav'),
        'spk1': os.path.join(data_dir, f'first_{experiment_index}.wav'),
        'spk2': os.path.join(data_dir, f'second_{experiment_index}.wav'),
        'loc1': os.path.join(data_dir, f'label_location_first_{experiment_index}.npy'),
        'loc2': os.path.join(data_dir, f'label_location_second_{experiment_index}.npy')
    }
    
    # Verify all files exist before processing
    for name, filepath in files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing expected file: {filepath}")

    # 2. Helper function to load, slice, and normalize audio
    def process_audio(filepath):
        fs, audio_data = wavfile.read(filepath)
        
        # Ensure we are only grabbing the specific microphones we want
        if audio_data.ndim > 1:
            audio_data = audio_data[:, mic_indices]
            
        # Normalization: Scale amplitudes to strictly fall between -1.0 and 1.0
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val
            
        return fs, audio_data

    # 3. Process all audio files
    fs, mix_audio = process_audio(files['mixed'])
    _, spk1_audio = process_audio(files['spk1'])
    _, spk2_audio = process_audio(files['spk2'])

    # 4. Load angle data for both speakers, frame by frame
    frame_angles_first = np.load(files['loc1'])
    frame_angles_second = np.load(files['loc2'])

    y2_first = frame_angles_first.reshape(-1, 1)
    y2_second = frame_angles_second.reshape(-1, 1)

    # 5. Return a structured dictionary for the main loop
    return {
        'fs': fs,
        'audio': {
            'mixed': mix_audio,
            'spk1': spk1_audio,
            'spk2': spk2_audio,
        },
        'angles': {
            'spk1': y2_first,
            'spk2': y2_second
        }
    }



def calc_n_frames(xlen, hop, wlen):
    '''
    Calculates the number of frames given the length of the signal, hop size, and window length.
    inputs:
        xlen (int): Length of the input signal in samples.
        hop (int): Hop size in samples.
        wlen (int): Window length in samples.
    outputs:
            n_frames (int): The total number of frames that will be generated from the input signal.
    formula_explanation:
    The formula 1 + (xlen - wlen) // hop is derived from the way frames are extracted from the signal:
    - The first frame starts at sample 0 and ends at sample wlen-1.
    - Each subsequent frame starts hop samples after the previous frame's start.
    '''
    return 1 + (xlen - wlen) // hop

