# LCMV_using_CSD

# First setup
To generate the complete dataset for training the Concurrent Speaker Detector (CSD) model, you need to run the scripts in the following order.

> download the TIMIT Dataset [here](https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) using a BitTorrent, or ask Gal(galevenzur2@gmail.com) for it. Next, put the timit dataset inside the main folder. 

Now, you need the diffuse noise srs files, which emulate a noisy caffe ambience. 
> Unzip the file "createAudio/Diff_noise_srs/Diff_srs.zip. 

When you'll create a file using diffuse noise, you'll get an error about the np.complex_. just change it for urself (temp fix).

# Synthetic Audio Generation 

**File to Run: createAudio/create_data_base.py** ``
> Need to change what's written here
- **Purpose:** This is the main driver script. It generates thousands of synthetic audio files representing dynamic acoustic scenarios.
- **Generates:**
    - **Mixed Audio:** `together_*.wav` (The main input for the model).
    - **Clean References:** `first_*.wav` and `second_*.wav` (Individual speakers, used for validation).
    - **Labels:** `label_location_*.npy` (Contains VAD activity and spatial location data).
- **Dependencies:** This script automatically calls:
    - `create_locations_18_dynamic.m` to calculate speaker trajectories.
    - `fun_create_deffuse_noise.m` to generate ambient diffuse noise.

# Dataset train/val creation

**File to Run:** `DataSamples_to_InputVectors/create_data_base`

- **Purpose:** Processes the raw WAV files from Phase 2 into the specific feature vectors required by the Neural Network.
- **Generates:**
    - **Features:** `feature_vector_*.npy` (STFT and spatial features).
    - **Labels:** `label_*.npy` (Speaker count labels) and `label2_*.npy` (Direction/Location labels).
    - **Index:** `idx.npy` (Keeps track of the total number of samples).


# Model training
Before using the GPU's, you'll need to run this command in the terminal first:
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c 'import os, glob; print(":".join(glob.glob("/home/evenzug/Sim-venv/lib/python3.12/site-packages/nvidia/*/lib")))')`

_In order to make ur life easier, go to the activate file of the py venv, and paste the command at the bottom of the file._