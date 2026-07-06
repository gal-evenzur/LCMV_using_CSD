# LCMV_using_CSD

# First setup
To generate the complete dataset for training the Concurrent Speaker Detector (CSD) model, you need to run the scripts in the following order.

> download the TIMIT Dataset [here](https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) using a BitTorrent, or ask Gal(galevenzur2@gmail.com) for it. Next, create a /data/ folder inside the main folder. put the timit dataset inside the /data/ folder. Extract the TIMIT folder from the data/lisa/data/timit/raw/TIMIT (All the other folders are empty). 
Now you should have a *'TIMIT' folder inside the main/data/ folder.*

Now, you need the diffuse noise srs files, which emulate a noisy caffe ambience. 
> Unzip the file "main/data/Diff_noise_srs/Diff_srs.zip. Now you should have *5 wav files inside that folder.*

When you'll create a file using diffuse noise, you'll get an error about the np.complex_ in the anf_generator code. just change it for urself (temp fix).

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



# Dynamic Audio Generation
Here there are two scripts that generate dynamic audio files. The first one generates a single speaker moving from 0 to 180, for DOA analysis.
While the second one generates two speakers moving in a dynamic environment, for pipeline analysis!
**File to Run:** `createAudio/dynamic_test_wavs.py`
- **Purpose:** Generates a single moving speaker audio file for DOA analysis.

- **How to run?** 
First, create a python env using the requirements.txt file in the createAudio folder. Then run the following command in the terminal:
```bash
python createAudio/dynamic_test_wavs.py --num_samples 5 --t60 0.3 --snr 10 --output_dir data/simulated_audio/test/dynamic
```

- **Generates:**
    - **Mixed Audio:** `together_*.wav` (The main input for the model).
    - **Clean References:** `first_*.wav` (Individual speaker, used for validation).
    - **Labels:** `label_location_*.npy` (Contains VAD activity and spatial location data).
    - Also generates a `metadata_*.npz` file containing the T60 and SNR values for each generated sample.


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