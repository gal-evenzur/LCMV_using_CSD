# LCMV_using_CSD

# Dataset_Generation
To generate the complete dataset for training the Concurrent Speaker Detector (CSD) model, you need to run the scripts in the following order.

> download the TIMIT Dataset [here](https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) using a BitTorrent, or ask Gal(galevenzur2@gmail.com) for it 

### **Phase 1: Setup & Compilation (MATLAB)**

**File to Run:** `Eyal_Files/rir_generator.cpp`

- **Action:** You do not "run" this script directly; you **compile** it.
- **Purpose:** Creates the `rir_generator` binary (MEX file) required to simulate room acoustics.
- **Command:** `mex rir_generator.cpp`

### **Phase 2: Synthetic Audio Generation (MATLAB)**

**File to Run:** `Eyal_Files/create_data_base.m`

- **Purpose:** This is the main driver script. It generates thousands of synthetic audio files representing dynamic acoustic scenarios.
- **Generates:**
    - **Mixed Audio:** `together_*.wav` (The main input for the model).
    - **Clean References:** `first_*.wav` and `second_*.wav` (Individual speakers, used for validation).
    - **Labels:** `label_location_*.mat` (Contains VAD activity and spatial location data).
- **Dependencies:** This script automatically calls:
    - `create_locations_18_dynamic.m` to calculate speaker trajectories.
    - `fun_create_deffuse_noise.m` to generate ambient diffuse noise.

### **Phase 3: Feature Extraction (Python)**

**File to Run:** `Eyal_Files/python_codes/create_data_base.py`

- **Purpose:** Processes the raw WAV files from Phase 2 into the specific feature vectors required by the Neural Network.
- **Generates:**
    - **Features:** `feature_vector_*.npy` (STFT and spatial features).
    - **Labels:** `label_*.npy` (Speaker count labels) and `label2_*.npy` (Direction/Location labels).
    - **Index:** `idx.npy` (Keeps track of the total number of samples).
- **Note:** You must update the `data_root_dir` variable in this script to point to the folder where MATLAB saved the WAV files.

