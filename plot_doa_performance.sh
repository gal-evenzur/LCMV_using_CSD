#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# set -e 

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
NUM_SAMPLES_PER_T60=4      # Number of audio files per T60 environment
BASE_SEED=411                # Starting seed
VELOCITY=0.3                # Linear velocity of the speaker
SNR_DIFFUSE=20              # SNR setting
START_ANGLE=0
END_ANGLE=180

# The specific T60 values we are testing
T60_VALUES=(0.3 0.4 0.6)

# Track the global file index so files don't overwrite each other
CURRENT_START_IDX=1000

echo "====================================================="
echo " STARTING DOA EXPERIMENT BATCH PIPELINE"
echo "====================================================="


for t60 in "${T60_VALUES[@]}"; do
    echo ""
    echo ">>> PHASE 1: GENERATING DATA FOR T60 = ${t60}s"
    echo "-----------------------------------------------------"
    
    source /home/evenzug/LCMV_using_CSD/createAudio/.audio-env/bin/activate
    
    # python createAudio/dynamic_test_wavs.py \
    #     --num_samples $NUM_SAMPLES_PER_T60 \
    #     --start_idx $CURRENT_START_IDX \
    #     --seed $BASE_SEED \
    #     --T60 $t60 \
    #     --SNR_diffuse $SNR_DIFFUSE \
    #     --linear_velocity $VELOCITY \
    #     --start_angle_deg $START_ANGLE \
    #     --end_angle_deg $END_ANGLE

    CURRENT_END_IDX=$((CURRENT_START_IDX + NUM_SAMPLES_PER_T60)) 

    echo ""
    echo ">>> PHASE 2: RUNNING TRACKING PIPELINE (Indices $CURRENT_START_IDX to $CURRENT_END_IDX)"
    echo "-----------------------------------------------------"
    
    deactivate

    python pipeline.py --start_idx $CURRENT_START_IDX --end_idx $CURRENT_END_IDX

    # Update indices and seed for the next T60 loop
    CURRENT_START_IDX=$((CURRENT_END_IDX + 1))
    BASE_SEED=$((BASE_SEED + 100)) 
done

echo ""
echo "====================================================="
echo " ALL DATA GENERATED AND PROCESSED."
echo ">>> PHASE 3: AGGREGATING RESULTS AND PLOTTING"
echo "====================================================="

# This runs the final plotting script you will create to analyze the numpy outputs
python plot_doa_performance.py --t60_list 0.3 0.4 0.6

echo "Pipeline complete."