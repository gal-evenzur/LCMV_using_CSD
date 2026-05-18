#!/bin/bash
set -e

# Get the absolute path to the workspace directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Default arguments 
N_TRAIN=${1:-200}
N_VAL=${2:-20}

# Start indices
START_IDX_TRAIN=1
START_IDX_VAL=1

# Lottery calculation for DataSamples_to_InputVectors script
# lottery parameter in that script represents "number of files to process per data set" + 1
# so we pass N_TRAIN + 1 and N_VAL + 1. (since it loops to `lottery` not inclusive).
LOTTERY_TRAIN=$((N_TRAIN + 1))
LOTTERY_VAL=$((N_VAL + 1))

echo "========================================"
echo " Generating TRAIN dataset... "
echo "========================================"
echo "1. Generating simulated audio for TRAIN (num_samples=$N_TRAIN)"
python3 "$WORKSPACE_DIR/createAudio/create_data_base.py" --num_samples $N_TRAIN --start_idx $START_IDX_TRAIN --trainORval train

echo "2. Converting DataSamples to Input vectors for TRAIN"
python3 "$WORKSPACE_DIR/DataSamples_to_InputVectors/create_data_base.py" --mode train --nom_data_sets 2 --lottery $LOTTERY_TRAIN

echo ""
echo "========================================"
echo " Generating VAL dataset... "
echo "========================================"
echo "1. Generating simulated audio for VAL (num_samples=$N_VAL)"
python3 "$WORKSPACE_DIR/createAudio/create_data_base.py" --num_samples $N_VAL --start_idx $START_IDX_VAL --trainORval val

echo "2. Converting DataSamples to Input vectors for VAL"
python3 "$WORKSPACE_DIR/DataSamples_to_InputVectors/create_data_base.py" --mode val --nom_data_sets 2 --lottery $LOTTERY_VAL

echo ""
echo "========================================"
echo " Done generating complete datasets! "
echo "========================================"

