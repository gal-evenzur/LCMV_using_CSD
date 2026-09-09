#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/evenzug/LCMV_using_CSD"
TEST_DIR="${REPO_ROOT}/data/simulated_audio/test/static_presentation"
RESULTS_DIR="${REPO_ROOT}/pipeline_results/static_presentation"
CSV_PATH="${RESULTS_DIR}/results/results.csv"
PIPELINE_START_IDX=1

mkdir -p "${TEST_DIR}" "${RESULTS_DIR}"
rm -f "${CSV_PATH}"

START_IDX=200
SEED_BASE=320
SWEEP_INDEX=0
NUM_FILES=25

run_sweep() {
  local label="$1"
  local snr="$2"
  local t60="$3"
  local sweep_seed=$((SEED_BASE + SWEEP_INDEX * 101))
  local end_idx=$((START_IDX + NUM_FILES - 1))

  echo "=== ${label} | runs ${START_IDX}-${end_idx} | SNR=${snr} | T60=${t60}  | seed=${sweep_seed} ==="

  python "${REPO_ROOT}/createAudio/create_test_wavs.py" \
    --num_samples "${NUM_FILES}" \
    --start_idx "${START_IDX}" \
    --seed "${sweep_seed}" \
    --SNR "${snr}" \
    --T60 "${t60}" \
    --output_path "${TEST_DIR}"

  python "${REPO_ROOT}/run_experiments.py" \
    --start_idx "${START_IDX}" \
    --end_idx "${end_idx}" \
    --folder_to_test_data "${TEST_DIR}" \
    --results_dir "${RESULTS_DIR}"

  START_IDX=$((end_idx + 1))
  SWEEP_INDEX=$((SWEEP_INDEX + 1))
}

# run_sweep "T60 sweep A" 10 0.3
# run_sweep "T60 sweep B" 10 0.5 
# run_sweep "T60 sweep C" 10 0.8 
# run_sweep "T60 sweep C" 10 1.2 

# run_sweep "SNR sweep A" 0 0.3 
# run_sweep "SNR sweep A" 5 0.3 
# run_sweep "SNR sweep B" 10 0.3 
# run_sweep "SNR sweep C" 30 0.3 


python "${REPO_ROOT}/pipeline_beamformer.py" \
  --start_idx "${PIPELINE_START_IDX}" \
  --end_idx $((START_IDX)) \
  --folder_to_test_data "${TEST_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --csv_path "${CSV_PATH}" \
  --methods pastd \
  --verbose 1