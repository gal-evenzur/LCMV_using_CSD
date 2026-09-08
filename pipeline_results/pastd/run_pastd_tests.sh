#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/evenzug/LCMV_using_CSD"
TEST_DIR="${REPO_ROOT}/data/simulated_audio/test/static"
RESULTS_DIR="${REPO_ROOT}/pipeline_results/pastd"
CSV_PATH="${RESULTS_DIR}/total_beamformer_results_time_graph.csv"
PIPELINE_START_IDX=2000

mkdir -p "${TEST_DIR}" "${RESULTS_DIR}"
rm -f "${CSV_PATH}"

START_IDX=2000
SEED_BASE=2
SWEEP_INDEX=0

run_sweep() {
  local label="$1"
  local snr="$2"
  local t60="$3"
  local audio_length="$4"
  local sweep_seed=$((SEED_BASE + SWEEP_INDEX * 101))
  local end_idx=$((START_IDX + 10))

  echo "=== ${label} | runs ${START_IDX}-${end_idx} | SNR=${snr} | T60=${t60} | length=${audio_length}s | seed=${sweep_seed} ==="

  python "${REPO_ROOT}/createAudio/create_test_wavs.py" \
    --num_samples 10 \
    --start_idx "${START_IDX}" \
    --seed "${sweep_seed}" \
    --SNR "${snr}" \
    --T60 "${t60}" \
    --audio_length "${audio_length}" \
    --output_path "${TEST_DIR}"

  python "${REPO_ROOT}/run_experiments.py" \
    --start_idx "${START_IDX}" \
    --end_idx "${end_idx}" \
    --folder_to_test_data "${TEST_DIR}" \
    --results_dir "${RESULTS_DIR}"

  START_IDX=$((end_idx + 1))
  SWEEP_INDEX=$((SWEEP_INDEX + 1))
}

# run_sweep "T60 sweep A" 30 0.3 10
# run_sweep "T60 sweep B" 30 0.5 10
# run_sweep "T60 sweep C" 30 0.8 10

# run_sweep "SNR sweep A" 3 0.2 10
# run_sweep "SNR sweep B" 10 0.2 10
# run_sweep "SNR sweep C" 30 0.2 10

# Run for length = 3, 10:10:60 seconds
run_sweep "Length sweep A" 10 0.3 3
run_sweep "Length sweep B" 10 0.3 10
run_sweep "Length sweep C" 10 0.3 20
run_sweep "Length sweep D" 10 0.3 30
run_sweep "Length sweep E" 10 0.3 40
run_sweep "Length sweep F" 10 0.3 50
run_sweep "Length sweep G" 10 0.3 60

python "${REPO_ROOT}/pipeline_BF_PASTD.py" \
  --start_idx "${PIPELINE_START_IDX}" \
  --end_idx $((START_IDX - 1)) \
  --folder_to_test_data "${TEST_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --csv_path "${CSV_PATH}" \
  --methods both \
  --verbose 1
