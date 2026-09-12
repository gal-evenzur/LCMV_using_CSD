#!/usr/bin/env bash
# =============================================================================
# run_dynamic_paper_tests.sh
#
# Orchestrates the full LCMV-NN pipeline for all three dynamic-paper test groups:
#
#   GROUP 1 – Varying SNR_diffuse  (T60 = 0.2, SNR = 30 / 20 / 10 / 3)
#   GROUP 2 – Varying T60          (SNR = 30, T60 = 0.2 / 0.4 / 0.6 / 0.8)
#   GROUP 3 – Varying closest_ang_diff  (SNR = 30, T60 = 0.2, diff = 40 / 30 / 20)
#
# For each condition the script:
#   1. Generates 10 WAV files  (create_dynamic_paperlike.py)
#   2. Runs the NN tracking pipeline   (pipeline.py)
#   3. Runs the LCMV beamformer        (pipeline_beamformer.py)
#
# Data layout  (relative to WORKSPACE_DIR):
#   data/simulated_audio/test/dynamic_paper_tests/<condition>/
#
# Results layout:
#   pipeline_results/dynamic_paper_tests/<condition>/
#
# Usage:
#   cd <workspace_dir>          # the directory that contains src/, data/, etc.
#   bash run_dynamic_paper_tests.sh
#
# Or override the workspace root:
#   WORKSPACE_DIR=/path/to/project bash run_dynamic_paper_tests.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Location of this script's directory: .../LCMV_using_CSD/pipeline_results/dynamic_paper_tests
CURRENT_DIR="$(dirname "$(realpath "$0")")"

# Project root (2 levels up: .../workspace_root containing LCMV_using_CSD and data)
WORKSPACE_DIR="/home/evenzug/LCMV_using_CSD"

DATA_ROOT="${WORKSPACE_DIR}/data/simulated_audio/test/dynamic_paper_tests"
RESULTS_ROOT="${WORKSPACE_DIR}/pipeline_results/dynamic_paper_tests"

# Updated Python file paths
GEN_SCRIPT="${WORKSPACE_DIR}/createAudio/create_dynamic_paperlike.py"
NN_SCRIPT="${WORKSPACE_DIR}/pipeline.py"
BF_SCRIPT="${WORKSPACE_DIR}/pipeline_beamformer.py"

NUM_SAMPLES=10
START_IDX=1
END_IDX=$((START_IDX + NUM_SAMPLES - 1))   # inclusive; = 10

# ---------------------------------------------------------------------------
# Helper: run one full condition (generate → NN → beamformer)
# ---------------------------------------------------------------------------
run_condition() {
    local LABEL="$1"          # e.g. "dynamic_SNR=30_T60=0.2"
    local SNR="$2"
    local T60="$3"
    local ANG_DIFF="$4"       # closest_ang_diff (degrees)

    local DATA_DIR="${DATA_ROOT}/${LABEL}"
    local RESULTS_DIR="${RESULTS_ROOT}/${LABEL}"

    echo ""
    echo "============================================================"
    echo " CONDITION: ${LABEL}"
    echo "   SNR_diffuse=${SNR}  T60=${T60}  closest_ang_diff=${ANG_DIFF}"
    echo "   Data    -> ${DATA_DIR}"
    echo "   Results -> ${RESULTS_DIR}"
    echo "============================================================"

    # ------------------------------------------------------------------
    # STEP 1: Generate audio
    # ------------------------------------------------------------------
    echo "[1/3] Generating ${NUM_SAMPLES} WAV files..."
    source /home/evenzug/LCMV_using_CSD/createAudio/.audio-env/bin/activate

    python "${GEN_SCRIPT}" \
        --num_samples  "${NUM_SAMPLES}" \
        --start_idx    "${START_IDX}" \
        --SNR_diffuse  "${SNR}" \
        --T60          "${T60}" \
        --closest_ang_diff "${ANG_DIFF}" \
        --dataset_title "${LABEL}"

    # ------------------------------------------------------------------
    # STEP 2: NN tracking pipeline
    # ------------------------------------------------------------------
    deactivate

    echo "[2/3] Running NN pipeline..."
    python "${NN_SCRIPT}" \
        --start_idx          "${START_IDX}" \
        --end_idx            "${END_IDX}" \
        --folder_to_test_data  "${DATA_DIR}" \
        --folder_to_results    "${RESULTS_DIR}"

    # ------------------------------------------------------------------
    # STEP 3: LCMV beamformer
    # Note: folder_to_results is the SAME directory — the beamformer reads
    # estimate_CSD/DOA_N.npy and true_CSD_N.npy written by the NN pipeline,
    # then appends its own outputs (separating_speaker_*.wav, CSVs, plots).
    # ------------------------------------------------------------------
    echo "[3/3] Running beamformer..."
    python "${BF_SCRIPT}" \
        --start_idx            "${START_IDX}" \
        --end_idx              "${END_IDX}" \
        --folder_to_test_data  "${DATA_DIR}" \
        --folder_to_results    "${RESULTS_DIR}" \
        --eval_mode            overlap \
        --verbose              1

    echo " -> DONE: ${LABEL}"

    python "${WORKSPACE_DIR}/run_experiments.py" \
        --start_idx            "${START_IDX}" \
        --end_idx              "${END_IDX}" \
        --folder_to_test_data  "${DATA_DIR}" \
        --folder_to_results    "${RESULTS_DIR}" \
        --eval_mode            overlap \
        --verbose              1
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
for f in "${GEN_SCRIPT}" "${NN_SCRIPT}" "${BF_SCRIPT}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Script not found: $f" >&2
        exit 1
    fi
done

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}"

# ---------------------------------------------------------------------------
# GROUP 1: Varying SNR_diffuse  (T60 fixed = 0.2)
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo " GROUP 1: Varying SNR_diffuse  (T60=0.2)"
echo "################################################################"

# for SNR in 20 10 3; do
#     run_condition "dynamic_SNR=${SNR}_T60=0.2" "${SNR}" "0.2" "50"
# done

# ---------------------------------------------------------------------------
# GROUP 2: Varying T60  (SNR_diffuse fixed = 30)
# NOTE: T60=0.2 with SNR=30 was already generated in Group 1 above.
#       We skip it here to avoid regenerating identical data.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo " GROUP 2: Varying T60  (SNR_diffuse=30)"
echo "################################################################"

# for T60 in 0.4 0.6 0.8; do
#     run_condition "dynamic_SNR=30_T60=${T60}" "30" "${T60}" "50"
# done

# ---------------------------------------------------------------------------
# GROUP 3: Varying closest_ang_diff  (SNR=30, T60=0.2)
# NOTE: SNR=30/T60=0.2 audio from Group 1 is reused here (same data_dir).
#       We only need to run the NN pipeline and beamformer with a different
#       results directory, but the label requires a new subdirectory so that
#       results from different ang_diff values don't overwrite each other.
#       Because the AUDIO is the same we SKIP re-generation for ang_diff
#       tests — closest_ang_diff only affects audio generation geometry,
#       so we DO regenerate with separate data dirs per ang_diff value.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo " GROUP 3: Varying closest_ang_diff  (SNR=30, T60=0.2)"
echo "################################################################"

# for ANG in 40 30 20; do
#     run_condition "dynamic_SNR=30_T60=0.2_angdiff=${ANG}" "30" "0.2" "${ANG}"
# done
run_condition "dynamic_SNR=10_T60=0.3_angdiff=60" "10" "0.3" "60"
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " ALL CONDITIONS COMPLETE."
echo " Results under: ${RESULTS_ROOT}"
echo "============================================================"