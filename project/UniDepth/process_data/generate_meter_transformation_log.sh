#!/bin/bash
# Run from the release root so all stages share the same relative paths.

SCRIPT_PATH="project/UniDepth/process_data/generate_meter_transformation.py"
REFERENCE_ROOT="data/ego4d_fho/v2/hdf5s/vggt_overlap1"
OUTPUT_ROOT="data/ego4d_fho/v2/hdf5s/unidepth_overlap1"
MODEL_PATH="model/unidepth-v2-vitl14"
MIN_VALID_RATIO=0.9

LOG_DIR="project/UniDepth/process_data/log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/main_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Mirror launcher messages to the persistent run log.
log() {
    echo "$1" | tee -a "$MAIN_LOG"
}

PIDS=()

# Stop the worker processes when the launcher is interrupted.
cleanup() {
    log ""
    log "Stopping all workers..."
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    exit 1
}

trap cleanup SIGINT SIGTERM

TOTAL_FILES=$(find "$REFERENCE_ROOT" -maxdepth 1 -name "*.h5" | wc -l)
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

if [ "$TOTAL_FILES" -eq 0 ]; then
    log "Error: No HDF5 files found"
    exit 1
fi

if [ "$NUM_GPUS" -eq 0 ]; then
    log "Error: No GPU detected"
    exit 1
fi

FILES_PER_GPU=$((TOTAL_FILES / NUM_GPUS))
REMAINDER=$((TOTAL_FILES % NUM_GPUS))

log "=============================================="
log "Start time: $(date)"
log "Total HDF5 files: $TOTAL_FILES | GPU count: $NUM_GPUS"
log "Minimum valid-frame fraction: $MIN_VALID_RATIO"
log "Main log file: $MAIN_LOG"
log "Press Ctrl+C to stop all workers"
log "=============================================="

current_start=0

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    if [ $gpu_id -lt $REMAINDER ]; then
        count=$((FILES_PER_GPU + 1))
    else
        count=$FILES_PER_GPU
    fi
    
    end=$((current_start + count - 1))

    GPU_LOG="${LOG_DIR}/gpu${gpu_id}_${TIMESTAMP}.log"
    
    log "GPU $gpu_id: HDF5 $current_start - $end -> $GPU_LOG"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
        --reference_root "$REFERENCE_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --model_path "$MODEL_PATH" \
        --min_valid_ratio $MIN_VALID_RATIO \
        --start $current_start \
        --end $end \
        --gpu_id $gpu_id 2>&1 | tee -a "$GPU_LOG" &
    
    PIDS+=($!)
    current_start=$((end + 1))
done

log "=============================================="
log "PIDs: ${PIDS[*]}"
log "=============================================="

wait

log "=============================================="
log "All workers finished! End time: $(date)"
log "=============================================="
