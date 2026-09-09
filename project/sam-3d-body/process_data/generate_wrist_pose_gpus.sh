#!/bin/bash
# Run from the release root so all stages share the same relative paths.

SCRIPT_PATH="project/sam-3d-body/process_data/generate_wrist_pose_gpus.py"
INPUT_ROOT="data/ego4d_fho/v2/segments"
OUTPUT_ROOT="data/ego4d_fho/v2/hdf5s/sam-3d-body"
MODEL_PATH="model/sam-3d-body-vith"
FRAME_INTERVAL=1

PIDS=()

# Stop the worker processes when the launcher is interrupted.
cleanup() {
    echo ""
    echo "Stopping all workers..."
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    exit 1
}

trap cleanup SIGINT SIGTERM

TOTAL_VIDEOS=$(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

if [ "$TOTAL_VIDEOS" -eq 0 ]; then
    echo "Error: No video directories found"
    exit 1
fi

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPU detected"
    exit 1
fi

VIDEOS_PER_GPU=$((TOTAL_VIDEOS / NUM_GPUS))
REMAINDER=$((TOTAL_VIDEOS % NUM_GPUS))

echo "=============================================="
echo "Total videos: $TOTAL_VIDEOS | GPU count: $NUM_GPUS"
echo "Press Ctrl+C to stop all workers"
echo "=============================================="

current_start=0

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    if [ $gpu_id -lt $REMAINDER ]; then
        count=$((VIDEOS_PER_GPU + 1))
    else
        count=$VIDEOS_PER_GPU
    fi
    
    if [ "$count" -eq 0 ]; then
        continue
    fi
    end=$((current_start + count - 1))
    echo "GPU $gpu_id: Video $current_start - $end"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --model_path "$MODEL_PATH" \
        --frame_interval $FRAME_INTERVAL \
        --start $current_start \
        --end $end \
        --gpu_id $gpu_id &
    
    PIDS+=($!)
    current_start=$((end + 1))
done

echo "=============================================="
echo "PIDs: ${PIDS[*]}"
echo "=============================================="

wait

echo "All workers finished!"
