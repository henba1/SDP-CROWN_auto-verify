#!/bin/bash
#SBATCH --job-name=sdp_crown
#SBATCH --partition=gpu_a100 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/prjs1681/runs/slurm_outputs/SDP-CROWN_%j.out
#SBATCH --error=/projects/prjs1681/runs/slurm_outputs/SDP-CROWN_%j.err

#h100 for now, need to check mem usage TODO
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

cd /home/jvrijn/code/rs/verif/SDP-CROWN

source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
#conda activate alpha-beta-crown
conda activate __av__sdpcrown

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Default params
MODEL="cifar10_convlarge"
RADIUS="8/255"
START=0
END=10
LR_ALPHA=0.5
LR_LAMBDA=0.05

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --radius)
            RADIUS="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --lr_alpha)
            LR_ALPHA="$2"
            shift 2
            ;;
        --lr_lambda)
            LR_LAMBDA="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Generate timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Sanitize model name for filename
SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9_-]/_/g')

echo "Running SDP-CROWN:"
echo "  model: $MODEL"
echo "  radius: $RADIUS"
echo "  range: $START-$END"
echo "  lr_alpha: $LR_ALPHA"
echo "  lr_lambda: $LR_LAMBDA"
echo "  output: $OUTPUT_FILE"

python sdp_crown.py \
    --model "$MODEL" \
    --radius "$RADIUS" \
    --start $START \
    --end $END \
    --lr_alpha $LR_ALPHA \
    --lr_lambda $LR_LAMBDA

echo "Completed at: $(date)"
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
conda deactivate

