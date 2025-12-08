#!/bin/bash
#SBATCH --job-name=sdp_crown_debug
#SBATCH --partition=gpu_a100 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=0:30:00
#SBATCH --output=/projects/prjs1681/runs/slurm_outputs/sdpcrown_debug_%j.out
#SBATCH --error=/projects/prjs1681/runs/slurm_outputs/sdpcrown_debug_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

echo "Loaded modules:"
module list 2>&1

cd /home/jvrijn/code/rs/verif/SDP-CROWN

source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
conda activate __av__sdpcrown

echo "Python location: $(which python)"
echo "Python version: $(python --version)"

#theirs .pth
#python /home/jvrijn/code/rs/verif/SDP-CROWN/sdp_crown.py   --model /home/jvrijn/code/rs/verif/SDP-CROWN/models/cifar10_convlarge.pth   --config /home/jvrijn/code/rs/verif/SDP-CROWN/SDP_CROWN_av_example_config.yaml   --vnnlib_property ./tmp_sdp_test/sample_0.vnnlib
#ours .pth
python /home/jvrijn/code/rs/verif/SDP-CROWN/sdp_crown.py   --model /home/jvrijn/code/rs/verif/SDP-CROWN/models/conv_big_best.pth   --config /home/jvrijn/code/rs/verif/SDP-CROWN/SDP_CROWN_av_example_config.yaml   --vnnlib_property ./tmp_sdp_test/sample_0.vnnlib
#ours onnx
#python /home/jvrijn/code/rs/verif/SDP-CROWN/sdp_crown.py   --model /projects/prjs1681/models/CIFAR-10/conv_big_standard.onnx   --config /home/jvrijn/code/rs/verif/SDP-CROWN/SDP_CROWN_av_example_config.yaml   --vnnlib_property ./tmp_sdp_test/sample_0.vnnlib

#ours with conversion that removes reshape 
#python /home/jvrijn/code/rs/verif/SDP-CROWN/sdp_crown.py   --model /home/jvrijn/code/rs/verif/SDP-CROWN/models/conv_big_from_onnx.pth   --config /home/jvrijn/code/rs/verif/SDP-CROWN/SDP_CROWN_av_example_config.yaml   --vnnlib_property ./tmp_sdp_test/sample_0.vnnlib
# Print completion information
echo "Job completed at: $(date)"
echo "Exit code: $?"

# Clean up resources
echo "Cleaning up resources..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Deactivate conda environment
conda deactivate

echo "Final resource usage:"
echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"

echo "Job finished successfully."