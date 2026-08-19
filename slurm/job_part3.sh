#!/bin/bash
#SBATCH --job-name=lab1-part3-ifs
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/part3_%j.out
#SBATCH --error=logs/part3_%j.err

# NOTE: Slurm does not create the directory in --output.  Run
#   mkdir -p ~/barnsley-fern-pytorch/logs ~/barnsley-fern-pytorch/out
# once before the first sbatch, or the job dies before it starts.
#
# Partition notes for Rangpur (check with `sinfo -s` before submitting):
#   comp3710 / a100 / a100-grind  -> the same a100-[0-9] nodes; same queue,
#                                    renaming does not shorten the wait
#   a100-test                     -> a100-a/b, usually idle, but QOS caps the
#                                    wall time at 20 min AND rejects --mem,
#                                    so it cannot run this 40 min / 32G job
#   p100                          -> older card, rarely queued

set -euo pipefail

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "started   : $(date -Is)"

source "$HOME/miniconda3/bin/activate"
conda activate torch

nvidia-smi -L
python -c "import torch; print('torch', torch.__version__,
           '| cuda', torch.cuda.is_available(),
           '|', torch.cuda.get_device_name(0))"

cd "$SLURM_SUBMIT_DIR"
python part3_figures.py --scale full --out out

echo "finished  : $(date -Is)"
ls -la out/
