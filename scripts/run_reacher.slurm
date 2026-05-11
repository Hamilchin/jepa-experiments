#!/bin/bash
#SBATCH --job-name=lewm-reacher-paper
#SBATCH --account=rao
#SBATCH --partition=ckpt-all
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/gscratch/rao/alexchin/le-wm
#SBATCH --output=/gscratch/rao/alexchin/le-wm/logs/%x_%j.out
#SBATCH --error=/gscratch/rao/alexchin/le-wm/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alexchin@uw.edu

# Reacher (DM Control) replication: paper §D, single L40S, batch 128, 10 epochs.
# Uses config/train/data/dmc.yaml.

set -euo pipefail

# --- node-local fast scratch ---
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID
mkdir -p $TMPDIR

# --- stage HDF5 dataset to node-local NVMe (93 GB, ~3 min) ---
export STABLEWM_HOME=$TMPDIR/stablewm
mkdir -p $STABLEWM_HOME/datasets
echo "[$(date)] staging dataset to $STABLEWM_HOME"
time cp /gscratch/rao/alexchin/stablewm/datasets/reacher.h5 $STABLEWM_HOME/datasets/
echo "[$(date)] staging done; $(du -sh $STABLEWM_HOME)"

# --- copy outputs back to durable storage on exit (success, error, or signal) ---
SAVE_DIR=/gscratch/rao/alexchin/stablewm/runs/reacher/$SLURM_JOB_ID
cleanup() {
  echo "[$(date)] cleanup: copying outputs to $SAVE_DIR"
  mkdir -p $SAVE_DIR
  for d in $STABLEWM_HOME/*/; do
    name=$(basename $d)
    [ "$name" = "datasets" ] && continue
    cp -r $d $SAVE_DIR/ 2>/dev/null || true
  done
  ls -lh $SAVE_DIR/ 2>/dev/null || true
  rm -rf $TMPDIR
}
trap cleanup EXIT TERM INT USR1

# --- environment ---
export LAB=/gscratch/rao/alexchin
export HF_HOME=/gscratch/rao/alexchin/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export PIP_CACHE_DIR=/gscratch/rao/alexchin/pip-cache
export WANDB_DIR=/gscratch/rao/alexchin/wandb-offline
export SPT_CACHE_DIR=/gscratch/rao/alexchin/cache/stable-pretraining

cd /gscratch/rao/alexchin/le-wm
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python train.py \
    data=dmc \
    trainer.max_epochs=10 \
    wandb.config.name=$SLURM_JOB_NAME