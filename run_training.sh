#!/bin/bash
# Script to run training pipeline with live output

# Activate conda environment and run training
eval "$(conda shell.bash hook)"
conda activate tf
python -u src/training/train_pipeline.py
