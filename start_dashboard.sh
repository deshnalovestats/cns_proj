#!/bin/bash
# Script to run the Flask dashboard

echo "Starting Session Token Abuse Detection Dashboard..."
echo "Dashboard will be available at: http://localhost:5000"
echo ""

# Activate conda environment and run Flask app
eval "$(conda shell.bash hook)"
conda activate tf
python app.py
