#!/bin/bash
# Complete Project Setup and Demo Script

echo "=========================================="
echo "Session Token Abuse Detection System"
echo "Complete Setup and Demo"
echo "=========================================="
echo ""

# Set the conda environment
CONDA_ENV="tf"

echo "Step 1: Checking environment..."
if conda env list | grep -q "$CONDA_ENV"; then
    echo "✓ Conda environment '$CONDA_ENV' found"
else
    echo "✗ Conda environment '$CONDA_ENV' not found"
    echo "  Please create it first: conda create -n tf python=3.9"
    exit 1
fi

echo ""
echo "Step 2: Installing dependencies..."
conda run -n $CONDA_ENV pip install -q -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "Step 3: Generating training data..."
conda run -n $CONDA_ENV python src/preprocessing/data_generator.py
echo "✓ Training data generated"

echo ""
echo "Step 4: Training all models..."
echo "   (This may take 5-10 minutes depending on your hardware)"
conda run -n $CONDA_ENV python src/training/train_pipeline.py
echo "✓ Models trained"

echo ""
echo "Step 5: Running detection on test data..."
conda run -n $CONDA_ENV python src/inference/detect.py \
    --input data/raw/session_logs.csv \
    --output outputs/detection
echo "✓ Detection complete"

echo ""
echo "Step 6: Generating visualizations..."
# Find the most recent detection results file
RESULTS_FILE=$(ls -t outputs/detection/detection_results_*.csv 2>/dev/null | head -1)
if [ -n "$RESULTS_FILE" ]; then
    conda run -n $CONDA_ENV python src/utils/visualization.py \
        --results "$RESULTS_FILE" \
        --metrics outputs/reports/evaluation_metrics.json \
        --output outputs/visualizations
    echo "✓ Visualizations generated"
else
    echo "⚠ Detection results file not found, skipping visualization"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Project structure:"
echo "  📁 data/raw/          - Training data"
echo "  📁 data/processed/    - Feature-engineered data"
echo "  📁 models/            - Trained AI models"
echo "  📁 outputs/reports/   - Evaluation metrics"
echo "  📁 outputs/detection/ - Detection results"
echo "  📁 outputs/visualizations/ - Charts and graphs"
echo ""
echo "Key files:"
echo "  📄 README.md          - Complete documentation"
echo "  📄 QUICKSTART.md      - Quick start guide"
echo "  📄 ARCHITECTURE.md    - System design"
echo "  📄 config.yaml        - Configuration"
echo ""
echo "Next steps:"
echo "  1. Review outputs/reports/evaluation_metrics.json"
echo "  2. Check outputs/visualizations/ for charts"
echo "  3. Read QUICKSTART.md for usage examples"
echo "  4. Try: conda run -n tf python src/inference/detect.py --help"
echo ""
echo "Happy detecting! 🛡️"
