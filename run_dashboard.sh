#!/bin/bash

echo "=========================================="
echo "Session Token Abuse Detection Dashboard"
echo "=========================================="
echo ""

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "⚠️  Models not found. Training models first..."
    echo ""
    python src/training/train_pipeline.py
    if [ $? -ne 0 ]; then
        echo "❌ Training failed. Please check the error messages above."
        exit 1
    fi
    echo ""
    echo "✅ Models trained successfully!"
    echo ""
fi

# Install dashboard dependencies if needed
echo "Checking dependencies..."
pip install flask flask-cors -q

echo ""
echo "🚀 Starting dashboard server..."
echo ""
echo "📱 Dashboard will be accessible at:"
echo "   • http://localhost:8000 (local)"
echo "   • http://172.16.5.50:8000 (network)"
echo ""
echo "⚠️  Keep this terminal open - Press Ctrl+C to stop"
echo ""

# Start the Flask app
python app.py
