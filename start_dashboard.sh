#!/bin/bash

echo "======================================================================="
echo "SESSION TOKEN ABUSE DETECTION - DASHBOARD LAUNCHER"
echo "======================================================================="
echo ""
echo "📱 Dashboard will be accessible at:"
echo "   • http://localhost:8000 (local access)"
echo "   • http://172.16.5.50:8000 (network access)"
echo ""
echo "⚠️  Keep this terminal open while using the dashboard"
echo "⚠️  Press Ctrl+C to stop the server"
echo ""
echo "Starting Dashboard..."
echo "======================================================================="
echo ""

# Activate conda environment and run Flask app
eval "$(conda shell.bash hook)"
conda activate tf 2>/dev/null || conda activate session_detection 2>/dev/null || true

# Run the Flask app directly
python app.py
