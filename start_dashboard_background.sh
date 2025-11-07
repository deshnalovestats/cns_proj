#!/bin/bash

echo "======================================================================="
echo "SESSION TOKEN ABUSE DETECTION - BACKGROUND LAUNCHER"
echo "======================================================================="
echo ""

# Kill any existing Flask processes
pkill -f "python app.py" 2>/dev/null

# Start Flask in background
cd /home/jmayank/deshna
nohup python app.py > dashboard.log 2>&1 &
PID=$!

sleep 3

# Check if it's running
if ps -p $PID > /dev/null; then
    echo "✅ Dashboard started successfully!"
    echo ""
    echo "📱 Access at:"
    echo "   • http://localhost:8000"
    echo "   • http://172.16.5.50:8000"
    echo ""
    echo "📋 Process ID: $PID"
    echo "📝 Logs: dashboard.log"
    echo ""
    echo "To stop: pkill -f 'python app.py'"
    echo "To view logs: tail -f dashboard.log"
    echo ""
else
    echo "❌ Failed to start dashboard. Check dashboard.log for errors."
    exit 1
fi

echo "======================================================================="
