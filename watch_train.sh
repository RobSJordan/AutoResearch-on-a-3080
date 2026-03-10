#!/usr/bin/env bash
# Stream training output. Ctrl+C stops watching but does NOT stop training.

LOGFILE="/tmp/autoresearch_train.log"
PIDFILE="/tmp/autoresearch_train.pid"

if [ ! -f "$LOGFILE" ]; then
    echo "No training log found at $LOGFILE. Run ./start_train.sh first."
    exit 1
fi

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Training is running (PID $PID). Streaming output (Ctrl+C to stop watching)..."
    else
        echo "Training has finished. Showing log:"
    fi
else
    echo "Showing log (no PID file found):"
fi

echo "---"
tail -f "$LOGFILE"
