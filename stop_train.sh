#!/usr/bin/env bash
# Stop the currently running training process.

PIDFILE="/tmp/autoresearch_train.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "No training run found (no PID file)."
    exit 0
fi

PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping training (PID $PID)..."
    kill "$PID"
    rm -f "$PIDFILE"
    echo "Done."
else
    echo "Training process $PID is not running (already finished?)."
    rm -f "$PIDFILE"
fi
