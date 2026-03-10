#!/usr/bin/env bash
# Start a training run in the background.
# Output streams to /tmp/autoresearch_train.log
# Training survives terminal/remote-control disconnects.

LOGFILE="/tmp/autoresearch_train.log"
PIDFILE="/tmp/autoresearch_train.pid"

# Kill any existing training run
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing training run (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 1
    fi
    rm -f "$PIDFILE"
fi

cd "$(dirname "$0")"

echo "Starting training run..."
echo "Log: $LOGFILE"

nohup ~/.local/bin/uv run train.py > "$LOGFILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PIDFILE"

echo "Training PID: $TRAIN_PID"
echo "Run './watch_train.sh' to stream output (Ctrl+C to stop watching without stopping training)"
