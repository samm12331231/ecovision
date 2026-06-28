#!/bin/bash
# Starts the backend (FastAPI) and frontend (Streamlit) together.
# Run from the project root with your conda env active:
#     bash start_app.sh
#
# The backend loads the model ONCE at startup, so after a retrain you must
# stop this (Ctrl+C) and run it again to pick up the new weights.

set -e
cd "$(dirname "$0")"

# Start backend in the background, logging to backend.log
uvicorn backend.main:app --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ Backend starting on http://localhost:8000 (logs: backend.log)"

# Stop the backend when this script is interrupted
trap "echo; echo 'Stopping backend...'; kill $BACKEND_PID 2>/dev/null" EXIT

# Give the backend a few seconds to load the models
sleep 4

# Start the frontend in the foreground (Ctrl+C stops everything)
echo "✅ Launching frontend on http://localhost:8501"
python -m streamlit run frontend/app.py
