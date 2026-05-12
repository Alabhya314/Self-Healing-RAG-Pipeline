#!/bin/bash
set -e

# Cleanup background processes on exit
trap "exit" INT TERM
trap "kill 0" EXIT

# 1. Start FastAPI backend in background[cite: 1]
echo "🚀 Starting FastAPI backend on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# 2. Wait for backend availability
sleep 5

# 3. Start Streamlit frontend on Railway's dynamic $PORT[cite: 1, 2]
echo "🎨 Starting Streamlit frontend on port $PORT..."
streamlit run app.py --server.port $PORT --server.address 0.0.0.0