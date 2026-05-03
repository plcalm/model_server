#!/usr/bin/env bash
set -e

ENV_DIR="/home/pl/Tools/miniforge3/envs/triton-server"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$ENV_DIR/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-help}" in
  flask)
    echo "Starting Flask (gunicorn+gevent) on port 5000..."
    exec "$ENV_DIR/bin/gunicorn" \
      --workers 1 --worker-class gevent \
      --worker-connections 64 \
      --bind 0.0.0.0:5000 --timeout 120 \
      --access-logfile - \
      flask_resnet:app
    ;;
  fastapi)
    echo "Starting FastAPI (uvicorn) on port 8000..."
    exec "$ENV_DIR/bin/uvicorn" \
      fastapi_resnet:app \
      --host 0.0.0.0 --port 8000 \
      --workers 1 --timeout-keep-alive 65
    ;;
  triton)
    echo "Starting PyTriton ResNet-50 on port 7000 (HTTP) / 7001 (gRPC)..."
    exec "$ENV_DIR/bin/python" triton_resnet.py server
    ;;
  *)
    echo "Usage: $0 {flask|fastapi|triton}"
    echo ""
    echo "  flask   — gunicorn+gevent on :5000"
    echo "  fastapi — uvicorn on :8000"
    echo "  triton  — PyTriton on :7000/:7001"
    exit 1
    ;;
esac
