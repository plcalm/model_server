#!/usr/bin/env bash
# FastAPI ResNet-50 启动
# 用法: bash fastapi_server.sh [workers=1]
set -euo pipefail
WORKERS="${1:-1}"
exec uvicorn \
    --workers "${WORKERS}" \
    --host 0.0.0.0 \
    --port 8000 \
    --timeout-keep-alive 65 \
    fastapi_resnet:app
