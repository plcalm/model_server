#!/usr/bin/env bash
# Flask ResNet-50 启动
# 用法: bash flask_server.sh [workers=1]
set -euo pipefail
WORKERS="${1:-1}"
exec gunicorn \
    -w "${WORKERS}" -k gevent \
    --worker-connections 64 \
    -b 0.0.0.0:5000 \
    --timeout 120 \
    flask_resnet:app
