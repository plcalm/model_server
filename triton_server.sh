#!/usr/bin/env bash
# Triton ResNet-50 启动 (PyTriton DynamicBatcher)
# 用法: bash triton_server.sh [instances=1]
set -euo pipefail
INSTANCES="${1:-1}"
exec python triton_resnet.py server --instances "${INSTANCES}"
