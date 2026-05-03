# ResNet-50 多框架推理部署与压测

分别在 **Flask (gunicorn+gevent)**、**FastAPI (uvicorn)** 和 **Triton (PyTriton + DynamicBatcher)** 三个框架上部署 ResNet-50 图片分类服务，并进行串行/并发压测对比。

---

## 环境要求

- Python 3.10
- CUDA / NVIDIA GPU（有 GPU 自动使用，无 GPU 自动切 CPU）
- conda 环境（推荐）

### 创建 conda 环境

```bash
conda create -n triton-server python=3.10
conda activate triton-server
pip install torch torchvision pillow requests numpy
pip install flask gunicorn gevent
pip install fastapi uvicorn pydantic
pip install nvidia-pytriton tritonclient
```

---

## 快速开始

```bash
# 启动 Flask 服务 (端口 5000)
bash run.sh flask

# 启动 FastAPI 服务 (端口 8000)
bash run.sh fastapi

# 启动 Triton 服务 (端口 7000 HTTP / 7001 gRPC)
bash run.sh triton
```

### 测试服务是否正常

```bash
# 方式一: 使用项目自带的 client
python triton_resnet.py client cat.jpg --http-port 7000

# 方式二: 用 curl 测试
curl -X POST http://localhost:5000/predict_base64 \
  -H "Content-Type: application/json" \
  -d "$(python -c "
import base64, json
b64 = base64.b64encode(open('cat.jpg','rb').read()).decode()
print(json.dumps({'image': b64}))
")"
```

---

## 各框架说明

| 框架 | 端口 | 启动命令 | 输入格式 | 特点 |
|------|------|----------|----------|------|
| Flask | 5000 | `gunicorn -k gevent flask_resnet:app` | JSON base64 | 简单直接，gevent 协程 |
| FastAPI | 8000 | `uvicorn fastapi_resnet:app` | JSON base64 | 异步原生，OpenAPI 文档 |
| Triton | 7000/7001 | `python triton_resnet.py server` | KServe tensor | 原生 DynamicBatcher 拼 batch |

### 输入格式

三个框架统一使用 base64 编码的图片作为输入：

```json
{"image": "/9j/4AAQ..."}
```

---

## 压测结果

测试环境: **NVIDIA GeForce GTX 1660 Ti** | 16 并发线程 × 1000 请求

```
----------------------------------------------------------------------------------------------------
Fw       Inst  GPU   构成            Ser_req/s  Ser_p50ms  Ser_p95ms  Con_req/s  Con_p50ms  Con_p95ms
----------------------------------------------------------------------------------------------------
FastAPI  1     264   模型×1             78.8       12.5       14.0       89.7       173.1      188.3
FastAPI  2     528   模型×2             76.6       12.9       13.9       120.6      130.0      209.0
FastAPI  4     1056  模型×4             74.1       12.7       14.1       63.0       220.5      521.9
Flask    1     264   模型×1             80.9       12.1       14.9       90.0       172.9      184.7
Flask    2     528   模型×2             77.6       12.6       14.1       121.3      128.6      147.9
Flask    4     1056  模型×4             74.9       12.9       15.0       63.7       245.4      300.8
Triton   1     400   py 264 + ts 136   72.7       13.6       15.2       123.7      126.3      133.3
Triton   2     536   py 400 + ts 136   68.5       14.4       15.9       180.5      86.9       112.4
Triton   4     648   py 512 + ts 136   66.6       14.7       16.3       197.2      75.3       123.0
----------------------------------------------------------------------------------------------------
```

### 关键结论

- **串行场景**: 三个框架差异不大 (~70-80 req/s)
- **并发场景**: Triton 优势明显，4 实例达到 **197 req/s**，Flask/FastAPI 4 worker 反而退化到 ~63 req/s（GPU 显存竞争）
- **显存效率**: Flask/FastAPI 每个 worker 独立加载模型 (+264 MB/worker)，Triton 多实例共享模型权重，边际成本仅 ~12 MB/实例
- **Triton 显存构成**: python 进程 (264→400→512 MB) + tritonserver 进程 (固定 136 MB)

---

## 运行压测

```bash
# 测试全部框架
python benchmark_concurrency.py \
  --python $(which python) \
  --env-dir $(dirname $(dirname $(which python))) \
  --instances 1 2 4

# 只测试 Triton
python benchmark_concurrency.py --framework Triton \
  --python $(which python) \
  --env-dir $(dirname $(dirname $(which python)))
```

---

## 项目结构

```
.
├── README.md                  ← 本文件
├── run.sh                     ← 一键启动脚本
├── benchmark_concurrency.py   ← 压测工具
├── benchmark_report.txt       ← 压测报告
├── request.json               ← 请求示例 (base64)
├── cat.jpg                    ← 测试图片
│
├── resnet_utils.py            ← 共享工具 (模型加载/预处理/top-5)
├── flask_resnet.py            ← Flask 服务
├── fastapi_resnet.py          ← FastAPI 服务
├── triton_resnet.py           ← PyTriton 服务 (含 server + client)
│
├── flask_server.sh            ← Flask 启动脚本
├── fastapi_server.sh          ← FastAPI 启动脚本
└── triton_server.sh           ← Triton 启动脚本
```
