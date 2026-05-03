"""
PyTriton ResNet-50 部署 (base64 输入, DynamicBatcher)

端口 7000: HTTP (base64 bytes tensor → infer_fn 内部解码 + 预处理)
端口 7001: gRPC

启动: python triton_resnet.py server
"""

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from resnet_utils import (
    _load_labels, load_model, preprocess, warmup,
)


# ---------------------------------------------------------------------------
# PyTriton 服务端
# ---------------------------------------------------------------------------

def _decode_and_preprocess(b64_bytes: bytes) -> np.ndarray:
    """base64 bytes → preprocessed tensor (1, 3, 224, 224)"""
    decoded = base64.b64decode(b64_bytes)
    img = Image.open(io.BytesIO(decoded))
    return preprocess(img)


def start_server(http_port=7000, grpc_port=7001, instances=1):
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ResNet-50 on {device_}...")
    model = load_model().to(device_)
    warmup(model, device_, n=3)
    labels = _load_labels()
    print(f"Loaded {len(labels)} labels.")

    @batch
    def infer_fn(**inputs: np.ndarray) -> dict[str, np.ndarray]:
        """base64 bytes tensor → 解码+预处理 → 推理。"""
        b64_batch = inputs["image_b64"]  # (batch_size, 1)
        batch_size = b64_batch.shape[0]

        tensors = []
        for i in range(batch_size):
            tensors.append(_decode_and_preprocess(b64_batch[i, 0]))
        input_tensor = torch.from_numpy(np.concatenate(tensors, axis=0)).to(device_)

        with torch.no_grad():
            logits = model(input_tensor).cpu().numpy()
        return {"logits": logits}

    config = TritonConfig(
        http_address="0.0.0.0",
        http_port=http_port,
        grpc_port=grpc_port,
    )

    model_config = ModelConfig(
        batching=True,
        max_batch_size=8,
        batcher=DynamicBatcher(
            max_queue_delay_microseconds=5000,
            preferred_batch_size=[1, 2, 4, 8],
        ),
    )

    infer_fns = [infer_fn for _ in range(instances)]

    with Triton(config=config) as triton:
        triton.bind(
            model_name="resnet50",
            infer_func=infer_fns,
            inputs=[Tensor(name="image_b64", dtype=np.bytes_, shape=(1,))],
            outputs=[Tensor(name="logits", dtype=np.float32, shape=(1000,))],
            config=model_config,
        )
        print(f"PyTriton serving on port {http_port} (HTTP) / {grpc_port} (gRPC) "
              f"with {instances} instance(s), DynamicBatcher 5ms, base64 input")
        triton.serve()


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

def run_client(image_path: str, url: str = "localhost:7000"):
    from pytriton.client import ModelClient

    b64_str = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    # (1,) — infer_sample prepends batch dim
    input_data = np.array([b64_str.encode("utf-8")], dtype=np.bytes_)

    with ModelClient(url, "resnet50") as client:
        result = client.infer_sample(image_b64=input_data)
        logits = result["logits"]  # (1000,) — infer_sample removes batch dim

    probs = torch.from_numpy(logits).softmax(dim=0)
    top5 = probs.topk(5)

    _labels = _load_labels()
    print(f"\nTop-5 predictions for: {image_path}\n")
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {_labels[idx]:>30s}  {score:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyTriton ResNet-50 (base64)")
    parser.add_argument("mode", choices=["server", "client"],
                        help="server: 启动服务; client: 运行客户端")
    parser.add_argument("image", nargs="?",
                        help="client 模式下输入图片路径")
    parser.add_argument("--http-port", type=int, default=7000)
    parser.add_argument("--grpc-port", type=int, default=7001)
    parser.add_argument("--instances", type=int, default=1,
                        help="PyTriton 模型实例数 (default: 1)")
    args = parser.parse_args()

    if args.mode == "server":
        _ensure_ld_library_path()
        start_server(args.http_port, args.grpc_port, args.instances)
    elif args.mode == "client":
        if not args.image:
            print("错误: client 模式需要提供图片路径", file=sys.stderr)
            sys.exit(1)
        if not Path(args.image).exists():
            print(f"错误: 图片不存在: {args.image}", file=sys.stderr)
            sys.exit(1)
        run_client(args.image, f"localhost:{args.http_port}")


def _ensure_ld_library_path():
    """确保 PyTriton 的 Triton server 能找到 libpython3.10.so.1.0。"""
    lib = os.path.join(os.path.dirname(sys.executable), "..", "lib")
    lib = os.path.normpath(lib)
    libpython = os.path.join(lib, "libpython3.10.so.1.0")
    if os.path.exists(libpython):
        lp = os.environ.get("LD_LIBRARY_PATH", "")
        if lib not in lp:
            os.environ["LD_LIBRARY_PATH"] = f"{lib}:{lp}" if lp else lib
            os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    main()
