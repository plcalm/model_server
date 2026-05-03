"""
PyTriton ResNet-50 部署示例

依赖安装:
  pip install pytriton torch torchvision pillow

启动服务:
  python triton_resnet.py server

客户端测试:
  python triton_resnet.py client <image_path>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------

def _image_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess(image: Image.Image) -> np.ndarray:
    """将 PIL Image 转为 (1, 3, 224, 224) float32 numpy 数组。"""
    tensor = _image_transform()(image.convert("RGB"))
    return tensor.unsqueeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# ImageNet 标签加载
# ---------------------------------------------------------------------------

def _load_labels() -> list[str]:
    """从 torchvision 内置权重读取 ImageNet 标签。"""
    weights = ResNet50_Weights.IMAGENET1K_V1
    return weights.meta["categories"]


# ---------------------------------------------------------------------------
# 服务端
# ---------------------------------------------------------------------------

def _infer_fn(batch: list[np.ndarray]) -> list[np.ndarray]:
    """PyTriton 推理回调: batch 为 list[np.ndarray], 返回 list[np.ndarray]。"""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _infer_fn.model.to(device)

    inputs = torch.from_numpy(batch[0]).to(device)
    with torch.no_grad():
        outputs = model(inputs)

    # shape: (batch_size, 1000)
    return [outputs.cpu().numpy().astype(np.float32)]


# 将模型挂在函数上, 避免全局变量与闭包序列化问题
_infer_fn.model = None


def _load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def start_server():
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    print("Loading ResNet-50...")
    model = _load_model()
    _infer_fn.model = model

    labels = _load_labels()
    print(f"Loaded {len(labels)} ImageNet labels")

    # @batch 自动将队列中的请求合并为 batch 传给 infer_fn
    @batch
    def infer_fn(**inputs: np.ndarray) -> dict[str, np.ndarray]:
        logits = _infer_fn([inputs["image"]])
        return {"logits": logits[0]}

    config = TritonConfig(
        http_address="0.0.0.0",
        http_port=7000,
        http_restricted_api="none",
        grpc_address="0.0.0.0",
        grpc_port=7001,
    )

    model_config = ModelConfig(
        batching=True,
        max_batch_size=8,
        batcher=DynamicBatcher(
            max_queue_delay_microseconds=5000,  # 最多等 5ms 攒 batch
            preferred_batch_size=[4, 8],        # 优先凑 4 或 8 的 batch
        ),
    )

    with Triton(config=config) as triton:
        triton.bind(
            model_name="resnet50",
            infer_func=infer_fn,
            # max_batch_size>0 时, Triton 自动管理 batch 维度, 配置中不包含 batch 维
            inputs=[Tensor(name="image", dtype=np.float32, shape=(3, 224, 224))],
            outputs=[Tensor(name="logits", dtype=np.float32, shape=(1000,))],
            config=model_config,
        )
        print("Serving ResNet-50 (batching enabled) on port 7000")
        triton.serve()


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

def run_client(image_path: str):
    from pytriton.client import ModelClient

    image = Image.open(image_path)
    input_data = preprocess(image)  # (1, 3, 224, 224)

    with ModelClient("localhost:7000", "resnet50") as client:
        result = client.infer_sample(image=input_data)
        logits = result["logits"]  # (1, 1000)

    # top-5 预测
    probs = torch.from_numpy(logits).softmax(dim=1).squeeze(0)
    top5 = probs.topk(5)
    labels = _load_labels()

    print(f"\nTop-5 predictions for: {image_path}\n")
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {labels[idx]:>30s}  {score:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyTriton ResNet-50 示例")
    parser.add_argument("mode", choices=["server", "client"],
                        help="运行模式: server 启动 Triton 服务, client 运行推理客户端")
    parser.add_argument("image", nargs="?",
                        help="client 模式下输入图片路径")
    args = parser.parse_args()

    if args.mode == "server":
        start_server()
    elif args.mode == "client":
        if not args.image:
            print("错误: client 模式需要提供图片路径", file=sys.stderr)
            sys.exit(1)
        if not Path(args.image).exists():
            print(f"错误: 图片不存在: {args.image}", file=sys.stderr)
            sys.exit(1)
        run_client(args.image)


if __name__ == "__main__":
    main()
