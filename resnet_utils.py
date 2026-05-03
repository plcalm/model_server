"""
ResNet-50 共享工具模块: 预处理、模型加载、top-5 预测
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _image_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL Image → (1, 3, 224, 224) float32 numpy array."""
    tensor = _image_transform()(image.convert("RGB"))
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def load_model() -> torch.nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def _load_labels() -> list[str]:
    weights = ResNet50_Weights.IMAGENET1K_V1
    return weights.meta["categories"]


def top5_predictions(logits: np.ndarray, labels: list[str]) -> list[dict]:
    """(1, 1000) logits → top-5 [{label, confidence, index}, ...]"""
    probs = torch.from_numpy(logits).softmax(dim=1).squeeze(0)
    top5 = probs.topk(5)
    return [
        {"label": labels[idx], "confidence": round(score, 4), "index": idx}
        for score, idx in zip(top5.values.tolist(), top5.indices.tolist())
    ]
