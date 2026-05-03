"""
FastAPI + ResNet-50 推理服务
启动: uvicorn --host 0.0.0.0 --port 8000 --workers <N> fastapi_resnet:app
"""

import base64
import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from resnet_utils import _load_labels, load_model, preprocess, top5_predictions, warmup, get_gpu_memory

model = None
labels = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictRequest(BaseModel):
    image: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, labels
    print(f"Loading ResNet-50 on {device}...")
    model = load_model().to(device)
    labels = _load_labels()
    print(f"Loaded {len(labels)} labels.")
    warmup(model, device, n=3)
    print("FastAPI server ready.")
    yield
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="ResNet-50 Inference", lifespan=lifespan)


def _run_inference(img: Image.Image) -> list:
    input_data = preprocess(img)
    input_tensor = torch.from_numpy(input_data).to(device)
    with torch.no_grad():
        logits = model(input_tensor).cpu().numpy()
    return top5_predictions(logits, labels)


@app.post("/predict_base64")
async def predict_base64(req: PredictRequest):
    try:
        img_bytes = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        return JSONResponse({"error": f"Invalid base64 image: {exc}"}, status_code=400)
    return _run_inference(img)


@app.get("/memory")
async def memory():
    return get_gpu_memory()
