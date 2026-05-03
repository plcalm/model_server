"""
ResNet-50 并发压测: Flask vs FastAPI vs PyTriton (Triton Server)

支持两种请求模式:
  multipart (默认): POST multipart/form-data (file upload)
  json:             POST application/json  (base64 编码图片)

用法:
  bash run_flask.sh        # port 5000
  bash run_fastapi.sh      # port 8000
  bash run_triton.sh       # port 7000

  # 默认 multipart 模式
  python benchmark_concurrency.py

  # base64 JSON 模式
  python benchmark_concurrency.py --payload-type json
"""

import argparse
import base64
import io
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

BASE_DIR = Path(__file__).parent

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
WARMUP_SECS = 3
MEASURE_SECS = 15
REQUEST_TIMEOUT = 60

# ---------------------------------------------------------------------------
# 负载生成
# ---------------------------------------------------------------------------

def _load_image():
    """返回 (raw_bytes, base64_str, preprocessed_numpy)。"""
    candidates = [
        BASE_DIR / "cat.jpg",
        BASE_DIR / "cat.png",
    ]
    for path in candidates:
        if path.exists():
            raw = path.read_bytes()
            break
    else:
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

    b64 = base64.b64encode(raw).decode("ascii")

    from resnet_utils import preprocess
    pil_img = Image.open(io.BytesIO(raw))
    arr = preprocess(pil_img)

    return raw, b64, arr


# ---------------------------------------------------------------------------
# Worker 函数
# ---------------------------------------------------------------------------

def _http_multipart_worker(url: str, image_bytes: bytes,
                           results: list, stop_event):
    """multipart/form-data 上传。"""
    session = requests.Session()
    while not stop_event.is_set():
        start = time.monotonic()
        try:
            resp = session.post(url, files={"image": image_bytes},
                                timeout=REQUEST_TIMEOUT)
            ok = resp.status_code == 200
        except requests.RequestException:
            ok = False
        elapsed = time.monotonic() - start
        results.append({"ok": ok, "latency": elapsed})


def _http_json_worker(url: str, base64_str: str,
                      results: list, stop_event):
    """application/json base64 上传。"""
    session = requests.Session()
    while not stop_event.is_set():
        start = time.monotonic()
        try:
            resp = session.post(url, json={"image": base64_str},
                                timeout=REQUEST_TIMEOUT)
            ok = resp.status_code == 200
        except requests.RequestException:
            ok = False
        elapsed = time.monotonic() - start
        results.append({"ok": ok, "latency": elapsed})


def _triton_worker(url: str, model_name: str, input_array: np.ndarray,
                   results: list, stop_event):
    """tritonclient.http (始终使用二进制协议, 与 payload_type 无关)。"""
    import tritonclient.http as httpclient
    client = httpclient.InferenceServerClient(url=url)
    while not stop_event.is_set():
        start = time.monotonic()
        try:
            inp = httpclient.InferInput("image", list(input_array.shape), "FP32")
            inp.set_data_from_numpy(input_array)
            result = client.infer(model_name, [inp])
            _ = result.as_numpy("logits")
            ok = True
        except Exception:
            ok = False
        elapsed = time.monotonic() - start
        results.append({"ok": ok, "latency": elapsed})


def _run_workers(worker_fn, args, concurrency, duration):
    stop_event = threading.Event()
    results = []
    threads = []
    for _ in range(concurrency):
        t = threading.Thread(target=worker_fn, args=(*args, results, stop_event),
                             daemon=True)
        threads.append(t)
        t.start()
    time.sleep(duration)
    stop_event.set()
    for t in threads:
        t.join()
    return results


def benchmark(name, worker_fn, worker_args, concurrency, duration):
    results = _run_workers(worker_fn, worker_args, concurrency, duration)
    if not results:
        return {"concurrency": concurrency, "total": 0, "throughput": 0.0,
                "success_rate": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    latencies = sorted(r["latency"] for r in results)
    successes = sum(1 for r in results if r["ok"])
    total = len(results)
    n = total
    return {
        "concurrency": concurrency,
        "total": total,
        "throughput": round(total / duration, 1),
        "success_rate": round(successes / total * 100, 1),
        "avg": round(statistics.mean(latencies), 3),
        "p50": round(latencies[int(n * 0.50)], 3),
        "p95": round(latencies[int(n * 0.95)], 3),
        "p99": round(latencies[int(n * 0.99)], 3),
    }


# ---------------------------------------------------------------------------
# 框架适配器
# ---------------------------------------------------------------------------

class FrameworkAdapter:
    def __init__(self, name, worker_fn, worker_args, check_fn, check_args):
        self.name = name
        self.worker_fn = worker_fn
        self.worker_args = worker_args
        self.check_fn = check_fn
        self.check_args = check_args

    def check_reachable(self):
        try:
            self.check_fn(*self.check_args)
            return True
        except Exception as e:
            print(f"\n  {self.name} 不可达: {e}")
            return False

    def run(self, concurrency, duration):
        return benchmark(self.name, self.worker_fn, self.worker_args,
                         concurrency, duration)


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def run_benchmark(name, adapter):
    print(f"\n===== {name} =====")
    rows = []
    for level in CONCURRENCY_LEVELS:
        print(f"  Concurrency={level:2d}  (warmup {WARMUP_SECS}s + measure {MEASURE_SECS}s) ...",
              end=" ", flush=True)
        adapter.run(level, WARMUP_SECS)
        stats = adapter.run(level, MEASURE_SECS)
        rows.append(stats)
        print(f"done  {stats['total']} requests, {stats['throughput']} req/s")
    return rows


def write_report(all_rows, payload_type, output_path):
    import torch
    buf = io.StringIO()
    sep_full = "=" * 80
    sep_line = "-" * 100

    payload_label = "multipart/form-data" if payload_type == "multipart" else "application/json (base64)"
    buf.write(f"{sep_full}\n")
    buf.write(f"  ResNet-50 并发压测报告  |  Payload: {payload_label}\n")
    buf.write(f"{sep_full}\n")
    buf.write(f"  Date:       {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    buf.write(f"  GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")
    buf.write(f"  Duration:   {MEASURE_SECS}s per concurrency level\n")
    buf.write(f"{sep_full}\n\n")

    buf.write(f"{sep_line}\n")
    buf.write(f"{'Framework':<14} {'Concurrency':<12} {'Requests':<10} {'Throughput':<12} "
              f"{'Success%':<10} {'Avg(s)':<8} {'P50(s)':<8} {'P95(s)':<8} {'P99(s)':<8}\n")
    buf.write(f"{sep_line}\n")
    for name, rows in all_rows.items():
        for r in rows:
            buf.write(f"{name:<14} {r['concurrency']:<12} {r['total']:<10} {r['throughput']:<12} "
                      f"{r['success_rate']:<10} {r['avg']:<8} {r['p50']:<8} {r['p95']:<8} {r['p99']:<8}\n")
        buf.write(f"{sep_line}\n")

    output = buf.getvalue()
    print(output)
    Path(output_path).write_text(output)
    print(f"报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ResNet-50 并发压测")
    parser.add_argument("--flask-url", default="http://localhost:5000")
    parser.add_argument("--fastapi-url", default="http://localhost:8000")
    parser.add_argument("--triton-url", default="localhost:7000")
    parser.add_argument("--triton-model", default="resnet50")
    parser.add_argument("--payload-type", choices=["multipart", "json"], default="multipart",
                        help="请求格式: multipart (文件上传) 或 json (base64)")
    parser.add_argument("--output", default=str(BASE_DIR / "benchmark_report.txt"))
    args = parser.parse_args()

    raw_image, b64_str, proc_array = _load_image()
    print(f"Image: {len(raw_image)} raw bytes, base64={len(b64_str)} chars, "
          f"preprocessed={proc_array.shape}")

    is_json = args.payload_type == "json"
    flask_url = f"{args.flask_url}/predict_base64" if is_json else f"{args.flask_url}/predict"
    fastapi_url = f"{args.fastapi_url}/predict_base64" if is_json else f"{args.fastapi_url}/predict"

    def _check_multipart(url):
        resp = requests.post(url, files={"image": raw_image}, timeout=10)
        resp.raise_for_status()

    def _check_json(url):
        resp = requests.post(url, json={"image": b64_str}, timeout=10)
        resp.raise_for_status()

    def _check_triton(url):
        resp = requests.get(f"http://{url}/v2/health/ready", timeout=10)
        resp.raise_for_status()

    check_fn = _check_json if is_json else _check_multipart

    if is_json:
        http_worker = _http_json_worker
        http_args = (b64_str,)
    else:
        http_worker = _http_multipart_worker
        http_args = (raw_image,)

    adapters = [
        FrameworkAdapter("Flask",     http_worker,      (flask_url, *http_args),
                         check_fn, (flask_url,)),
        FrameworkAdapter("FastAPI",   http_worker,      (fastapi_url, *http_args),
                         check_fn, (fastapi_url,)),
        FrameworkAdapter("PyTriton",  _triton_worker,   (args.triton_url, args.triton_model, proc_array),
                         _check_triton, (args.triton_url,)),
    ]

    all_rows = {}
    for adapter in adapters:
        if adapter.check_reachable():
            rows = run_benchmark(adapter.name, adapter)
            all_rows[adapter.name] = rows

    if not all_rows:
        print("没有可用的服务, 请先启动。")
        sys.exit(1)

    write_report(all_rows, args.payload_type, args.output)


if __name__ == "__main__":
    main()
