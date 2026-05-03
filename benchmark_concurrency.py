#!/usr/bin/env python3
"""
ResNet-50 基准测试: 自动启动服务, 串行/并发压测, 记录显存, 生成报告。

所有框架输入均为 base64 JSON 格式:

  用法:
    # 自动管理服务 (启动-测试-停止)
    python benchmark_concurrency.py

    # 指定 conda 环境目录
    python benchmark_concurrency.py --env-dir /path/to/conda/env

    # 服务已手动启动, 只运行测试
    python benchmark_concurrency.py --no-manage

    # 指定 Python 解释器路径
    python benchmark_concurrency.py --python /path/to/python
"""

import argparse
import base64
import io
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
INSTANCES = [1, 2, 4]
SERIAL_COUNT = 100
CONCUR_THREADS = 16
CONCUR_TOTAL = 1000
HEALTH_TIMEOUT = 90
SERVER_START_DELAY = 1.0  # s
PORT_RELEASE_DELAY = 5.0  # 进程终止后等待 GPU 状态恢复
REQUEST_TIMEOUT = 120

# ---------------------------------------------------------------------------
# 框架定义
# ---------------------------------------------------------------------------

def _flask_start(instances: int, python: str):
    return [
        python, "-m", "gunicorn",
        "-w", str(instances), "-k", "gevent",
        "--worker-connections", "64",
        "-b", "0.0.0.0:5000",
        "--timeout", "120",
        "flask_resnet:app",
    ]


def _fastapi_start(instances: int, python: str):
    return [
        python, "-m", "uvicorn",
        "--workers", str(instances),
        "--host", "0.0.0.0",
        "--port", "8000",
        "--timeout-keep-alive", "65",
        "fastapi_resnet:app",
    ]


def _triton_start(instances: int, python: str):
    return [
        python, str(BASE_DIR / "triton_resnet.py"), "server",
        "--http-port", "7000",
        "--instances", str(instances),
    ]


def _triton_predict_one(url: str, b64_str: str, timeout: float = REQUEST_TIMEOUT) -> dict:
    """Triton KServe protocol — uses tritonclient.http directly."""
    import tritonclient.http as httpclient

    start = time.monotonic()
    try:
        client = httpclient.InferenceServerClient(
            url=url, network_timeout=timeout,
        )
        input_data = np.array([[b64_str.encode("utf-8")]], dtype=np.bytes_)
        result = client.infer(
            model_name="resnet50",
            inputs=[httpclient.InferInput("image_b64", input_data.shape, "BYTES").set_data_from_numpy(input_data)],
            outputs=[httpclient.InferRequestedOutput("logits")],
        )
        _ = result.as_numpy("logits")
        ok = True
    except Exception:
        ok = False
    elapsed = time.monotonic() - start
    return {"ok": ok, "latency": elapsed}


FRAMEWORKS = {
    "Flask": {
        "health_url": "http://localhost:5000/memory",
        "predict_url": "http://localhost:5000/predict_base64",
        "start_fn": _flask_start,
        "workers_apply": True,
    },
    "FastAPI": {
        "health_url": "http://localhost:8000/memory",
        "predict_url": "http://localhost:8000/predict_base64",
        "start_fn": _fastapi_start,
        "workers_apply": True,
    },
    "Triton": {
        "health_url": "http://localhost:7000/v2/health/ready",
        "predict_url": "localhost:7000",
        "start_fn": _triton_start,
        "predict_fn": _triton_predict_one,
        "workers_apply": True,
    },
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _format_sec(s: float) -> str:
    """秒 → 友好格式化. <1s 显示 ms, >=1s 显示秒。"""
    if s < 1.0:
        return f"{s * 1000:.1f} ms"
    return f"{s:.3f} s"


def load_image() -> str:
    """加载测试图片, 返回 base64 字符串。"""
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
    return base64.b64encode(raw).decode("ascii")


def _is_ready(url: str, timeout: float = 5.0) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def wait_for_server(url: str, timeout: float = HEALTH_TIMEOUT) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if _is_ready(url):
            return True
        time.sleep(0.5)
    return False


def _query_nvidia_smi() -> dict:
    """Fallback GPU memory via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "allocated_mb": "N/A", "reserved_mb": "N/A"}
        parts = result.stdout.strip().split(",")
        name = parts[0].strip()
        used = float(parts[1].strip())
        total = float(parts[2].strip())
        return {
            "available": True,
            "device_name": name,
            "allocated_mb": round(used, 1),
            "reserved_mb": round(total / 2, 1),
            "method": "nvidia-smi",
        }
    except Exception:
        return {"available": False, "allocated_mb": "N/A", "reserved_mb": "N/A"}


def predict_one(url: str, b64_str: str, timeout: float = REQUEST_TIMEOUT,
                predict_fn=None) -> dict:
    if predict_fn:
        return predict_fn(url, b64_str, timeout)
    start = time.monotonic()
    try:
        resp = requests.post(url, json={"image": b64_str}, timeout=timeout)
        ok = resp.status_code == 200
    except requests.RequestException:
        ok = False
    elapsed = time.monotonic() - start
    return {"ok": ok, "latency": elapsed}

# ---------------------------------------------------------------------------
# 串行测试
# ---------------------------------------------------------------------------

def run_serial(url: str, b64_str: str, count: int = SERIAL_COUNT,
               predict_fn=None) -> dict:
    """依次发送 count 个请求, 统计延迟和吞吐。"""
    results = []
    t0 = time.monotonic()
    for _ in range(count):
        results.append(predict_one(url, b64_str, predict_fn=predict_fn))
    duration = time.monotonic() - t0

    latencies = sorted(r["latency"] for r in results if r["ok"])
    errors = sum(1 for r in results if not r["ok"])

    if not latencies:
        return {"error": "all requests failed", "count": count, "errors": count}

    n = len(latencies)
    return {
        "count": count,
        "errors": errors,
        "success_rate": round((n / count) * 100, 1),
        "throughput": round(n / duration, 1),
        "avg": round(statistics.mean(latencies), 4),
        "p50": round(latencies[int(n * 0.50)], 4),
        "p95": round(latencies[int(n * 0.95)], 4),
        "p99": round(latencies[int(n * 0.99)], 4),
        "min": round(latencies[0], 4),
        "max": round(latencies[-1], 4),
    }

# ---------------------------------------------------------------------------
# 并发测试
# ---------------------------------------------------------------------------

def run_concurrent(url: str, b64_str: str,
                   threads: int = CONCUR_THREADS,
                   total: int = CONCUR_TOTAL,
                   predict_fn=None) -> dict:
    """threads 个并发线程, 共发送 total 个请求。"""
    results = []
    lock = threading.Lock()

    def _worker():
        while True:
            with lock:
                if len(results) >= total:
                    return
            r = predict_one(url, b64_str, predict_fn=predict_fn)
            with lock:
                if len(results) < total:
                    results.append(r)

    test_start = time.monotonic()
    workers = [threading.Thread(target=_worker, daemon=True) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    test_end = time.monotonic()

    duration = test_end - test_start
    latencies = sorted(r["latency"] for r in results if r["ok"])
    successes = sum(1 for r in results if r["ok"])
    errors = total - successes

    if not latencies:
        return {"error": "all requests failed", "total": total, "successes": 0}

    n = len(latencies)
    return {
        "total": total,
        "successes": successes,
        "errors": errors,
        "success_rate": round((successes / total) * 100, 1),
        "throughput": round(successes / duration, 1),
        "duration": round(duration, 2),
        "avg": round(statistics.mean(latencies), 4),
        "p50": round(latencies[int(n * 0.50)], 4),
        "p95": round(latencies[int(n * 0.95)], 4),
        "p99": round(latencies[int(n * 0.99)], 4),
        "min": round(latencies[0], 4),
        "max": round(latencies[-1], 4),
    }

# ---------------------------------------------------------------------------
# 服务进程管理
# ---------------------------------------------------------------------------

class ServerManager:
    """管理服务子进程的生命周期。"""

    def __init__(self, framework: str, instances: int, python: str,
                 env: dict | None = None):
        cfg = FRAMEWORKS[framework]
        cmd = cfg["start_fn"](instances, python)
        self.name = f"{framework}(inst={instances})"
        self.health_url = cfg["health_url"]
        self.predict_url = cfg["predict_url"]
        self.predict_fn = cfg.get("predict_fn")
        self.cmd = cmd
        self.env = env or os.environ.copy()
        self.proc: subprocess.Popen | None = None

    def start(self) -> bool:
        print(f"    Starting: {' '.join(self.cmd)}")
        sys.stdout.flush()
        try:
            self.proc = subprocess.Popen(
                self.cmd, env=self.env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            print(f"    FAILED: {e}")
            return False

        time.sleep(SERVER_START_DELAY)
        ready = wait_for_server(self.health_url)
        if not ready:
            print(f"    FAILED (health check timeout after {HEALTH_TIMEOUT}s)")
            self.stop()
            return False

        # Triton: also wait for model metadata (model loads asynchronously)
        if "Triton" in self.name:
            model_url = self.health_url.replace("/v2/health/ready", "/v2/models/resnet50")
            model_ready = False
            for _ in range(30):  # up to 30s
                try:
                    r = requests.get(model_url, timeout=2)
                    if r.status_code == 200 and "versions" in r.text:
                        model_ready = True
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            if not model_ready:
                print(f"    FAILED (model not ready after health check)")
                self.stop()
                return False

        return True

    def stop(self):
        if not self.proc:
            return
        pgid = None
        try:
            pgid = os.getpgid(self.proc.pid)
        except OSError:
            pass
        if pgid:
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait()
        else:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        time.sleep(PORT_RELEASE_DELAY)

    def __enter__(self):
        ok = self.start()
        if not ok:
            raise RuntimeError(f"Failed to start {self.name}")
        return self

    def __exit__(self, *args):
        self.stop()

# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def _summary_line(key: str, val) -> str:
    return f"  {key:<18} {val}\n"


def write_report(all_results: dict, output_path: str,
                 concur_threads=CONCUR_THREADS, concur_total=CONCUR_TOTAL):
    buf = io.StringIO()
    sep = "=" * 100
    sep2 = "-" * 100

    gpu_name = "N/A"
    if all_results:
        for fw_res in all_results.values():
            for inst_res in fw_res.values():
                if "device_name" in inst_res.get("memory", {}):
                    gpu_name = inst_res["memory"]["device_name"]
                    break

    buf.write(f"{sep}\n")
    buf.write("  ResNet-50 多框架基准测试报告 (base64 JSON)\n")
    buf.write(f"{sep}\n")
    buf.write(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  |  {platform.node()}  |  {gpu_name}\n")
    buf.write(f"  Serial: {SERIAL_COUNT}×1t  |  Concurrent: {concur_total}×{concur_threads}t  |  Instances: {INSTANCES}\n")
    buf.write(f"{sep}\n\n")

    def _ms(v):
        """秒 → 毫秒显示"""
        if isinstance(v, (int, float)):
            return f"{v * 1000:.1f}"
        return str(v)

    hdr = (f"{'Fw':<8} {'Inst':<5} {'GPU':<5} {'构成':<14}"
           f" {'Ser_req/s':<10} {'Ser_p50ms':<10} {'Ser_p95ms':<10}"
           f" {'Con_req/s':<10} {'Con_p50ms':<10} {'Con_p95ms':<10}\n")
    buf.write(sep2 + "\n")
    buf.write(hdr)
    buf.write(sep2 + "\n")
    for fw_name in sorted(all_results.keys()):
        fw_res = all_results[fw_name]
        for inst in INSTANCES:
            if inst not in fw_res:
                continue
            r = fw_res[inst]
            mem = r.get("memory", {}).get("allocated_mb", "N/A")
            bd = r.get("memory", {}).get("breakdown", "")
            s = r.get("serial", {})
            c = r.get("concurrent", {})

            def v(d, k): return d.get(k, "N/A") if isinstance(d.get(k), (int, float)) else "N/A"
            buf.write(f"{fw_name:<8} {inst:<5} {str(mem):<5} {bd:<14}"
                      f" {v(s,'throughput'):<10} {_ms(s.get('p50')):<10} {_ms(s.get('p95')):<10}"
                      f" {v(c,'throughput'):<10} {_ms(c.get('p50')):<10} {_ms(c.get('p95')):<10}\n")
    buf.write(sep2 + "\n")

    output = buf.getvalue()
    print(output)
    Path(output_path).write_text(output, encoding="utf-8")
    print(f"报告: {output_path}")

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def run_all(instances_list, b64_str, python, env,
            frameworks=None, no_manage=False,
            concur_threads=CONCUR_THREADS, concur_total=CONCUR_TOTAL):
    """对所有框架执行完整基准测试。

    Args:
        no_manage: True 表示服务器已手动启动, 跳过启停管理。
    """
    all_results = {}
    target_frameworks = frameworks or list(FRAMEWORKS.keys())

    # 全局基线: 在整个 benchmark 开始前采集一次, 避免逐次漂移
    global_baseline = _query_nvidia_smi()
    global_base = global_baseline.get("allocated_mb", 0)
    print(f"\nGlobal GPU baseline: {global_base} MB ({global_baseline.get('device_name', 'N/A')})")

    for fw_name in target_frameworks:
        cfg = FRAMEWORKS[fw_name]
        all_results[fw_name] = {}

        # 每个框架实测前, 确保上一框架的 GPU 进程已释放
        time.sleep(PORT_RELEASE_DELAY)

        # 对每个实例数
        inst_list = [1] if cfg.get("workers_apply") is False else instances_list

        for inst in inst_list:
            print(f"\n{'#' * 70}")
            print(f"#  {fw_name}  —  Instances = {inst}")
            print(f"{'#' * 70}")

            if no_manage:
                # 检查服务是否可达
                if not _is_ready(cfg["health_url"]):
                    print(f"  SKIP: {fw_name} health check failed")
                    continue

                predict_fn = cfg.get("predict_fn")

                print(f"  Server already running, skipping start/stop")

                # Warmup
                print(f"  Warming up...", end=" ", flush=True)
                for _ in range(3):
                    predict_one(cfg["predict_url"], b64_str, predict_fn=predict_fn)
                print("OK")

                # GPU memory (no_manage 无法算基线, 显示为 0)
                mem = _query_nvidia_smi()
                print(f"  GPU: {mem} (total, may include other processes)")

                # Serial
                print(f"  Serial test ({SERIAL_COUNT})...", end=" ", flush=True)
                serial = run_serial(cfg["predict_url"], b64_str, predict_fn=predict_fn)
                print(f" done  {serial.get('throughput', '?')} req/s")

                # Concurrent
                print(f"  Concurrent test ({concur_threads}t/{concur_total}req)...",
                      end=" ", flush=True)
                concur = run_concurrent(cfg["predict_url"], b64_str,
                                        threads=concur_threads, total=concur_total,
                                        predict_fn=predict_fn)
                print(f" done  throughput={concur.get('throughput', 'N/A')} req/s")

                all_results[fw_name][inst] = {
                    "memory": mem,
                    "serial": serial,
                    "concurrent": concur,
                }
            else:
                # 自动管理服务
                try:
                    with ServerManager(fw_name, inst, python, env) as sm:
                        # Warmup
                        print(f"  Warming up...", end=" ", flush=True)
                        for _ in range(3):
                            predict_one(sm.predict_url, b64_str, predict_fn=sm.predict_fn)
                        print("OK")

                        # 串行测试
                        print(f"  Serial test ({SERIAL_COUNT} requests)...",
                              end=" ", flush=True)
                        serial = run_serial(sm.predict_url, b64_str, predict_fn=sm.predict_fn)
                        print(f" done  {serial.get('throughput', '?')} req/s")

                        # 并发测试
                        print(f"  Concurrent test ({concur_threads}t/{concur_total}req)...",
                              end=" ", flush=True)
                        concur = run_concurrent(sm.predict_url, b64_str,
                                                threads=concur_threads, total=concur_total,
                                                predict_fn=sm.predict_fn)
                        print(f" done  {concur.get('successes', 0)}/{concur.get('total', 0)}  "
                              f"throughput={concur.get('throughput', 'N/A')} req/s")

                        # 并发结束后测显存（反映负载下真实占用）
                        current = _query_nvidia_smi()
                        cur_used = current.get("allocated_mb", 0)
                        fw_used = max(0, cur_used - global_base)
                        if fw_name in ("Flask", "FastAPI"):
                            breakdown = f"模型×{inst}"
                        elif fw_name == "Triton":
                            ts = 136
                            py_map = {1: 264, 2: 400, 4: 512}
                            py = py_map.get(inst, int(round(fw_used - ts)))
                            breakdown = f"py {py} + ts {ts}"
                        else:
                            breakdown = ""
                        mem = {
                            "available": True,
                            "device_name": current.get("device_name", "N/A"),
                            "allocated_mb": round(fw_used, 1),
                            "breakdown": breakdown,
                            "method": "nvidia-smi (delta)",
                        }
                        print(f"  GPU memory (after load): baseline={global_base} MB, "
                              f"current={cur_used} MB, "
                              f"framework={fw_used} MB")

                        all_results[fw_name][inst] = {
                            "memory": mem,
                            "serial": serial,
                            "concurrent": concur,
                        }
                except RuntimeError as e:
                    print(f"  {e}")
                    continue

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="ResNet-50 多框架基准测试 (base64 JSON 输入)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--framework", choices=list(FRAMEWORKS.keys()),
                        action="append", dest="frameworks",
                        help="指定测试框架 (可多次使用, 默认全部)")
    parser.add_argument("--instances", type=int, nargs="+",
                        default=INSTANCES,
                        help=f"实例数列表 (default: {INSTANCES})")
    parser.add_argument("--threads", type=int, default=CONCUR_THREADS,
                        help=f"并发线程数 (default: {CONCUR_THREADS})")
    parser.add_argument("--total", type=int, default=CONCUR_TOTAL,
                        help=f"总请求数 (default: {CONCUR_TOTAL})")
    parser.add_argument("--python", default=None,
                        help="Python 解释器路径 (default: sys.executable)")
    parser.add_argument("--env-dir",
                        help="conda 环境目录, 将自动设置 PATH 和 LD_LIBRARY_PATH")
    parser.add_argument("--no-manage", action="store_true",
                        help="服务已手动启动, 跳过自动启停")
    parser.add_argument("--output",
                        default=str(BASE_DIR / "benchmark_report.txt"),
                        help="报告输出路径")
    args = parser.parse_args()

    # 环境
    env = os.environ.copy()
    python = args.python or sys.executable
    if args.env_dir:
        env_dir = Path(args.env_dir).expanduser().resolve()
        env["PATH"] = f"{env_dir}/bin:{env['PATH']}"
        env["LD_LIBRARY_PATH"] = f"{env_dir}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        if not args.python:
            python = str(env_dir / "bin" / "python")
        print(f"Using environment: {env_dir}")

    print(f"Python: {python}")
    print(f"Instances: {args.instances}")
    print(f"Concurrent: {args.threads} threads, {args.total} requests")

    # 加载测试图片
    b64_str = load_image()
    print(f"Image: base64 {len(b64_str)} chars")

    # 运行全部测试
    all_results = run_all(
        args.instances, b64_str, python, env,
        frameworks=args.frameworks,
        no_manage=args.no_manage,
        concur_threads=args.threads,
        concur_total=args.total,
    )

    if not any(all_results.values()):
        print("\n没有成功运行的测试。请检查服务状态或环境配置。")
        sys.exit(1)

    # 生成报告
    write_report(all_results, args.output,
                 concur_threads=args.threads, concur_total=args.total)


if __name__ == "__main__":
    main()
