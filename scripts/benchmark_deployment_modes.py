"""Compare one-off PyTorch inference with resident torch and TensorRT services."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyannote_dual.pipeline import MODEL_ID


def parse_audio(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("audio must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid audio input: {value}")
    return label, path


def segment_fingerprint(payload: dict) -> str:
    normalized = {
        mode: payload[mode]["segments"] for mode in ("complete", "non_conflict")
    }
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_payload(output_dir: Path) -> dict:
    return {
        mode: json.loads(
            (output_dir / mode / "segments.json").read_text(encoding="utf-8")
        )
        for mode in ("complete", "non_conflict")
    }


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONHOME": "",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYANNOTE_METRICS_ENABLED": "0",
        }
    )
    ffmpeg_bin = ROOT / "tools" / "ffmpeg-shared" / "bin"
    environment["PATH"] = str(ffmpeg_bin) + os.pathsep + environment.get("PATH", "")
    return environment


def run_one_off(
    audios: list[tuple[str, Path]], output_root: Path, model: str
) -> dict:
    rows = []
    mode_root = output_root / "no_deployment_torch"
    mode_root.mkdir(parents=True, exist_ok=True)
    for index, (label, audio) in enumerate(audios, start=1):
        target = mode_root / f"{index:02d}_{label}"
        command = [
            sys.executable,
            "-m",
            "pyannote_dual.cli",
            "--audio",
            str(audio),
            "--output-dir",
            str(target),
            "--model",
            model,
            "--device",
            "cuda",
            "--backend",
            "torch",
            "--offline",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
        end_to_end = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(
                f"one-off inference failed for {label}:\n{completed.stdout[-4000:]}"
            )
        summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        payload = output_payload(target)
        row = {
            "label": label,
            "audio_seconds": summary["duration_seconds"],
            "input_bytes": audio.stat().st_size,
            "inference_seconds": summary["inference_seconds"],
            "end_to_end_seconds": round(end_to_end, 6),
            "repeated_startup_and_io_seconds": round(
                end_to_end - summary["inference_seconds"], 6
            ),
            "segment_fingerprint": segment_fingerprint(payload),
            "num_speakers": summary["num_speakers"],
        }
        rows.append(row)
        print(json.dumps({"mode": "no_deployment_torch", **row}), flush=True)
    return {
        "mode": "no_deployment_torch",
        "service_startup_seconds": None,
        "rows": rows,
    }


def http_json(base: str, path: str) -> dict:
    with urlopen(base + path, timeout=60) as response:
        return json.loads(response.read())


def run_service(
    backend: str,
    audios: list[tuple[str, Path]],
    output_root: Path,
    model: str,
    engine: str | None,
    poll_interval: float,
) -> dict:
    mode = f"deployed_{backend}"
    mode_root = output_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    max_input = max(path.stat().st_size for _, path in audios)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "pyannote_dual.service",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--data-dir",
        str(mode_root / "service_data"),
        "--model",
        model,
        "--device",
        "cuda",
        "--backend",
        backend,
        "--offline",
        "--max-pending",
        "2",
        "--max-upload-mb",
        str(max(1, (max_input + 2 * 1024**2) // 1024**2)),
    ]
    if engine:
        command.extend(("--engine", engine))
    log_path = mode_root / "service.log"
    log_stream = log_path.open("w", encoding="utf-8")
    startup_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=child_environment(),
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 180
    while True:
        if process.poll() is not None:
            log_stream.flush()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"{mode} service exited during startup:\n{tail}")
        try:
            health = http_json(base, "/health")
            if health.get("status") == "ready" and health.get("backend") == backend:
                break
        except (OSError, URLError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{mode} service startup")
        time.sleep(poll_interval)
    service_startup_seconds = time.perf_counter() - startup_started
    rows = []
    try:
        for label, audio in audios:
            query = urlencode({"filename": audio.name})
            started = time.perf_counter()
            request = Request(
                base + "/jobs?" + query,
                data=audio.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            with urlopen(request, timeout=60) as response:
                accepted = json.loads(response.read())
            deadline = time.monotonic() + 1800
            while True:
                job = http_json(base, accepted["status_url"])
                if job["status"] in {"succeeded", "failed"}:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(label)
                time.sleep(poll_interval)
            if job["status"] != "succeeded":
                raise RuntimeError(f"{mode} failed for {label}: {job.get('error')}")
            result = http_json(base, accepted["result_url"])
            end_to_end = time.perf_counter() - started
            summary = result["summary"]
            row = {
                "label": label,
                "audio_seconds": summary["duration_seconds"],
                "input_bytes": audio.stat().st_size,
                "inference_seconds": summary["inference_seconds"],
                "end_to_end_seconds": round(end_to_end, 6),
                "http_and_output_overhead_seconds": round(
                    end_to_end - summary["inference_seconds"], 6
                ),
                "segment_fingerprint": segment_fingerprint(result),
                "num_speakers": summary["num_speakers"],
            }
            rows.append(row)
            print(json.dumps({"mode": mode, **row}), flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
        log_stream.close()
    return {
        "mode": mode,
        "service_startup_seconds": round(service_startup_seconds, 6),
        "rows": rows,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(result: dict) -> dict:
    rows = result["rows"]
    inference = [row["inference_seconds"] for row in rows]
    end_to_end = [row["end_to_end_seconds"] for row in rows]
    service_startup = result["service_startup_seconds"] or 0.0
    return {
        "mode": result["mode"],
        "files": len(rows),
        "audio_seconds": round(sum(row["audio_seconds"] for row in rows), 6),
        "service_startup_seconds": result["service_startup_seconds"],
        "inference_total_seconds": round(sum(inference), 6),
        "end_to_end_total_seconds": round(sum(end_to_end), 6),
        "batch_total_including_service_start_seconds": round(
            sum(end_to_end) + service_startup, 6
        ),
        "mean_end_to_end_seconds": round(statistics.mean(end_to_end), 6),
        "median_end_to_end_seconds": round(statistics.median(end_to_end), 6),
        "p95_end_to_end_seconds": round(percentile(end_to_end, 0.95), 6),
        "first_request_end_to_end_seconds": end_to_end[0],
        "rtf_inference": round(
            sum(inference) / sum(row["audio_seconds"] for row in rows), 6
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="append", type=parse_audio, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    args = parser.parse_args()
    if len(args.audio) < 2 or args.poll_interval <= 0:
        parser.error("provide at least two audio files and a positive poll interval")
    labels = [label for label, _ in args.audio]
    if len(labels) != len(set(labels)):
        parser.error("audio labels must be unique")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results = []
    results.append(run_one_off(args.audio, output_root, args.model))
    results.append(
        run_service(
            "torch",
            args.audio,
            output_root,
            args.model,
            args.engine,
            args.poll_interval,
        )
    )
    results.append(
        run_service(
            "tensorrt",
            args.audio,
            output_root,
            args.model,
            args.engine,
            args.poll_interval,
        )
    )

    summaries = [summarize(result) for result in results]
    baseline = summaries[0]["batch_total_including_service_start_seconds"]
    for summary in summaries:
        summary["speedup_vs_no_deployment"] = round(
            baseline / summary["batch_total_including_service_start_seconds"], 6
        )
    reference = {
        row["label"]: row["segment_fingerprint"] for row in results[0]["rows"]
    }
    output_match = {
        result["mode"]: all(
            row["segment_fingerprint"] == reference[row["label"]]
            for row in result["rows"]
        )
        for result in results
    }
    import torch

    report = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "methodology": {
            "no_deployment_torch": "A fresh Python process and model load for every audio file.",
            "deployed_torch": "One resident HTTP service and one torch model load reused by all requests.",
            "deployed_tensorrt": "One resident HTTP service and one TensorRT-enabled model load reused by all requests.",
            "warmup": "No unreported warmup request; the first measured request includes first-use overhead.",
            "end_to_end": "Client file read, localhost upload, queue/poll, inference, output writing, and response retrieval.",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "inputs": [
            {"label": label, "bytes": path.stat().st_size} for label, path in args.audio
        ],
        "summaries": summaries,
        "outputs_match_no_deployment": output_match,
        "results": results,
    }
    destination = output_root / "benchmark.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summaries": summaries, "outputs_match": output_match}, indent=2))
    print(f"Benchmark written to {destination}")
    if not all(output_match.values()):
        raise SystemExit("Output mismatch detected between benchmark modes")


if __name__ == "__main__":
    main()
