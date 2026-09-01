#!/usr/bin/env python3
# Jetson 性能数据分析。
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import signal
import threading
import time


GPU_FREQ_PATH = Path("/sys/class/devfreq/17000000.gpu/cur_freq")
GPU_LOAD_PATH = Path("/sys/devices/platform/bus@0/17000000.gpu/load")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Record timestamped Jetson GPU, EMC, thermal, and power telemetry.")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--full-interval-ms", type=int, default=200)
  parser.add_argument("--fast-interval-ms", type=int, default=50)
  args = parser.parse_args()
  if args.full_interval_ms < 50:
    parser.error("--full-interval-ms must be at least 50")
  if args.fast_interval_ms < 10:
    parser.error("--fast-interval-ms must be at least 10")
  return args


def read_int(path: Path) -> int | None:
  try:
    return int(path.read_text(encoding="utf-8").strip())
  except (OSError, ValueError):
    return None


def local_iso_timestamp(timestamp_unix_ns: int) -> str:
  return datetime.fromtimestamp(timestamp_unix_ns / 1_000_000_000.0).astimezone().isoformat(
      timespec="microseconds"
  )


def fast_sample_loop(output_path: Path, interval_seconds: float, stop_event: threading.Event) -> None:
  with output_path.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.writer(output_file)
    writer.writerow([
        "timestamp_unix_ns",
        "timestamp_iso",
        "gpu_freq_hz",
        "gpu_load_raw",
        "gpu_load_pct",
    ])
    deadline = time.monotonic()
    while not stop_event.is_set():
      timestamp_unix_ns = time.time_ns()
      gpu_freq_hz = read_int(GPU_FREQ_PATH)
      gpu_load_raw = read_int(GPU_LOAD_PATH)
      gpu_load_pct = None if gpu_load_raw is None else gpu_load_raw / 10.0
      writer.writerow([
          timestamp_unix_ns,
          local_iso_timestamp(timestamp_unix_ns),
          gpu_freq_hz,
          gpu_load_raw,
          gpu_load_pct,
      ])
      output_file.flush()
      deadline += interval_seconds
      stop_event.wait(max(0.0, deadline - time.monotonic()))


def main() -> int:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  stop_event = threading.Event()
  jetson_holder: list[object] = []

  def request_stop(_signum=None, _frame=None) -> None:
    stop_event.set()
    if jetson_holder:
      try:
        jetson_holder[0].close()
      except Exception:
        pass

  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)

  fast_thread = threading.Thread(
      target=fast_sample_loop,
      args=(args.output_dir / "gpu_fast.csv", args.fast_interval_ms / 1000.0, stop_event),
      name="gpu-fast-telemetry",
      daemon=True,
  )
  fast_thread.start()

  try:
    from jtop import jtop
  except ImportError as exc:
    request_stop()
    fast_thread.join(timeout=2.0)
    raise SystemExit("jtop Python package is required on the Jetson host") from exc

  full_path = args.output_dir / "jetson_telemetry.csv"
  with full_path.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.writer(output_file)
    writer.writerow([
        "timestamp_unix_ns",
        "timestamp_iso",
        "gpu_freq_khz",
        "gpu_load_pct",
        "emc_freq_khz",
        "emc_load_pct",
        "temp_gpu_c",
        "temp_cpu_c",
        "temp_soc0_c",
        "temp_tj_c",
        "vdd_in_mw",
        "vdd_cpu_gpu_cv_mw",
        "vdd_soc_mw",
    ])

    def record(jetson) -> None:
      gpu = jetson.gpu.get("gpu", {})
      gpu_frequency = gpu.get("freq", {})
      emc = jetson.memory.get("EMC", {})
      temperatures = jetson.temperature
      power = jetson.power
      rails = power.get("rail", {})
      timestamp_unix_ns = time.time_ns()
      writer.writerow([
          timestamp_unix_ns,
          local_iso_timestamp(timestamp_unix_ns),
          gpu_frequency.get("cur"),
          gpu.get("status", {}).get("load"),
          emc.get("cur"),
          emc.get("val"),
          temperatures.get("gpu", {}).get("temp"),
          temperatures.get("cpu", {}).get("temp"),
          temperatures.get("soc0", {}).get("temp"),
          temperatures.get("tj", {}).get("temp"),
          power.get("tot", {}).get("power"),
          rails.get("VDD_CPU_GPU_CV", {}).get("power"),
          rails.get("VDD_SOC", {}).get("power"),
      ])
      output_file.flush()

    jetson = jtop(interval=args.full_interval_ms / 1000.0)
    jetson_holder.append(jetson)
    jetson.attach(record)
    try:
      jetson.loop_for_ever()
    finally:
      request_stop()
      fast_thread.join(timeout=2.0)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
