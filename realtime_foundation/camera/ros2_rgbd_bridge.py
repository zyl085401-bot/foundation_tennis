#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import signal
import sys
import threading
import traceback

from rgbd_shared_memory import LatestRgbdFrameBuffer


PR_SET_PDEATHSIG = 1


def _set_parent_death_signal(expected_parent_pid: int) -> None:
  parent_pid_before = os.getppid()
  if parent_pid_before != expected_parent_pid:
    raise RuntimeError(
        f"FoundationPose parent changed before ROS2 bridge startup: "
        f"expected {expected_parent_pid}, got {parent_pid_before}"
    )
  libc = ctypes.CDLL(None, use_errno=True)
  if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))
  parent_pid_after = os.getppid()
  if parent_pid_after != expected_parent_pid:
    raise RuntimeError(
        f"FoundationPose parent changed while ROS2 bridge was starting: "
        f"expected {expected_parent_pid}, got {parent_pid_after}"
    )


def _parse_cpu_cores(value: str) -> list[int]:
  cores = [int(item.strip()) for item in value.split(",") if item.strip()]
  if not cores:
    raise ValueError("At least one ROS2 bridge CPU core is required")
  if any(core < 0 for core in cores):
    raise ValueError(f"CPU core indices must be non-negative: {cores}")
  return cores


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="ROS2 RGB-D to FoundationPose shared-memory bridge")
  parser.add_argument("--config", required=True, type=Path)
  parser.add_argument("--parent-pid", required=True, type=int)
  parser.add_argument("--shm-name", required=True)
  parser.add_argument("--width", required=True, type=int)
  parser.add_argument("--height", required=True, type=int)
  parser.add_argument("--slot-count", required=True, type=int)
  parser.add_argument("--cpu-cores", required=True)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  shared_frame = None
  reader = None
  stop_event = threading.Event()

  def request_stop(signum, _frame):
    print(f"[ROS2 SHM BRIDGE] received signal {signum}; stopping", flush=True)
    stop_event.set()

  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)
  try:
    _set_parent_death_signal(args.parent_pid)
    cpu_cores = _parse_cpu_cores(args.cpu_cores)
    os.sched_setaffinity(0, cpu_cores)
    shared_frame = LatestRgbdFrameBuffer.attach(
        name=args.shm_name,
        width=args.width,
        height=args.height,
        slot_count=args.slot_count,
    )

    from ros2_rgbd_reader import Ros2RgbdReader
    import yaml

    with args.config.expanduser().resolve().open("r", encoding="utf-8") as config_file:
      config = yaml.safe_load(config_file) or {}
    camera_config = dict(config.get("camera", {}) or {})
    ros2_config = dict(camera_config.get("ros2", {}) or {})
    ros2_config["frame_timeout_sec"] = 0.2
    camera_config["ros2"] = ros2_config
    reader = Ros2RgbdReader.from_camera_config(camera_config)
    reader.start()

    ready = False
    while not stop_event.is_set():
      frame = reader.get_frame()
      if frame is None:
        continue
      color, depth, K, timestamp = frame
      shared_frame.write(color, depth, K, timestamp)
      if not ready:
        ready = True
        print(
            f"BRIDGE_READY pid={os.getpid()} affinity={sorted(os.sched_getaffinity(0))}",
            flush=True,
        )
    return 0
  except Exception as exc:
    print(f"BRIDGE_ERROR {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc(file=sys.stdout)
    return 1
  finally:
    if reader is not None:
      try:
        reader.stop()
      except Exception:
        traceback.print_exc(file=sys.stdout)
    if shared_frame is not None:
      shared_frame.close(unlink=True)


if __name__ == "__main__":
  raise SystemExit(main())
