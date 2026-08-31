from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

try:
  from .rgbd_shared_memory import LatestRgbdFrameBuffer
except ImportError:
  from rgbd_shared_memory import LatestRgbdFrameBuffer


class MrosRgbdSharedMemoryReader:
  def __init__(
      self,
      config_path: Path,
      width: int,
      height: int,
      bridge_cpu_cores: list[int],
      model_cpu_cores: list[int],
      shared_slot_count: int = 3,
      startup_timeout_sec: float = 10.0,
  ):
    self.config_path = Path(config_path).expanduser().resolve()
    self.width = int(width)
    self.height = int(height)
    self.bridge_cpu_cores = self._validate_cpu_cores(bridge_cpu_cores, "bridge_cpu_cores")
    self.model_cpu_cores = self._validate_cpu_cores(model_cpu_cores, "model_cpu_cores")
    self.shared_slot_count = int(shared_slot_count)
    self.startup_timeout_sec = float(startup_timeout_sec)
    if self.width <= 0 or self.height <= 0:
      raise ValueError("mROS shared-memory width and height must be positive")
    if self.startup_timeout_sec <= 0:
      raise ValueError("mROS bridge startup_timeout_sec must be positive")
    if self.shared_slot_count < 2:
      raise ValueError("mROS shared_slot_count must be at least 2")
    overlap = set(self.bridge_cpu_cores) & set(self.model_cpu_cores)
    if overlap:
      raise ValueError(f"mROS bridge and model CPU cores must be disjoint; overlap={sorted(overlap)}")

    self.available_cpu_cores = sorted(os.sched_getaffinity(0))
    self.bridge_path = Path(__file__).with_name("mros_rgbd_bridge.py").resolve()
    self.shared_name = f"foundationpose_rgbd_{os.getpid()}_{uuid.uuid4().hex}"
    self.shared_frame = None
    self.process = None
    self.output_thread = None
    self.output_queue = queue.Queue()
    self.output_tail = deque(maxlen=40)
    self.output_lock = threading.Lock()
    self.io_lock = threading.Lock()
    self.stopping = threading.Event()
    self.last_sequence = 0
    self.started = False

  @classmethod
  def from_config(
      cls,
      config_path: Path,
      camera_config: dict,
      mros_config: dict,
  ) -> "MrosRgbdSharedMemoryReader":
    return cls(
        config_path=config_path,
        width=int(camera_config.get("width", 640)),
        height=int(camera_config.get("height", 480)),
        bridge_cpu_cores=mros_config.get("bridge_cpu_cores", [0, 1, 2]),
        model_cpu_cores=mros_config.get("model_cpu_cores", [3, 4, 5, 6, 7]),
        shared_slot_count=int(mros_config.get("shared_slot_count", 3)),
        startup_timeout_sec=float(mros_config.get("startup_timeout_sec", 10.0)),
    )

  @staticmethod
  def _validate_cpu_cores(value, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
      raise ValueError(f"camera.mros.{name} must be a non-empty list")
    cores = [int(core) for core in value]
    if any(core < 0 for core in cores):
      raise ValueError(f"camera.mros.{name} contains a negative core index: {cores}")
    if len(set(cores)) != len(cores):
      raise ValueError(f"camera.mros.{name} contains duplicate cores: {cores}")
    return cores

  def apply_model_cpu_affinity(self) -> None:
    self._validate_cores_available(self.model_cpu_cores, "model")
    for task_path in Path("/proc/self/task").iterdir():
      try:
        os.sched_setaffinity(int(task_path.name), self.model_cpu_cores)
      except ProcessLookupError:
        continue
    print(
        f"[CPU AFFINITY] FoundationPose pid={os.getpid()} cores="
        f"{sorted(os.sched_getaffinity(0))}"
    )

  def start(self) -> None:
    if self.started:
      return
    self.stopping.clear()
    self.output_queue = queue.Queue()
    with self.output_lock:
      self.output_tail.clear()
    self._validate_cores_available(self.bridge_cpu_cores, "mROS bridge")
    taskset_path = shutil.which("taskset")
    if taskset_path is None:
      raise RuntimeError("taskset is required for mROS bridge CPU isolation")

    self.shared_frame = LatestRgbdFrameBuffer.create(
        name=self.shared_name,
        width=self.width,
        height=self.height,
        slot_count=self.shared_slot_count,
    )
    command = [
        taskset_path,
        "--cpu-list",
        ",".join(str(core) for core in self.bridge_cpu_cores),
        sys.executable,
        str(self.bridge_path),
        "--config",
        str(self.config_path),
        "--parent-pid",
        str(os.getpid()),
        "--shm-name",
        self.shared_name,
        "--width",
        str(self.width),
        "--height",
        str(self.height),
        "--slot-count",
        str(self.shared_slot_count),
        "--cpu-cores",
        ",".join(str(core) for core in self.bridge_cpu_cores),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
      self.process = subprocess.Popen(
          command,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          bufsize=1,
          env=environment,
          start_new_session=True,
      )
      self.output_thread = threading.Thread(
          target=self._collect_output,
          name="mros-rgbd-bridge-output",
          daemon=True,
      )
      self.output_thread.start()
      self._wait_until_ready()
      self.started = True
      self.last_sequence = 0
    except BaseException:
      self.stop()
      raise

  def stop(self) -> None:
    self.stopping.set()
    process = self.process
    termination_error = None
    try:
      if process is not None and process.poll() is None:
        try:
          os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
          pass
        try:
          process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
          try:
            os.killpg(process.pid, signal.SIGKILL)
          except ProcessLookupError:
            pass
          process.wait(timeout=2.0)
    except Exception as exc:
      termination_error = exc
    finally:
      output_thread = self.output_thread
      if output_thread is not None:
        output_thread.join(timeout=1.0)
      if process is not None and process.stdout is not None:
        try:
          process.stdout.close()
        except (OSError, ValueError) as exc:
          if termination_error is None:
            termination_error = exc
      if output_thread is not None and output_thread.is_alive():
        output_thread.join(timeout=1.0)

      with self.io_lock:
        shared_frame = self.shared_frame
        self.shared_frame = None
        if shared_frame is not None:
          try:
            shared_frame.close(unlink=True)
          except Exception as exc:
            if termination_error is None:
              termination_error = exc

      self.process = None
      self.output_thread = None
      self.started = False
    if termination_error is not None:
      print(f"[mROS SHM] Warning: bridge termination was incomplete: {termination_error}")

  def get_frame(self, timeout_sec: float = 5.0):
    if not self.started or self.shared_frame is None:
      raise RuntimeError("mROS shared-memory reader is not started")
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
      if self.stopping.is_set():
        return None
      with self.io_lock:
        shared_frame = self.shared_frame
        if shared_frame is None:
          return None
        frame = shared_frame.read_latest(self.last_sequence)
      if frame is not None:
        sequence, color, depth, K, timestamp = frame
        self.last_sequence = sequence
        return color, depth, K, timestamp
      self._raise_if_bridge_failed()
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return None
      time.sleep(min(0.001, remaining))

  def _wait_until_ready(self) -> None:
    deadline = time.monotonic() + self.startup_timeout_sec
    while True:
      self._raise_if_bridge_failed()
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise TimeoutError(
            f"mROS shared-memory bridge did not become ready within "
            f"{self.startup_timeout_sec:.1f} seconds{self._formatted_output_tail()}"
        )
      try:
        line = self.output_queue.get(timeout=min(0.1, remaining))
      except queue.Empty:
        continue
      if line.startswith("BRIDGE_READY "):
        print(f"[mROS SHM] {line}")
        return
      if line.startswith("BRIDGE_ERROR "):
        raise RuntimeError(f"mROS shared-memory bridge failed: {line}{self._formatted_output_tail()}")

  def _raise_if_bridge_failed(self) -> None:
    if self.process is None:
      raise RuntimeError("mROS shared-memory bridge process was not created")
    return_code = self.process.poll()
    if return_code is not None:
      raise RuntimeError(
          f"mROS shared-memory bridge exited with code {return_code}"
          f"{self._formatted_output_tail()}"
      )

  def _collect_output(self) -> None:
    process = self.process
    if process is None or process.stdout is None:
      return
    try:
      for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        with self.output_lock:
          self.output_tail.append(line)
        self.output_queue.put(line)
        if not line.startswith("BRIDGE_READY "):
          print(f"[mROS SHM BRIDGE] {line}")
    except (OSError, ValueError):
      if not self.stopping.is_set():
        raise

  def _formatted_output_tail(self) -> str:
    with self.output_lock:
      lines = list(self.output_tail)
    if not lines:
      return ""
    return "\nBridge output:\n" + "\n".join(lines)

  def _validate_cores_available(self, cores: list[int], role: str) -> None:
    unavailable = sorted(set(cores) - set(self.available_cpu_cores))
    if unavailable:
      raise ValueError(
          f"Configured {role} CPU cores {unavailable} are unavailable; "
          f"allowed cores={self.available_cpu_cores}"
      )
