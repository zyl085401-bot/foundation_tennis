from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import yaml


DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.yaml"


def _require_mapping(value, name: str) -> dict:
  if not isinstance(value, dict):
    raise RuntimeError(f"{name} must be a YAML mapping")
  return value


def _require_topic(config: dict, name: str) -> str:
  value = config.get(name)
  if not isinstance(value, str) or not value.strip():
    raise RuntimeError(f"camera.mros.{name} must be a non-empty topic name")
  return value.strip()


def _stamp_seconds(message) -> float:
  stamp = message.header.stamp
  return float(stamp.sec) + float(stamp.nsec) * 1e-9


def _color_image_to_rgb(message) -> np.ndarray:
  encoding = str(message.encoding).lower()
  channel_counts = {
      "rgb8": 3,
      "bgr8": 3,
      "rgba8": 4,
      "bgra8": 4,
  }
  if encoding not in channel_counts:
    raise RuntimeError(
        f"Unsupported color encoding {message.encoding!r}; expected rgb8, bgr8, rgba8, or bgra8"
    )

  width = int(message.width)
  height = int(message.height)
  channels = channel_counts[encoding]
  row_bytes = width * channels
  step = int(message.step)
  if width <= 0 or height <= 0:
    raise RuntimeError(f"Color image dimensions must be positive, got {width}x{height}")
  if step < row_bytes:
    raise RuntimeError(f"Color image step {step} is smaller than the required row size {row_bytes}")
  raw = np.frombuffer(message.data, dtype=np.uint8)
  required_bytes = step * height
  if raw.size < required_bytes:
    raise RuntimeError(f"Color image data is truncated: {raw.size} < {required_bytes} bytes")

  rows = raw[:required_bytes].reshape(height, step)
  image = rows[:, :row_bytes].reshape(height, width, channels)
  if encoding == "rgb8":
    return np.ascontiguousarray(image)
  if encoding == "bgr8":
    return np.ascontiguousarray(image[..., ::-1])
  if encoding == "rgba8":
    return np.ascontiguousarray(image[..., :3])
  return np.ascontiguousarray(image[..., [2, 1, 0]])


def _depth_image_to_meters(message, depth_scale: float) -> np.ndarray:
  encoding = str(message.encoding).lower()
  if encoding in ("16uc1", "mono16"):
    dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
    scale = float(depth_scale)
  elif encoding == "32fc1":
    dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    scale = 1.0
  else:
    raise RuntimeError(
        f"Unsupported depth encoding {message.encoding!r}; expected 16UC1, mono16, or 32FC1"
    )

  width = int(message.width)
  height = int(message.height)
  step = int(message.step)
  if width <= 0 or height <= 0:
    raise RuntimeError(f"Depth image dimensions must be positive, got {width}x{height}")
  if step % dtype.itemsize != 0:
    raise RuntimeError(f"Depth image step {step} is not aligned to {dtype.itemsize}-byte pixels")
  row_elements = step // dtype.itemsize
  if row_elements < width:
    raise RuntimeError(f"Depth image step contains {row_elements} pixels but width is {width}")
  raw = np.frombuffer(message.data, dtype=dtype)
  required_elements = row_elements * height
  if raw.size < required_elements:
    raise RuntimeError(f"Depth image data is truncated: {raw.size} < {required_elements} elements")

  depth = raw[:required_elements].reshape(height, row_elements)[:, :width]
  return np.ascontiguousarray(depth, dtype=np.float32) * scale


def _camera_matrix(message) -> np.ndarray:
  values = message.K
  if len(values) != 9:
    raise RuntimeError(f"CameraInfo.K must contain 9 values, got {len(values)}")
  matrix = np.asarray(values, dtype=np.float32).reshape(3, 3)
  if not np.isfinite(matrix).all() or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
    raise RuntimeError(f"CameraInfo contains invalid intrinsics: {matrix.tolist()}")
  return np.ascontiguousarray(matrix)


@dataclass(frozen=True)
class _TimedMessage:
  timestamp: float
  message: object


class ApproximateImageSynchronizer:
  """Pair color and depth messages by nearest header timestamp without reusing messages."""

  def __init__(self, queue_size: int, slop_sec: float):
    self.queue_size = int(queue_size)
    self.slop_sec = float(slop_sec)
    if self.queue_size <= 0:
      raise ValueError("queue_size must be positive")
    if self.slop_sec < 0.0 or not math.isfinite(self.slop_sec):
      raise ValueError("slop_sec must be non-negative and finite")
    self._colors: deque[_TimedMessage] = deque(maxlen=self.queue_size)
    self._depths: deque[_TimedMessage] = deque(maxlen=self.queue_size)

  def add_color(self, message) -> None:
    self._colors.append(self._timed(message, "Color"))

  def add_depth(self, message) -> None:
    self._depths.append(self._timed(message, "Depth"))

  def pop_match(self) -> tuple[object, object] | None:
    if not self._colors or not self._depths:
      return None
    best = min(
        (
            (abs(color.timestamp - depth.timestamp), color_index, depth_index)
            for color_index, color in enumerate(self._colors)
            for depth_index, depth in enumerate(self._depths)
        ),
        key=lambda candidate: candidate[0],
    )
    delta, color_index, depth_index = best
    if delta > self.slop_sec:
      self._discard_unmatchable_oldest()
      return None
    color = self._remove_at(self._colors, color_index)
    depth = self._remove_at(self._depths, depth_index)
    return color.message, depth.message

  @staticmethod
  def _timed(message, label: str) -> _TimedMessage:
    timestamp = _stamp_seconds(message)
    if timestamp < 0.0 or not math.isfinite(timestamp):
      raise RuntimeError(f"{label} message contains an invalid timestamp: {timestamp}")
    return _TimedMessage(timestamp, message)

  def _discard_unmatchable_oldest(self) -> None:
    oldest_color = self._colors[0].timestamp
    oldest_depth = self._depths[0].timestamp
    if oldest_color < oldest_depth - self.slop_sec:
      self._colors.popleft()
    elif oldest_depth < oldest_color - self.slop_sec:
      self._depths.popleft()

  @staticmethod
  def _remove_at(messages: deque[_TimedMessage], index: int) -> _TimedMessage:
    messages.rotate(-index)
    try:
      return messages.popleft()
    finally:
      messages.rotate(index)


class MrosRgbdReader:
  def __init__(
      self,
      color_topic: str,
      depth_topic: str,
      camera_info_topic: str,
      width: int = 640,
      height: int = 480,
      depth_min: float = 0.001,
      depth_max: float = 3.0,
      depth_scale: float = 0.001,
      sync_queue_size: int = 10,
      sync_slop_sec: float = 0.03,
      frame_timeout_sec: float = 5.0,
      node_name: str = "foundationpose_rgbd_reader",
      reliable: bool = False,
  ):
    self.color_topic = str(color_topic)
    self.depth_topic = str(depth_topic)
    self.camera_info_topic = str(camera_info_topic)
    self.width = int(width)
    self.height = int(height)
    self.depth_min = float(depth_min)
    self.depth_max = float(depth_max)
    self.depth_scale = float(depth_scale)
    self.sync_queue_size = int(sync_queue_size)
    self.sync_slop_sec = float(sync_slop_sec)
    self.frame_timeout_sec = float(frame_timeout_sec)
    self.node_name = str(node_name)
    self.reliable = bool(reliable)

    if self.width <= 0 or self.height <= 0:
      raise ValueError("width and height must be positive")
    if not 0.0 <= self.depth_min < self.depth_max:
      raise ValueError("depth range must satisfy 0 <= depth_min < depth_max")
    if self.depth_scale <= 0.0:
      raise ValueError("depth_scale must be positive")
    if self.frame_timeout_sec <= 0.0:
      raise ValueError("frame_timeout_sec must be positive")
    if not self.node_name:
      raise ValueError("node_name must not be empty")

    self._synchronizer = ApproximateImageSynchronizer(
        queue_size=self.sync_queue_size,
        slop_sec=self.sync_slop_sec,
    )
    self._mros = None
    self._color_subscriber = None
    self._depth_subscriber = None
    self._camera_info_subscriber = None
    self._camera_info = None
    self._started = False
    self._stopping = False
    self._last_error_log_time = 0.0
    self._last_timeout_log_time = 0.0

  @classmethod
  def from_camera_config(cls, camera_config: dict) -> "MrosRgbdReader":
    camera_config = _require_mapping(camera_config, "camera")
    mros_config = _require_mapping(camera_config.get("mros"), "camera.mros")
    return cls(
        color_topic=_require_topic(mros_config, "color_topic"),
        depth_topic=_require_topic(mros_config, "depth_topic"),
        camera_info_topic=_require_topic(mros_config, "camera_info_topic"),
        width=int(camera_config.get("width", 640)),
        height=int(camera_config.get("height", 480)),
        depth_min=float(camera_config.get("depth_min", 0.001)),
        depth_max=float(camera_config.get("depth_max", 3.0)),
        depth_scale=float(mros_config.get("depth_scale", 0.001)),
        sync_queue_size=int(mros_config.get("sync_queue_size", 10)),
        sync_slop_sec=float(mros_config.get("sync_slop_sec", 0.03)),
        frame_timeout_sec=float(mros_config.get("frame_timeout_sec", 5.0)),
        node_name=str(mros_config.get("reader_node_name", "foundationpose_rgbd_reader")),
        reliable=bool(mros_config.get("reliable", False)),
    )

  def start(self) -> None:
    if self._started:
      raise RuntimeError("MrosRgbdReader.start() was called more than once")
    try:
      import mros
      from mros.sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
      raise RuntimeError(
          "mROS camera input is enabled, but the mROS Python package is unavailable"
      ) from exc

    mros.init(self.node_name)
    try:
      self._color_subscriber = mros.subscribe(
          self.color_topic, Image, None, self.sync_queue_size, False, self.reliable
      )
      self._depth_subscriber = mros.subscribe(
          self.depth_topic, Image, None, self.sync_queue_size, False, self.reliable
      )
      self._camera_info_subscriber = mros.subscribe(
          self.camera_info_topic, CameraInfo, None, 1, True, self.reliable
      )
    except Exception:
      mros.shutdown()
      raise

    self._mros = mros
    self._started = True
    self._stopping = False
    print("[mROS RGB-D] Subscriptions started")
    print(f"  color: {self.color_topic}")
    print(f"  aligned depth: {self.depth_topic}")
    print(f"  camera info: {self.camera_info_topic}")
    print(f"  expected size: {self.width}x{self.height}")
    print(f"  sync queue/slop: {self.sync_queue_size}/{self.sync_slop_sec:.3f} s")

  def stop(self) -> None:
    if not self._started:
      return
    self._stopping = True
    try:
      if self._mros is not None:
        self._mros.shutdown()
    finally:
      self._color_subscriber = None
      self._depth_subscriber = None
      self._camera_info_subscriber = None
      self._mros = None
      self._started = False

  def get_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if not self._started:
      raise RuntimeError("MrosRgbdReader.get_frame() called before start()")
    deadline = time.monotonic() + self.frame_timeout_sec
    while not self._stopping:
      if self._mros is None or not self._mros.ok():
        raise RuntimeError("mROS stopped while waiting for RGB-D input")
      try:
        camera_info = self._camera_info_subscriber.readMsgRT()
        if camera_info is not None:
          self._validate_camera_info_size(camera_info)
          self._camera_info = camera_info
        color_message = self._color_subscriber.readMsgRT()
        if color_message is not None:
          self._synchronizer.add_color(color_message)
        depth_message = self._depth_subscriber.readMsgRT()
        if depth_message is not None:
          self._synchronizer.add_depth(depth_message)

        match = self._synchronizer.pop_match()
        if match is not None and self._camera_info is not None:
          return self._convert_frame(match[0], match[1], self._camera_info)
      except Exception as exc:
        self._log_error(exc)

      remaining = deadline - time.monotonic()
      if remaining <= 0.0:
        self._log_timeout()
        return None
      time.sleep(min(0.001, remaining))
    return None

  def _convert_frame(self, color_message, depth_message, camera_info):
    color = _color_image_to_rgb(color_message)
    depth = _depth_image_to_meters(depth_message, self.depth_scale)
    K = _camera_matrix(camera_info)
    if color.shape != (self.height, self.width, 3):
      raise RuntimeError(
          f"mROS color shape {color.shape} does not match configured {(self.height, self.width, 3)}"
      )
    if depth.shape != (self.height, self.width):
      raise RuntimeError(
          f"mROS aligned-depth shape {depth.shape} does not match configured {(self.height, self.width)}"
      )
    depth[~np.isfinite(depth)] = 0.0
    depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0
    timestamp = _stamp_seconds(color_message)
    return color, depth, K, timestamp

  def _validate_camera_info_size(self, camera_info) -> None:
    if int(camera_info.width) != self.width or int(camera_info.height) != self.height:
      raise RuntimeError(
          "CameraInfo size does not match configured RGB size: "
          f"{camera_info.width}x{camera_info.height} != {self.width}x{self.height}"
      )

  def _log_error(self, exc: Exception) -> None:
    now = time.monotonic()
    if now - self._last_error_log_time < 1.0:
      return
    self._last_error_log_time = now
    print(f"[mROS RGB-D] Rejected input: {exc}")

  def _log_timeout(self) -> None:
    now = time.monotonic()
    if now - self._last_timeout_log_time < 1.0:
      return
    self._last_timeout_log_time = now
    print(
        f"[mROS RGB-D] No synchronized frame received for {self.frame_timeout_sec:.1f} s; "
        "check publishers, topic names, reliability, CameraInfo, and RGB/depth timestamps"
    )


def _load_camera_config(path: Path) -> dict:
  resolved = path.expanduser().resolve()
  if not resolved.is_file():
    raise RuntimeError(f"Configuration file does not exist: {resolved}")
  with resolved.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}
  return _require_mapping(config.get("camera"), "camera")


def _run_check(config_path: Path, frame_count: int) -> None:
  reader = MrosRgbdReader.from_camera_config(_load_camera_config(config_path))
  timestamps = []
  started_at = time.monotonic()
  reader.start()
  try:
    while len(timestamps) < frame_count:
      frame = reader.get_frame()
      if frame is None:
        continue
      color, depth, K, timestamp = frame
      timestamps.append(timestamp)
      if len(timestamps) == 1:
        valid_depth = depth[depth > 0.0]
        median_depth = float(np.median(valid_depth)) if valid_depth.size else float("nan")
        print(
            f"[mROS RGB-D CHECK] first frame: color={color.shape}/{color.dtype}, "
            f"depth={depth.shape}/{depth.dtype}, valid_depth={valid_depth.size}, "
            f"median_depth={median_depth:.3f} m, timestamp={timestamp:.9f}"
        )
        print(f"[mROS RGB-D CHECK] K:\n{K}")
  finally:
    reader.stop()

  elapsed = time.monotonic() - started_at
  intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
  if intervals.size and np.any(intervals <= 0.0):
    raise RuntimeError("mROS image timestamps are not strictly increasing")
  source_fps = float(1.0 / np.median(intervals)) if intervals.size else float("nan")
  receive_fps = float(frame_count / max(elapsed, 1e-9))
  print(
      f"[mROS RGB-D CHECK] PASS: frames={frame_count}, "
      f"source_fps={source_fps:.2f}, receive_fps={receive_fps:.2f}"
  )


def main() -> int:
  parser = argparse.ArgumentParser(description="Validate synchronized mROS RGB-D input.")
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
  parser.add_argument("--frames", type=int, default=30)
  args = parser.parse_args()
  if args.frames <= 0:
    parser.error("--frames must be positive")
  try:
    _run_check(args.config, args.frames)
  except (KeyboardInterrupt, OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
    print(f"ERROR: {exc}")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
