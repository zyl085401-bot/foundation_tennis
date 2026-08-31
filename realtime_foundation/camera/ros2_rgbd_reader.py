from __future__ import annotations

import argparse
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import yaml

import message_filters
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.yaml"


def _require_mapping(value, name: str) -> dict:
  if not isinstance(value, dict):
    raise RuntimeError(f"{name} must be a YAML mapping")
  return value


def _require_topic(config: dict, name: str) -> str:
  value = config.get(name)
  if not isinstance(value, str) or not value.startswith("/"):
    raise RuntimeError(f"camera.ros2.{name} must be an absolute ROS topic")
  return value


def _stamp_seconds(message) -> float:
  stamp = message.header.stamp
  return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _color_image_to_rgb(message: Image) -> np.ndarray:
  encoding = message.encoding.lower()
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

  channels = channel_counts[encoding]
  row_bytes = int(message.width) * channels
  step = int(message.step)
  height = int(message.height)
  if step < row_bytes:
    raise RuntimeError(f"Color image step {step} is smaller than the required row size {row_bytes}")
  raw = np.frombuffer(message.data, dtype=np.uint8)
  required_bytes = step * height
  if raw.size < required_bytes:
    raise RuntimeError(f"Color image data is truncated: {raw.size} < {required_bytes} bytes")

  rows = raw[:required_bytes].reshape(height, step)
  image = rows[:, :row_bytes].reshape(height, int(message.width), channels)
  if encoding == "rgb8":
    return np.ascontiguousarray(image)
  if encoding == "bgr8":
    return np.ascontiguousarray(image[..., ::-1])
  if encoding == "rgba8":
    return np.ascontiguousarray(image[..., :3])
  return np.ascontiguousarray(image[..., [2, 1, 0]])


def _depth_image_to_meters(message: Image, depth_scale: float) -> np.ndarray:
  encoding = message.encoding.lower()
  if encoding in ("16uc1", "mono16"):
    dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    scale = depth_scale
  elif encoding == "32fc1":
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    scale = 1.0
  else:
    raise RuntimeError(
        f"Unsupported depth encoding {message.encoding!r}; expected 16UC1, mono16, or 32FC1"
    )

  step = int(message.step)
  height = int(message.height)
  width = int(message.width)
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
  return np.ascontiguousarray(depth, dtype=np.float32) * float(scale)


def _camera_matrix(message: CameraInfo) -> np.ndarray:
  if len(message.k) != 9:
    raise RuntimeError(f"CameraInfo.k must contain 9 values, got {len(message.k)}")
  matrix = np.asarray(message.k, dtype=np.float32).reshape(3, 3)
  if not np.isfinite(matrix).all() or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
    raise RuntimeError(f"CameraInfo contains invalid intrinsics: {matrix.tolist()}")
  return np.ascontiguousarray(matrix)


class Ros2RgbdReader:
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
      verbose: bool = False,
  ):
    self.color_topic = color_topic
    self.depth_topic = depth_topic
    self.camera_info_topic = camera_info_topic
    self.width = int(width)
    self.height = int(height)
    self.depth_min = float(depth_min)
    self.depth_max = float(depth_max)
    self.depth_scale = float(depth_scale)
    self.sync_queue_size = int(sync_queue_size)
    self.sync_slop_sec = float(sync_slop_sec)
    self.frame_timeout_sec = float(frame_timeout_sec)
    self.node_name = node_name
    self.verbose = verbose

    if self.width <= 0 or self.height <= 0:
      raise ValueError("width and height must be positive")
    if not 0.0 <= self.depth_min < self.depth_max:
      raise ValueError("depth range must satisfy 0 <= depth_min < depth_max")
    if self.depth_scale <= 0.0:
      raise ValueError("depth_scale must be positive")
    if self.sync_queue_size <= 0:
      raise ValueError("sync_queue_size must be positive")
    if self.sync_slop_sec < 0.0:
      raise ValueError("sync_slop_sec must be non-negative")
    if self.frame_timeout_sec <= 0.0:
      raise ValueError("frame_timeout_sec must be positive")

    self._condition = threading.Condition()
    self._latest_frame = None
    self._latest_sequence = 0
    self._delivered_sequence = 0
    self._started = False
    self._stopping = False
    self._last_error_log_time = 0.0
    self._last_timeout_log_time = 0.0
    self._context: Context | None = None
    self._node: Node | None = None
    self._executor: SingleThreadedExecutor | None = None
    self._spin_thread: threading.Thread | None = None
    self._subscribers = []
    self._synchronizer = None

  @classmethod
  def from_camera_config(cls, camera_config: dict, verbose: bool = False) -> "Ros2RgbdReader":
    camera_config = _require_mapping(camera_config, "camera")
    ros2_config = _require_mapping(camera_config.get("ros2"), "camera.ros2")
    return cls(
        color_topic=_require_topic(ros2_config, "color_topic"),
        depth_topic=_require_topic(ros2_config, "depth_topic"),
        camera_info_topic=_require_topic(ros2_config, "camera_info_topic"),
        width=int(camera_config.get("width", 640)),
        height=int(camera_config.get("height", 480)),
        depth_min=float(camera_config.get("depth_min", 0.001)),
        depth_max=float(camera_config.get("depth_max", 3.0)),
        depth_scale=float(ros2_config.get("depth_scale", 0.001)),
        sync_queue_size=int(ros2_config.get("sync_queue_size", 10)),
        sync_slop_sec=float(ros2_config.get("sync_slop_sec", 0.03)),
        frame_timeout_sec=float(ros2_config.get("frame_timeout_sec", 5.0)),
        node_name=str(ros2_config.get("reader_node_name", "foundationpose_rgbd_reader")),
        verbose=verbose,
    )

  def start(self) -> None:
    if self._started:
      raise RuntimeError("Ros2RgbdReader.start() was called more than once")

    self._context = Context()
    rclpy.init(args=None, context=self._context)
    self._node = Node(self.node_name, context=self._context)
    self._executor = SingleThreadedExecutor(context=self._context)
    self._executor.add_node(self._node)

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, self.sync_queue_size),
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    color_subscriber = message_filters.Subscriber(
        self._node, Image, self.color_topic, qos_profile=qos
    )
    depth_subscriber = message_filters.Subscriber(
        self._node, Image, self.depth_topic, qos_profile=qos
    )
    camera_info_subscriber = message_filters.Subscriber(
        self._node, CameraInfo, self.camera_info_topic, qos_profile=qos
    )
    self._subscribers = [color_subscriber, depth_subscriber, camera_info_subscriber]
    self._synchronizer = message_filters.ApproximateTimeSynchronizer(
        self._subscribers,
        queue_size=self.sync_queue_size,
        slop=self.sync_slop_sec,
        allow_headerless=False,
    )
    self._synchronizer.registerCallback(self._synchronized_callback)

    self._started = True
    self._spin_thread = threading.Thread(
        target=self._executor.spin,
        name="ros2-rgbd-executor",
        daemon=True,
    )
    self._spin_thread.start()
    print("[ROS2 RGB-D] Subscriptions started")
    print(f"  color: {self.color_topic}")
    print(f"  aligned depth: {self.depth_topic}")
    print(f"  camera info: {self.camera_info_topic}")
    print(f"  expected size: {self.width}x{self.height}")
    print(f"  sync queue/slop: {self.sync_queue_size}/{self.sync_slop_sec:.3f} s")

  def stop(self) -> None:
    if not self._started:
      return
    with self._condition:
      self._stopping = True
      self._condition.notify_all()

    if self._executor is not None:
      self._executor.shutdown(timeout_sec=2.0)
    if self._spin_thread is not None and self._spin_thread.is_alive():
      self._spin_thread.join(timeout=2.0)
    if self._node is not None:
      self._node.destroy_node()
    if self._context is not None and self._context.ok():
      self._context.shutdown()

    self._started = False
    self._subscribers = []
    self._synchronizer = None

  def get_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if not self._started:
      raise RuntimeError("Ros2RgbdReader.get_frame() called before start()")

    with self._condition:
      has_frame = self._condition.wait_for(
          lambda: self._stopping or self._latest_sequence != self._delivered_sequence,
          timeout=self.frame_timeout_sec,
      )
      if self._stopping:
        return None
      if not has_frame or self._latest_frame is None:
        self._log_timeout()
        return None
      self._delivered_sequence = self._latest_sequence
      return self._latest_frame

  def _synchronized_callback(
      self,
      color_message: Image,
      depth_message: Image,
      camera_info_message: CameraInfo,
  ) -> None:
    try:
      color = _color_image_to_rgb(color_message)
      depth = _depth_image_to_meters(depth_message, self.depth_scale)
      K = _camera_matrix(camera_info_message)
      self._validate_shapes(color, depth, camera_info_message)
      depth[~np.isfinite(depth)] = 0.0
      depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0
      timestamp = _stamp_seconds(color_message)
      if not math.isfinite(timestamp):
        raise RuntimeError(f"Color message contains a non-finite timestamp: {timestamp}")
    except Exception as exc:
      self._log_callback_error(exc)
      return

    with self._condition:
      self._latest_frame = (color, depth, K, timestamp)
      self._latest_sequence += 1
      self._condition.notify_all()

  def _validate_shapes(
      self,
      color: np.ndarray,
      depth: np.ndarray,
      camera_info: CameraInfo,
  ) -> None:
    if color.shape != (self.height, self.width, 3):
      raise RuntimeError(
          f"ROS color shape {color.shape} does not match configured {(self.height, self.width, 3)}"
      )
    if depth.shape != (self.height, self.width):
      raise RuntimeError(
          f"ROS aligned-depth shape {depth.shape} does not match configured {(self.height, self.width)}"
      )
    if int(camera_info.width) != self.width or int(camera_info.height) != self.height:
      raise RuntimeError(
          "CameraInfo size does not match configured RGB size: "
          f"{camera_info.width}x{camera_info.height} != {self.width}x{self.height}"
      )

  def _log_callback_error(self, exc: Exception) -> None:
    now = time.monotonic()
    if now - self._last_error_log_time < 1.0:
      return
    self._last_error_log_time = now
    message = f"Rejected synchronized frame: {exc}"
    if self._node is not None:
      self._node.get_logger().error(message)
    else:
      print(f"[ROS2 RGB-D] {message}")

  def _log_timeout(self) -> None:
    now = time.monotonic()
    if now - self._last_timeout_log_time < 1.0:
      return
    self._last_timeout_log_time = now
    print(
        f"[ROS2 RGB-D] No synchronized frame received for {self.frame_timeout_sec:.1f} s; "
        "check publishers, topic names, QoS, and RGB/depth timestamps"
    )


def _load_camera_config(path: Path) -> dict:
  resolved = path.expanduser().resolve()
  if not resolved.is_file():
    raise RuntimeError(f"Configuration file does not exist: {resolved}")
  with resolved.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}
  return _require_mapping(config.get("camera"), "camera")


def _run_check(config_path: Path, frame_count: int) -> None:
  reader = Ros2RgbdReader.from_camera_config(_load_camera_config(config_path), verbose=True)
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
            f"[ROS2 RGB-D CHECK] first frame: color={color.shape}/{color.dtype}, "
            f"depth={depth.shape}/{depth.dtype}, valid_depth={valid_depth.size}, "
            f"median_depth={median_depth:.3f} m, timestamp={timestamp:.9f}"
        )
        print(f"[ROS2 RGB-D CHECK] K:\n{K}")
  finally:
    reader.stop()

  elapsed = time.monotonic() - started_at
  intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
  if intervals.size and np.any(intervals <= 0.0):
    raise RuntimeError("ROS image timestamps are not strictly increasing")
  source_fps = float(1.0 / np.median(intervals)) if intervals.size else float("nan")
  receive_fps = float(frame_count / max(elapsed, 1e-9))
  print(
      f"[ROS2 RGB-D CHECK] PASS: frames={frame_count}, "
      f"source_fps={source_fps:.2f}, receive_fps={receive_fps:.2f}"
  )


def main() -> int:
  parser = argparse.ArgumentParser(description="Validate synchronized ROS 2 RGB-D input.")
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