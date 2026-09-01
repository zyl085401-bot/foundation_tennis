from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading
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


def _optional_topic(config: dict, name: str) -> str | None:
  value = config.get(name)
  if value is None:
    return None
  if not isinstance(value, str) or not value.strip():
    raise RuntimeError(f"camera.mros.{name} must be null or a non-empty topic name")
  return value.strip()


def _finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
  try:
    array = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError) as exc:
    raise RuntimeError(f"{name} must contain numeric values") from exc
  if array.shape != shape:
    raise RuntimeError(f"{name} must have shape {shape}, got {array.shape}")
  if not np.isfinite(array).all():
    raise RuntimeError(f"{name} must contain only finite values")
  return np.ascontiguousarray(array)


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


def _depth_image_to_native(message) -> np.ndarray:
  encoding = str(message.encoding).lower()
  if encoding in ("16uc1", "mono16"):
    dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
    native_dtype = np.uint16
  elif encoding == "32fc1":
    dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    native_dtype = np.float32
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
  return np.ascontiguousarray(depth, dtype=native_dtype)


def _depth_native_to_meters(depth: np.ndarray, depth_scale: float) -> np.ndarray:
  if depth.dtype == np.uint16:
    return np.ascontiguousarray(depth, dtype=np.float32) * float(depth_scale)
  if depth.dtype == np.float32:
    return np.array(depth, dtype=np.float32, order="C", copy=True)
  raise RuntimeError(f"Unsupported native depth dtype {depth.dtype}")


def _depth_image_to_meters(message, depth_scale: float) -> np.ndarray:
  return _depth_native_to_meters(_depth_image_to_native(message), depth_scale)


def _camera_matrix(message) -> np.ndarray:
  values = message.K
  if len(values) != 9:
    raise RuntimeError(f"CameraInfo.K must contain 9 values, got {len(values)}")
  matrix = np.asarray(values, dtype=np.float32).reshape(3, 3)
  if not np.isfinite(matrix).all() or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
    raise RuntimeError(f"CameraInfo contains invalid intrinsics: {matrix.tolist()}")
  return np.ascontiguousarray(matrix)


class DepthToColorAlignment:
  """Register a raw depth image into the configured color image plane."""

  def __init__(self, config: dict, width: int, height: int):
    config = _require_mapping(config, "camera.mros.depth_to_color_alignment")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
      raise RuntimeError("camera.mros.depth_to_color_alignment.enabled must be true or false")
    self.enabled = enabled
    self.backend = str(config.get("backend", "opencv_rgbd")).strip()
    if self.backend != "opencv_rgbd":
      raise RuntimeError(
          "camera.mros.depth_to_color_alignment.backend must be 'opencv_rgbd'"
      )
    depth_dilation = config.get("depth_dilation", False)
    if not isinstance(depth_dilation, bool):
      raise RuntimeError(
          "camera.mros.depth_to_color_alignment.depth_dilation must be true or false"
      )
    self.depth_dilation = depth_dilation
    self.width = int(width)
    self.height = int(height)

    calibration_width = int(config.get("calibration_width", 0))
    calibration_height = int(config.get("calibration_height", 0))
    calibration_fps = int(config.get("calibration_fps", 0))
    if calibration_width <= 0 or calibration_height <= 0 or calibration_fps <= 0:
      raise RuntimeError(
          "Depth-to-color calibration width, height, and fps must be positive"
      )

    validation = _require_mapping(
        config.get("validation", {}),
        "camera.mros.depth_to_color_alignment.validation",
    )
    require_resolution_match = validation.get("require_resolution_match", True)
    require_zero_distortion = validation.get("require_zero_distortion", True)
    if not isinstance(require_resolution_match, bool) or not isinstance(
        require_zero_distortion, bool
    ):
      raise RuntimeError("Depth-to-color validation switches must be true or false")
    if require_resolution_match and (
        calibration_width != self.width or calibration_height != self.height
    ):
      raise RuntimeError(
          "Depth-to-color calibration resolution does not match camera input: "
          f"{calibration_width}x{calibration_height} != {self.width}x{self.height}"
      )

    depth = _require_mapping(config.get("depth"), "depth-to-color depth calibration")
    color = _require_mapping(config.get("color"), "depth-to-color color calibration")
    extrinsics = _require_mapping(
        config.get("depth_to_color"), "depth-to-color extrinsics"
    )
    self.depth_K = self._validate_intrinsics(
        _finite_array(depth.get("K"), (3, 3), "depth calibration K"), "depth"
    )
    self.color_K = self._validate_intrinsics(
        _finite_array(color.get("K"), (3, 3), "color calibration K"), "color"
    )
    self.depth_distortion_model = str(depth.get("distortion_model", "")).strip().lower()
    self.color_distortion_model = str(color.get("distortion_model", "")).strip().lower()
    if self.depth_distortion_model != "brown_conrady":
      raise RuntimeError("depth distortion_model must be 'brown_conrady'")
    if self.color_distortion_model != "inverse_brown_conrady":
      raise RuntimeError("color distortion_model must be 'inverse_brown_conrady'")
    depth_distortion = _finite_array(
        depth.get("distortion"), (5,), "depth distortion"
    )
    color_distortion = _finite_array(
        color.get("distortion"), (5,), "color distortion"
    )
    if np.any(np.abs(depth_distortion) > 1e-12) or np.any(
        np.abs(color_distortion) > 1e-12
    ):
      requirement = "require_zero_distortion=true" if require_zero_distortion else "this version"
      raise RuntimeError(
          f"Non-zero depth/color distortion is unsupported by {requirement}; "
          "export a rectified zero-distortion profile or add an explicit rectification stage"
      )

    rotation = _finite_array(
        extrinsics.get("rotation"), (3, 3), "depth-to-color rotation"
    )
    translation = _finite_array(
        extrinsics.get("translation_m"), (3,), "depth-to-color translation_m"
    )
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-3 or abs(determinant - 1.0) > 1e-3:
      raise RuntimeError(
          "Depth-to-color rotation is not a valid rotation matrix: "
          f"orthogonality_error={orthogonality_error:.3g}, det={determinant:.9g}"
      )
    self.Rt = np.eye(4, dtype=np.float64)
    self.Rt[:3, :3] = rotation
    self.Rt[:3, 3] = translation

    self.camera_model = str(config.get("camera_model", "unknown"))
    self.serial = str(config.get("serial", "unknown"))
    self.firmware = str(config.get("firmware", "unknown"))
    self.calibration_profile = (
        calibration_width,
        calibration_height,
        calibration_fps,
    )

    diagnostics = _require_mapping(
        config.get("diagnostics", {}),
        "camera.mros.depth_to_color_alignment.diagnostics",
    )
    diagnostics_enabled = diagnostics.get("enabled", False)
    if not isinstance(diagnostics_enabled, bool):
      raise RuntimeError("Depth-to-color diagnostics.enabled must be true or false")
    self.diagnostics_enabled = diagnostics_enabled
    self.diagnostics_log_interval_sec = float(diagnostics.get("log_interval_sec", 5.0))
    self.expected_fps = float(diagnostics.get("expected_fps", calibration_fps))
    self.fps_tolerance = float(diagnostics.get("fps_tolerance", 2.0))
    if (
        not math.isfinite(self.diagnostics_log_interval_sec)
        or self.diagnostics_log_interval_sec <= 0.0
        or not math.isfinite(self.expected_fps)
        or self.expected_fps <= 0.0
        or not math.isfinite(self.fps_tolerance)
        or self.fps_tolerance < 0.0
    ):
      raise RuntimeError("Depth-to-color diagnostic timing values are invalid")

    self._cv2 = None
    if self.enabled:
      try:
        import cv2
      except ImportError as exc:
        raise RuntimeError("OpenCV is required for local depth-to-color alignment") from exc
      if not hasattr(cv2, "rgbd") or not hasattr(cv2.rgbd, "registerDepth"):
        raise RuntimeError("OpenCV was built without cv2.rgbd.registerDepth")
      self._cv2 = cv2

  @staticmethod
  def _validate_intrinsics(matrix: np.ndarray, label: str) -> np.ndarray:
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
      raise RuntimeError(f"{label} calibration focal lengths must be positive")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9):
      raise RuntimeError(f"{label} calibration K must use a pinhole final row [0, 0, 1]")
    return matrix

  def register(self, depth: np.ndarray, depth_scale: float) -> np.ndarray:
    if depth.shape != (self.height, self.width):
      raise RuntimeError(
          f"mROS raw-depth shape {depth.shape} does not match configured "
          f"{(self.height, self.width)}"
      )
    if not self.enabled:
      return _depth_native_to_meters(depth, depth_scale)
    registered = self._cv2.rgbd.registerDepth(
        self.depth_K,
        self.color_K,
        None,
        self.Rt,
        depth,
        (self.width, self.height),
        depthDilation=self.depth_dilation,
    )
    return _depth_native_to_meters(registered, depth_scale)


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
    self.last_match_delta_sec = None
    self.matched_pair_count = 0
    self.dropped_color_count = 0
    self.dropped_depth_count = 0

  def add_color(self, message) -> None:
    if len(self._colors) == self.queue_size:
      self._colors.popleft()
      self.dropped_color_count += 1
    self._colors.append(self._timed(message, "Color"))

  def add_depth(self, message) -> None:
    if len(self._depths) == self.queue_size:
      self._depths.popleft()
      self.dropped_depth_count += 1
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
    self.last_match_delta_sec = float(delta)
    self.matched_pair_count += 1
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
      self.dropped_color_count += 1
    elif oldest_depth < oldest_color - self.slop_sec:
      self._depths.popleft()
      self.dropped_depth_count += 1

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
      camera_info_topic: str | None,
      width: int = 640,
      height: int = 480,
      depth_min: float = 0.001,
      depth_max: float = 3.0,
      depth_scale: float = 0.001,
      sync_queue_size: int = 10,
      sync_slop_sec: float = 0.01,
      frame_timeout_sec: float = 5.0,
      disconnect_timeout_sec: float = 3.0,
      node_name: str = "foundationpose_rgbd_reader",
      reliable: bool = False,
      alignment_config: dict | None = None,
  ):
    self.color_topic = str(color_topic)
    self.depth_topic = str(depth_topic)
    self.camera_info_topic = None if camera_info_topic is None else str(camera_info_topic)
    self.width = int(width)
    self.height = int(height)
    self.depth_min = float(depth_min)
    self.depth_max = float(depth_max)
    self.depth_scale = float(depth_scale)
    self.sync_queue_size = int(sync_queue_size)
    self.sync_slop_sec = float(sync_slop_sec)
    self.frame_timeout_sec = float(frame_timeout_sec)
    self.disconnect_timeout_sec = float(disconnect_timeout_sec)
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
    if self.disconnect_timeout_sec <= 0.0 or not math.isfinite(self.disconnect_timeout_sec):
      raise ValueError("disconnect_timeout_sec must be positive and finite")
    if not self.node_name:
      raise ValueError("node_name must not be empty")

    self.alignment = (
        None
        if alignment_config is None
        else DepthToColorAlignment(alignment_config, self.width, self.height)
    )
    if self.alignment is None and not self.camera_info_topic:
      raise ValueError(
          "camera_info_topic is required when static depth-to-color calibration is absent"
      )

    self._synchronizer = ApproximateImageSynchronizer(
        queue_size=self.sync_queue_size,
        slop_sec=self.sync_slop_sec,
    )
    self._message_condition = threading.Condition()
    self._mros = None
    self._color_subscriber = None
    self._depth_subscriber = None
    self._camera_info_subscriber = None
    self._camera_info = None
    self._started = False
    self._stopping = False
    self._last_error_log_time = 0.0
    self._last_timeout_log_time = 0.0
    self._last_synchronized_frame_monotonic = None
    self._diagnostic_samples = {
        "sync_delta_ms": deque(maxlen=300),
        "color_decode_ms": deque(maxlen=300),
        "depth_decode_ms": deque(maxlen=300),
        "register_depth_ms": deque(maxlen=300),
        "depth_convert_filter_ms": deque(maxlen=300),
        "reader_total_ms": deque(maxlen=300),
        "source_interval_sec": deque(maxlen=300),
        "invalid_depth_ratio": deque(maxlen=300),
    }
    self._last_source_timestamp = None
    self._last_diagnostic_log_time = 0.0

  @classmethod
  def from_camera_config(cls, camera_config: dict) -> "MrosRgbdReader":
    camera_config = _require_mapping(camera_config, "camera")
    mros_config = _require_mapping(camera_config.get("mros"), "camera.mros")
    alignment_config = mros_config.get("depth_to_color_alignment")
    return cls(
        color_topic=_require_topic(mros_config, "color_topic"),
        depth_topic=_require_topic(mros_config, "depth_topic"),
        camera_info_topic=_optional_topic(mros_config, "camera_info_topic"),
        width=int(camera_config.get("width", 640)),
        height=int(camera_config.get("height", 480)),
        depth_min=float(camera_config.get("depth_min", 0.001)),
        depth_max=float(camera_config.get("depth_max", 3.0)),
        depth_scale=float(mros_config.get("depth_scale", 0.001)),
        sync_queue_size=int(mros_config.get("sync_queue_size", 10)),
        sync_slop_sec=float(mros_config.get("sync_slop_sec", 0.01)),
        frame_timeout_sec=float(mros_config.get("frame_timeout_sec", 5.0)),
        disconnect_timeout_sec=float(mros_config.get("disconnect_timeout_sec", 3.0)),
        node_name=str(mros_config.get("reader_node_name", "foundationpose_rgbd_reader")),
        reliable=bool(mros_config.get("reliable", False)),
        alignment_config=alignment_config,
    )

  def start(self) -> None:
    if self._started:
      raise RuntimeError("MrosRgbdReader.start() was called more than once")
    try:
      import mros
      from mros.sensor_msgs.msg import Image
    except ImportError as exc:
      raise RuntimeError(
          "mROS camera input is enabled, but the mROS Python package is unavailable"
      ) from exc

    with self._message_condition:
      self._synchronizer = ApproximateImageSynchronizer(
        queue_size=self.sync_queue_size,
        slop_sec=self.sync_slop_sec,
      )
      self._camera_info = None
      self._stopping = False
      self._last_synchronized_frame_monotonic = None
      self._last_source_timestamp = None
      self._last_diagnostic_log_time = time.monotonic()
      for samples in self._diagnostic_samples.values():
        samples.clear()

    mros.init(self.node_name)
    try:
      self._color_subscriber = mros.subscribe(
        self.color_topic,
        Image,
        self._handle_color_message,
        self.sync_queue_size,
        False,
        self.reliable,
      )
      self._depth_subscriber = mros.subscribe(
        self.depth_topic,
        Image,
        self._handle_depth_message,
        self.sync_queue_size,
        False,
        self.reliable,
      )
      if self.alignment is None:
        from mros.sensor_msgs.msg import CameraInfo
        self._camera_info_subscriber = mros.subscribe(
            self.camera_info_topic,
            CameraInfo,
            self._handle_camera_info_message,
            1,
            True,
            self.reliable,
        )
    except Exception:
      mros.shutdown()
      raise

    self._mros = mros
    self._started = True
    print("[mROS RGB-D] Subscriptions started")
    print(f"  color: {self.color_topic}")
    print(f"  raw depth: {self.depth_topic}")
    print(f"  camera info: {self.camera_info_topic or 'disabled (static calibration)'}")
    print(f"  expected size: {self.width}x{self.height}")
    print(f"  sync queue/slop: {self.sync_queue_size}/{self.sync_slop_sec:.3f} s")
    if self.alignment is not None:
      width, height, fps = self.alignment.calibration_profile
      print(
        f"  calibration: {self.alignment.camera_model}, serial={self.alignment.serial}, "
        f"firmware={self.alignment.firmware}, profile={width}x{height}@{fps}"
      )
      print(
        f"  depth-to-color alignment: {'enabled' if self.alignment.enabled else 'DISABLED'} "
        f"(backend={self.alignment.backend}, depth_dilation={self.alignment.depth_dilation})"
      )
      if not self.alignment.enabled:
        print("[WARNING] depth_to_color_alignment is disabled")
        print("[WARNING] Depth is NOT registered to color")
        print("[WARNING] RGB-D geometry is invalid for FoundationPose")

  def stop(self) -> None:
    if not self._started:
      return
    with self._message_condition:
      self._stopping = True
      self._message_condition.notify_all()
    try:
      if self._mros is not None:
        self._mros.shutdown()
    finally:
      self._color_subscriber = None
      self._depth_subscriber = None
      self._camera_info_subscriber = None
      self._mros = None
      self._started = False
      with self._message_condition:
        self._camera_info = None
        self._last_synchronized_frame_monotonic = None

  def get_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if not self._started:
      raise RuntimeError("MrosRgbdReader.get_frame() called before start()")
    deadline = time.monotonic() + self.frame_timeout_sec
    while True:
      if self._mros is None or not self._mros.ok():
        raise RuntimeError("mROS stopped while waiting for RGB-D input")
      with self._message_condition:
        if self._stopping:
          return None
        camera_info = self._camera_info
        calibration_ready = self.alignment is not None or camera_info is not None
        match = self._synchronizer.pop_match() if calibration_ready else None
        if match is None:
          remaining = deadline - time.monotonic()
          if remaining > 0.0:
            self._message_condition.wait(timeout=remaining)
            continue

      if match is not None:
        try:
          frame = self._convert_frame(match[0], match[1], camera_info)
          self._last_synchronized_frame_monotonic = time.monotonic()
          return frame
        except Exception as exc:
          self._log_error(exc)
          continue

      self._raise_if_disconnected()
      self._log_timeout()
      return None

  def _handle_color_message(self, message) -> None:
    try:
      with self._message_condition:
        if self._stopping:
          return
        self._synchronizer.add_color(message)
        self._message_condition.notify_all()
    except Exception as exc:
      self._log_error(exc)

  def _handle_depth_message(self, message) -> None:
    try:
      with self._message_condition:
        if self._stopping:
          return
        self._synchronizer.add_depth(message)
        self._message_condition.notify_all()
    except Exception as exc:
      self._log_error(exc)

  def _handle_camera_info_message(self, message) -> None:
    try:
      self._validate_camera_info_size(message)
      with self._message_condition:
        if self._stopping:
          return
        self._camera_info = message
        self._message_condition.notify_all()
    except Exception as exc:
      self._log_error(exc)

  def _convert_frame(self, color_message, depth_message, camera_info=None):
    total_start = time.perf_counter()
    color_decode_start = time.perf_counter()
    color = _color_image_to_rgb(color_message)
    color_decode_ms = (time.perf_counter() - color_decode_start) * 1000.0
    depth_decode_start = time.perf_counter()
    depth_native = _depth_image_to_native(depth_message)
    depth_decode_ms = (time.perf_counter() - depth_decode_start) * 1000.0
    if color.shape != (self.height, self.width, 3):
      raise RuntimeError(
          f"mROS color shape {color.shape} does not match configured {(self.height, self.width, 3)}"
      )
    if depth_native.shape != (self.height, self.width):
      raise RuntimeError(
          f"mROS raw-depth shape {depth_native.shape} does not match configured "
          f"{(self.height, self.width)}"
      )

    registration_start = time.perf_counter()
    if self.alignment is not None:
      depth = self.alignment.register(depth_native, self.depth_scale)
      K = np.ascontiguousarray(self.alignment.color_K, dtype=np.float32)
    else:
      depth = _depth_native_to_meters(depth_native, self.depth_scale)
      if camera_info is None:
        raise RuntimeError("CameraInfo is unavailable and no static calibration was configured")
      K = _camera_matrix(camera_info)
    register_depth_ms = (time.perf_counter() - registration_start) * 1000.0

    filter_start = time.perf_counter()
    depth[~np.isfinite(depth)] = 0.0
    depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    depth_convert_filter_ms = (time.perf_counter() - filter_start) * 1000.0
    timestamp = _stamp_seconds(color_message)
    self._record_diagnostics(
        color_timestamp=timestamp,
        depth_timestamp=_stamp_seconds(depth_message),
        color_decode_ms=color_decode_ms,
        depth_decode_ms=depth_decode_ms,
        register_depth_ms=register_depth_ms,
        depth_convert_filter_ms=depth_convert_filter_ms,
        reader_total_ms=(time.perf_counter() - total_start) * 1000.0,
        invalid_depth_ratio=float(np.count_nonzero(depth <= 0.0) / depth.size),
    )
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
        "check publishers, topic names, reliability, calibration, and RGB/depth timestamps"
    )

  def _record_diagnostics(
      self,
      color_timestamp: float,
      depth_timestamp: float,
      color_decode_ms: float,
      depth_decode_ms: float,
      register_depth_ms: float,
      depth_convert_filter_ms: float,
      reader_total_ms: float,
      invalid_depth_ratio: float,
  ) -> None:
    alignment = self.alignment
    if alignment is None or not alignment.diagnostics_enabled:
      return
    values = {
        "sync_delta_ms": abs(color_timestamp - depth_timestamp) * 1000.0,
        "color_decode_ms": color_decode_ms,
        "depth_decode_ms": depth_decode_ms,
        "register_depth_ms": register_depth_ms,
        "depth_convert_filter_ms": depth_convert_filter_ms,
        "reader_total_ms": reader_total_ms,
        "invalid_depth_ratio": invalid_depth_ratio,
    }
    for name, value in values.items():
      self._diagnostic_samples[name].append(float(value))
    if self._last_source_timestamp is not None:
      interval = color_timestamp - self._last_source_timestamp
      if interval > 0.0 and math.isfinite(interval):
        self._diagnostic_samples["source_interval_sec"].append(float(interval))
    self._last_source_timestamp = color_timestamp

    now = time.monotonic()
    if now - self._last_diagnostic_log_time < alignment.diagnostics_log_interval_sec:
      return
    self._last_diagnostic_log_time = now
    self._log_diagnostics()

  def _log_diagnostics(self) -> None:
    def summary(name: str) -> tuple[float, float, float]:
      values = np.asarray(self._diagnostic_samples[name], dtype=np.float64)
      if values.size == 0:
        return float("nan"), float("nan"), float("nan")
      return (
          float(np.percentile(values, 50.0)),
          float(np.percentile(values, 95.0)),
          float(np.max(values)),
      )

    sync_p50, sync_p95, sync_max = summary("sync_delta_ms")
    register_p50, register_p95, register_max = summary("register_depth_ms")
    total_p50, total_p95, total_max = summary("reader_total_ms")
    _, color_decode_p95, _ = summary("color_decode_ms")
    _, depth_decode_p95, _ = summary("depth_decode_ms")
    _, depth_filter_p95, _ = summary("depth_convert_filter_ms")
    invalid_p50, invalid_p95, _ = summary("invalid_depth_ratio")
    intervals = np.asarray(
        self._diagnostic_samples["source_interval_sec"], dtype=np.float64
    )
    source_fps = (
        float(1.0 / np.median(intervals)) if intervals.size else float("nan")
    )
    alignment = self.alignment
    fps_status = (
        "OK"
        if math.isfinite(source_fps)
        and abs(source_fps - alignment.expected_fps) <= alignment.fps_tolerance
        else "WARN"
    )
    print(
        "[mROS RGB-D DIAG] "
        f"alignment={'ON' if alignment.enabled else 'OFF'} geometry_valid={alignment.enabled} "
        f"fps={source_fps:.2f}/{alignment.expected_fps:.2f}({fps_status}) "
        f"sync_ms[p50/p95/max]={sync_p50:.3f}/{sync_p95:.3f}/{sync_max:.3f} "
        f"register_ms[p50/p95/max]={register_p50:.3f}/{register_p95:.3f}/{register_max:.3f} "
        f"reader_ms[p50/p95/max]={total_p50:.3f}/{total_p95:.3f}/{total_max:.3f} "
        f"stage_p95_ms=color_decode:{color_decode_p95:.3f},"
        f"depth_decode:{depth_decode_p95:.3f},depth_filter:{depth_filter_p95:.3f} "
        f"invalid_depth[p50/p95]={invalid_p50:.3f}/{invalid_p95:.3f} "
        f"pairs={self._synchronizer.matched_pair_count} "
        f"dropped=color:{self._synchronizer.dropped_color_count},"
        f"depth:{self._synchronizer.dropped_depth_count}"
    )

  def _raise_if_disconnected(self) -> None:
    last_frame_time = self._last_synchronized_frame_monotonic
    if last_frame_time is None:
      return
    elapsed = time.monotonic() - last_frame_time
    if elapsed < self.disconnect_timeout_sec:
      return
    raise TimeoutError(
        f"mROS RGB-D stream disconnected: no synchronized frame for {elapsed:.1f} s "
        f"after streaming started (limit={self.disconnect_timeout_sec:.1f} s); "
        "check publisher health, subscriber counts, topic names, reliability, "
        "calibration, and RGB/depth timestamps"
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
