from __future__ import annotations

from collections.abc import Mapping
import math
import threading
import time

import numpy as np


FOUNDATIONPOSE_STATUS_STATES = frozenset(("SEARCHING", "TRACKING", "LOST"))


class MrosPosePublisher:
  """Publish FoundationPose outputs with mROS standard messages."""

  def __init__(self, config: Mapping | None = None):
    config = {} if config is None else config
    visualization_config = config.get("visualization", {})
    if not isinstance(visualization_config, Mapping):
      raise ValueError("pose_mros.visualization must be a mapping")
    wheelarm_config = config.get("wheelarm_target", {})
    if not isinstance(wheelarm_config, Mapping):
      raise ValueError("pose_mros.wheelarm_target must be a mapping")

    self.enabled = bool(config.get("enabled", False))
    self.node_name = str(config.get("node_name", "foundationpose_pose_publisher"))
    self.pose_topic = str(config.get("pose_topic", "/foundationpose/object_pose"))
    self.status_topic = str(config.get("status_topic", "/foundationpose/status"))
    self.frame_id = str(config.get("frame_id", "camera_color_optical_frame"))
    self.child_frame_id = str(config.get("child_frame_id", "detected_object"))
    self.publish_tf = bool(config.get("publish_tf", False))
    self.publish_status_enabled = bool(config.get("publish_status", True))
    self.pose_queue_size = _positive_queue_size(config.get("queue_size", 1), "queue_size")
    self.status_queue_size = _positive_queue_size(
        config.get("status_queue_size", 1),
        "status_queue_size",
    )

    self.visualization_enabled = self.enabled and bool(
        visualization_config.get("enabled", False)
    )
    self.visualization_topic = str(
        visualization_config.get("topic", "/foundationpose/visualization")
    )
    self.visualization_frame_id = str(
        visualization_config.get("frame_id", self.frame_id)
    )
    self.visualization_publish_rate_hz = float(
        visualization_config.get("publish_rate_hz", 0.0)
    )
    self.visualization_only_with_subscribers = bool(
        visualization_config.get("only_with_subscribers", True)
    )
    self.visualization_queue_size = _positive_queue_size(
        visualization_config.get("queue_size", 1),
        "visualization.queue_size",
    )

    self.wheelarm_enabled = bool(wheelarm_config.get("enabled", False))
    self.wheelarm_topic = str(wheelarm_config.get("topic", "/wheelarm/target"))
    self.wheelarm_base_pose_topic = str(
        wheelarm_config.get("base_pose_topic", "/foundationpose/object_pose_base")
    )
    self.wheelarm_source_frame = str(
        wheelarm_config.get("source_frame", "camera_color_optical_frame")
    )
    self.wheelarm_target_frame = str(wheelarm_config.get("target_frame", "base_Link"))
    self.wheelarm_publish_static_tf = bool(wheelarm_config.get("publish_static_tf", False))
    self.wheelarm_publish_rate_hz = float(wheelarm_config.get("publish_rate_hz", 20.0))
    self.wheelarm_stale_timeout_sec = float(wheelarm_config.get("stale_timeout_sec", 1.0))
    self.wheelarm_override_position_z = bool(
      wheelarm_config.get("override_position_z", False)
    )
    self.wheelarm_fixed_position_z_m = float(
      wheelarm_config.get("fixed_position_z_m", 0.0)
    )
    if not math.isfinite(self.wheelarm_fixed_position_z_m):
      raise ValueError(
        "pose_mros.wheelarm_target.fixed_position_z_m must be finite"
      )
    self.wheelarm_override_orientation = bool(
        wheelarm_config.get("override_orientation", False)
    )
    fixed_orientation_wxyz = np.asarray(
        wheelarm_config.get("fixed_orientation_wxyz", (0.7, 0.0, 0.7, 0.0)),
        dtype=np.float64,
    )
    if fixed_orientation_wxyz.size != 4:
      raise ValueError(
          "pose_mros.wheelarm_target.fixed_orientation_wxyz must contain 4 values"
      )
    fixed_orientation_wxyz = fixed_orientation_wxyz.reshape(4)
    if not np.isfinite(fixed_orientation_wxyz).all():
      raise ValueError(
          "pose_mros.wheelarm_target.fixed_orientation_wxyz contains NaN or Inf"
      )
    self.wheelarm_fixed_orientation_wxyz = tuple(
        float(value) for value in fixed_orientation_wxyz
    )
    self.wheelarm_queue_size = _positive_queue_size(
        wheelarm_config.get("queue_size", 1),
        "wheelarm_target.queue_size",
    )
    base_from_camera = wheelarm_config.get("base_from_camera", np.eye(4))
    self.wheelarm_base_from_camera = _validate_pose_matrix(base_from_camera).copy()

    if self.visualization_publish_rate_hz < 0.0 or not math.isfinite(
        self.visualization_publish_rate_hz
    ):
      raise ValueError(
          "pose_mros.visualization.publish_rate_hz must be zero or positive and finite"
      )
    if self.wheelarm_publish_rate_hz <= 0.0 or not math.isfinite(self.wheelarm_publish_rate_hz):
      raise ValueError("pose_mros.wheelarm_target.publish_rate_hz must be positive and finite")
    if self.wheelarm_stale_timeout_sec <= 0.0 or not math.isfinite(self.wheelarm_stale_timeout_sec):
      raise ValueError("pose_mros.wheelarm_target.stale_timeout_sec must be positive and finite")
    if (
        self.wheelarm_enabled
        and self.wheelarm_publish_static_tf
        and self.wheelarm_source_frame == self.wheelarm_target_frame
    ):
      raise ValueError("pose_mros.wheelarm_target source_frame and target_frame must differ")
    for name, value in (
        ("node_name", self.node_name),
        ("pose_topic", self.pose_topic),
        ("status_topic", self.status_topic),
        ("frame_id", self.frame_id),
        ("child_frame_id", self.child_frame_id),
        ("visualization.topic", self.visualization_topic),
        ("visualization.frame_id", self.visualization_frame_id),
        ("wheelarm_target.topic", self.wheelarm_topic),
        ("wheelarm_target.base_pose_topic", self.wheelarm_base_pose_topic),
        ("wheelarm_target.source_frame", self.wheelarm_source_frame),
        ("wheelarm_target.target_frame", self.wheelarm_target_frame),
    ):
      if not value:
        raise ValueError(f"pose_mros.{name} must not be empty")

    self._mros = None
    self._started = False
    self._pose_publisher = None
    self._status_publisher = None
    self._visualization_publisher = None
    self._tf_broadcaster = None
    self._wheelarm_static_tf_broadcaster = None
    self._PoseStamped = None
    self._String = None
    self._Image = None
    self._TransformStamped = None
    self._Float32MultiArray = None
    self._last_status = None
    self._wheelarm_publisher = None
    self._wheelarm_base_pose_publisher = None
    self._wheelarm_lock = threading.Lock()
    self._wheelarm_stop_event = threading.Event()
    self._wheelarm_thread = None
    self._latest_wheelarm_data = None
    self._latest_wheelarm_update_monotonic = None
    self._latest_wheelarm_quaternion_xyzw = None
    self._wheelarm_stale_reported = False
    self._last_visualization_publish_monotonic = None
    self._pose_sequence = 0
    self._visualization_sequence = 0
    self._wheelarm_base_pose_sequence = 0

  def start(self) -> None:
    if not self.enabled:
      return
    if self._started:
      raise RuntimeError("MrosPosePublisher.start() was called more than once")

    try:
      import mros
      from mros.geometry_msgs.msg import PoseStamped, TransformStamped
      from mros.std_msgs.msg import Float32MultiArray, String
      if self.visualization_enabled:
        from mros.sensor_msgs.msg import Image
      else:
        Image = None
      if self.publish_tf:
        from mros.tf import TransformBroadcaster
      else:
        TransformBroadcaster = None
      if self.wheelarm_enabled and self.wheelarm_publish_static_tf:
        from mros.tf import StaticTransformBroadcaster
      else:
        StaticTransformBroadcaster = None
    except ImportError as exc:
      raise RuntimeError(
          "mROS pose publishing is enabled, but the mROS Python package is unavailable. "
          "Run inside the FoundationPose Jetson container or set pose_mros.enabled=false."
      ) from exc

    mros.init(self.node_name)
    try:
      pose_publisher = mros.advertise(
          self.pose_topic,
          PoseStamped,
          False,
          self.pose_queue_size,
      )
      status_publisher = None
      if self.publish_status_enabled:
        status_publisher = mros.advertise(
            self.status_topic,
            String,
            False,
            self.status_queue_size,
        )
      visualization_publisher = None
      if self.visualization_enabled:
        visualization_publisher = mros.advertise(
            self.visualization_topic,
            Image,
            False,
            self.visualization_queue_size,
        )
      wheelarm_publisher = None
      wheelarm_base_pose_publisher = None
      if self.wheelarm_enabled:
        wheelarm_publisher = mros.advertise(
            self.wheelarm_topic,
            Float32MultiArray,
            False,
            self.wheelarm_queue_size,
        )
        wheelarm_base_pose_publisher = mros.advertise(
            self.wheelarm_base_pose_topic,
            PoseStamped,
            False,
            self.pose_queue_size,
        )
      tf_broadcaster = TransformBroadcaster() if TransformBroadcaster is not None else None
      wheelarm_static_tf_broadcaster = (
          StaticTransformBroadcaster() if StaticTransformBroadcaster is not None else None
      )
    except Exception:
      mros.shutdown()
      raise

    self._mros = mros
    self._pose_publisher = pose_publisher
    self._status_publisher = status_publisher
    self._visualization_publisher = visualization_publisher
    self._tf_broadcaster = tf_broadcaster
    self._wheelarm_static_tf_broadcaster = wheelarm_static_tf_broadcaster
    self._PoseStamped = PoseStamped
    self._String = String
    self._Image = Image
    self._TransformStamped = TransformStamped
    self._Float32MultiArray = Float32MultiArray
    self._wheelarm_publisher = wheelarm_publisher
    self._wheelarm_base_pose_publisher = wheelarm_base_pose_publisher
    self._started = True

    print("[mROS Pose] Publisher started")
    print(f"  node: {self.node_name}")
    print(f"  pose: {self.pose_topic} (queue={self.pose_queue_size})")
    if self.publish_status_enabled:
      print(f"  status: {self.status_topic} (queue={self.status_queue_size})")
    if self.visualization_enabled:
      rate_text = (
          "window cadence"
          if self.visualization_publish_rate_hz == 0.0
          else f"max {self.visualization_publish_rate_hz:g} Hz"
      )
      subscriber_text = (
          ", subscribers only" if self.visualization_only_with_subscribers else ""
      )
      print(
          f"  visualization: {self.visualization_topic} (rgb8, "
          f"queue={self.visualization_queue_size}, {rate_text}{subscriber_text})"
      )
    if self.publish_tf:
      print(f"  TF: {self.frame_id} -> {self.child_frame_id}")
    if self.wheelarm_enabled:
      print(
          f"  wheelarm: {self.wheelarm_topic} at {self.wheelarm_publish_rate_hz:g} Hz "
          f"({self.wheelarm_source_frame} -> {self.wheelarm_target_frame}, "
          f"stale={self.wheelarm_stale_timeout_sec:g}s)"
      )
      if self.wheelarm_override_orientation:
        print(
            "  wheelarm orientation override (wxyz): "
            f"{self.wheelarm_fixed_orientation_wxyz}"
        )
      if self.wheelarm_override_position_z:
        print(
            "  wheelarm position Z override (base_Link, meters): "
            f"{self.wheelarm_fixed_position_z_m:g}"
        )
      print(
          f"  wheelarm base pose: {self.wheelarm_base_pose_topic} "
          f"({self.wheelarm_target_frame})"
      )
      if self.wheelarm_publish_static_tf:
        self._publish_wheelarm_static_transform()
        print(f"  static TF: {self.wheelarm_target_frame} -> {self.wheelarm_source_frame}")
      self._wheelarm_stop_event.clear()
      self._wheelarm_thread = threading.Thread(
          target=self._wheelarm_publish_loop,
          name="wheelarm-target-publisher",
          daemon=True,
      )
      self._wheelarm_thread.start()

  def publish_pose(self, pose: np.ndarray, stamp=None, frame_id: str | None = None) -> None:
    if not self.enabled:
      return
    self._require_started()

    matrix = _validate_pose_matrix(pose)
    qx, qy, qz, qw = rotation_matrix_to_quaternion(matrix[:3, :3])
    stamp_message = self._resolve_stamp(stamp)
    parent_frame = self.frame_id if frame_id is None else str(frame_id)
    if not parent_frame:
      raise ValueError("Pose frame_id must not be empty")

    message = self._PoseStamped()
    message.header.seq = self._take_sequence("_pose_sequence")
    message.header.stamp = stamp_message
    message.header.frame_id = parent_frame
    message.pose.position.x = float(matrix[0, 3])
    message.pose.position.y = float(matrix[1, 3])
    message.pose.position.z = float(matrix[2, 3])
    message.pose.orientation.x = qx
    message.pose.orientation.y = qy
    message.pose.orientation.z = qz
    message.pose.orientation.w = qw
    self._pose_publisher.publish(message)

    if self._tf_broadcaster is not None:
      transform = self._TransformStamped()
      transform.header.stamp = stamp_message
      transform.header.frame_id = parent_frame
      transform.child_frame_id = self.child_frame_id
      transform.transform.translation.x = float(matrix[0, 3])
      transform.transform.translation.y = float(matrix[1, 3])
      transform.transform.translation.z = float(matrix[2, 3])
      transform.transform.rotation.x = qx
      transform.transform.rotation.y = qy
      transform.transform.rotation.z = qz
      transform.transform.rotation.w = qw
      self._tf_broadcaster.sendTransform([transform])

  def visualization_due(self) -> bool:
    """Return whether the realtime-window image should be built for mROS now."""
    publisher = self._visualization_publisher
    if not self.visualization_enabled or publisher is None:
      return False
    if self.visualization_only_with_subscribers and publisher.getNumSubscribers() <= 0:
      return False
    if self.visualization_publish_rate_hz == 0.0:
      return True
    last_publish = self._last_visualization_publish_monotonic
    if last_publish is None:
      return True
    return time.monotonic() - last_publish >= 1.0 / self.visualization_publish_rate_hz

  def publish_visualization(
      self,
      image_rgb: np.ndarray,
      timestamp_seconds: float | None = None,
  ) -> bool:
    """Publish the same RGB image used by the realtime OpenCV window."""
    if not self.visualization_due():
      return False

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
      raise ValueError(f"Visualization image must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
      raise ValueError(f"Visualization image must use uint8, got {image.dtype}")
    image = np.ascontiguousarray(image)

    message = self._Image()
    message.header.seq = self._take_sequence("_visualization_sequence")
    message.header.stamp = self._resolve_stamp(timestamp_seconds)
    message.header.frame_id = self.visualization_frame_id
    message.height = int(image.shape[0])
    message.width = int(image.shape[1])
    message.encoding = "rgb8"
    message.is_bigendian = 0
    message.step = int(image.shape[1] * 3)
    message.data = image.tobytes()
    self._visualization_publisher.publish(message)
    self._last_visualization_publish_monotonic = time.monotonic()
    return True

  def _publish_wheelarm_static_transform(self) -> None:
    broadcaster = self._wheelarm_static_tf_broadcaster
    message_type = self._TransformStamped
    if broadcaster is None or message_type is None:
      return

    matrix = self.wheelarm_base_from_camera
    qx, qy, qz, qw = rotation_matrix_to_quaternion(matrix[:3, :3])
    transform = message_type()
    transform.header.stamp = self._resolve_stamp(None)
    transform.header.frame_id = self.wheelarm_target_frame
    transform.child_frame_id = self.wheelarm_source_frame
    transform.transform.translation.x = float(matrix[0, 3])
    transform.transform.translation.y = float(matrix[1, 3])
    transform.transform.translation.z = float(matrix[2, 3])
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    broadcaster.sendTransform([transform])

  def update_wheelarm_target(self, camera_pose: np.ndarray, stamp=None) -> list[float] | None:
    """Replace the target continuously published on the configured mROS topic."""
    if not self.enabled or not self.wheelarm_enabled:
      return None
    self._require_started()

    base_pose = compose_pose(self.wheelarm_base_from_camera, camera_pose)
    with self._wheelarm_lock:
      data, quaternion_xyzw = pose_matrix_to_wheelarm_target(
          base_pose,
          previous_quaternion_xyzw=self._latest_wheelarm_quaternion_xyzw,
      )
      if self.wheelarm_override_position_z:
        data[2] = self.wheelarm_fixed_position_z_m
      if self.wheelarm_override_orientation:
        qw, qx, qy, qz = self.wheelarm_fixed_orientation_wxyz
        data[3:] = (qw, qx, qy, qz)
        quaternion_xyzw = (qx, qy, qz, qw)
      self._latest_wheelarm_data = data
      self._latest_wheelarm_quaternion_xyzw = quaternion_xyzw
      self._latest_wheelarm_update_monotonic = time.monotonic()
      self._wheelarm_stale_reported = False
      self._publish_wheelarm_base_pose(base_pose, data, stamp=stamp)
      self._publish_wheelarm_data(data)
    return list(data)

  def _publish_wheelarm_base_pose(self, base_pose: np.ndarray, wheelarm_data, stamp=None) -> None:
    publisher = self._wheelarm_base_pose_publisher
    message_type = self._PoseStamped
    if publisher is None or message_type is None:
      return
    matrix = _validate_pose_matrix(base_pose)
    data = np.asarray(wheelarm_data, dtype=np.float64).reshape(7)
    if not np.isfinite(data).all():
      raise ValueError("WheelArm target contains NaN or Inf")
    _, _, _, qw, qx, qy, qz = (float(value) for value in data)
    message = message_type()
    message.header.seq = self._take_sequence("_wheelarm_base_pose_sequence")
    message.header.stamp = self._resolve_stamp(stamp)
    message.header.frame_id = self.wheelarm_target_frame
    message.pose.position.x = float(data[0])
    message.pose.position.y = float(data[1])
    message.pose.position.z = float(data[2])
    message.pose.orientation.x = qx
    message.pose.orientation.y = qy
    message.pose.orientation.z = qz
    message.pose.orientation.w = qw
    publisher.publish(message)

  def clear_wheelarm_target(self) -> None:
    with self._wheelarm_lock:
      self._latest_wheelarm_data = None
      self._latest_wheelarm_update_monotonic = None
      self._latest_wheelarm_quaternion_xyzw = None
      self._wheelarm_stale_reported = False

  def _wheelarm_publish_loop(self) -> None:
    period_sec = 1.0 / self.wheelarm_publish_rate_hz
    while not self._wheelarm_stop_event.wait(period_sec):
      expired = False
      with self._wheelarm_lock:
        updated_at = self._latest_wheelarm_update_monotonic
        if self._latest_wheelarm_data is not None and updated_at is not None:
          age_sec = time.monotonic() - updated_at
          if age_sec <= self.wheelarm_stale_timeout_sec:
            self._publish_wheelarm_data(self._latest_wheelarm_data)
          elif not self._wheelarm_stale_reported:
            self._wheelarm_stale_reported = True
            expired = True
      if expired:
        print(
            f"[WheelArm] Latest target exceeded {self.wheelarm_stale_timeout_sec:g}s; "
            "publishing paused until the next valid pose"
        )

  def _publish_wheelarm_data(self, data) -> None:
    publisher = self._wheelarm_publisher
    message_type = self._Float32MultiArray
    if publisher is None or message_type is None:
      return
    message = message_type()
    message.data = [float(np.float32(value)) for value in data]
    publisher.publish(message)

  def publish_status(self, status: str) -> None:
    if not self.enabled or not self.publish_status_enabled:
      return
    self._require_started()
    status_text = str(status).strip()
    if not status_text:
      raise ValueError("FoundationPose status must not be empty")
    state_text, separator, detail_text = status_text.partition(":")
    state = state_text.strip().upper()
    if state not in FOUNDATIONPOSE_STATUS_STATES:
      allowed = ", ".join(sorted(FOUNDATIONPOSE_STATUS_STATES))
      raise ValueError(f"FoundationPose status state must be one of: {allowed}")
    detail = detail_text.strip() if separator else ""
    if not detail:
      raise ValueError("FoundationPose status must include detail after ':'")
    normalized = f"{state}: {detail}"
    if normalized == self._last_status:
      return

    message = self._String()
    message.data = normalized
    self._status_publisher.publish(message)
    self._last_status = normalized

  def stop(self) -> None:
    if not self.enabled or not self._started:
      return
    self._wheelarm_stop_event.set()
    wheelarm_thread = self._wheelarm_thread
    self._wheelarm_thread = None
    if wheelarm_thread is not None and wheelarm_thread.is_alive():
      wheelarm_thread.join(timeout=2.0)
    self.clear_wheelarm_target()

    mros = self._mros
    self._started = False
    self._pose_publisher = None
    self._status_publisher = None
    self._visualization_publisher = None
    self._tf_broadcaster = None
    self._wheelarm_static_tf_broadcaster = None
    self._wheelarm_publisher = None
    self._wheelarm_base_pose_publisher = None
    self._last_visualization_publish_monotonic = None
    self._mros = None
    if mros is not None:
      mros.shutdown()

  def _resolve_stamp(self, stamp):
    if stamp is None:
      return self._mros.Time.now()
    if isinstance(stamp, (int, float, np.integer, np.floating)):
      timestamp = float(stamp)
      if timestamp < 0.0 or not math.isfinite(timestamp):
        raise ValueError("mROS timestamp must be non-negative and finite")
      seconds = math.floor(timestamp)
      nanoseconds = int(round((timestamp - seconds) * 1_000_000_000.0))
      if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
      return self._mros.Time(int(seconds), nanoseconds)
    if hasattr(stamp, "sec") and hasattr(stamp, "nsec"):
      return stamp
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
      return self._mros.Time(int(stamp.sec), int(stamp.nanosec))
    raise TypeError("stamp must be mros.Time, a non-negative numeric timestamp, or None")

  def _take_sequence(self, attribute: str) -> int:
    sequence = int(getattr(self, attribute))
    setattr(self, attribute, (sequence + 1) & 0xFFFFFFFF)
    return sequence

  def _require_started(self) -> None:
    if not self._started:
      raise RuntimeError("MrosPosePublisher.start() must be called before publishing")


def _positive_queue_size(value, name: str) -> int:
  if isinstance(value, bool):
    raise ValueError(f"pose_mros.{name} must be a positive integer")
  try:
    queue_size = int(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"pose_mros.{name} must be a positive integer") from exc
  if queue_size <= 0:
    raise ValueError(f"pose_mros.{name} must be a positive integer")
  return queue_size


def _validate_pose_matrix(pose: np.ndarray) -> np.ndarray:
  matrix = np.asarray(pose, dtype=np.float64)
  if matrix.size != 16:
    raise ValueError(f"Pose must contain 16 values, got shape {matrix.shape}")
  matrix = matrix.reshape(4, 4)
  if not np.isfinite(matrix).all():
    raise ValueError("Pose contains NaN or Inf")
  if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-4):
    raise ValueError(f"Pose has an invalid homogeneous last row: {matrix[3].tolist()}")

  rotation = matrix[:3, :3]
  if np.linalg.det(rotation) <= 0.0:
    raise ValueError("Pose rotation must have a positive determinant")
  if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-2):
    raise ValueError("Pose rotation is not sufficiently orthonormal")
  return matrix


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
  """Return a normalized (x, y, z, w) quaternion."""
  matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
  if not np.isfinite(matrix).all():
    raise ValueError("Rotation matrix contains NaN or Inf")

  u, _, vh = np.linalg.svd(matrix)
  projected = u @ vh
  if np.linalg.det(projected) < 0.0:
    u[:, -1] *= -1.0
    projected = u @ vh

  trace = float(np.trace(projected))
  if trace > 0.0:
    scale = math.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * scale
    qx = (projected[2, 1] - projected[1, 2]) / scale
    qy = (projected[0, 2] - projected[2, 0]) / scale
    qz = (projected[1, 0] - projected[0, 1]) / scale
  elif projected[0, 0] > projected[1, 1] and projected[0, 0] > projected[2, 2]:
    scale = math.sqrt(1.0 + projected[0, 0] - projected[1, 1] - projected[2, 2]) * 2.0
    qw = (projected[2, 1] - projected[1, 2]) / scale
    qx = 0.25 * scale
    qy = (projected[0, 1] + projected[1, 0]) / scale
    qz = (projected[0, 2] + projected[2, 0]) / scale
  elif projected[1, 1] > projected[2, 2]:
    scale = math.sqrt(1.0 + projected[1, 1] - projected[0, 0] - projected[2, 2]) * 2.0
    qw = (projected[0, 2] - projected[2, 0]) / scale
    qx = (projected[0, 1] + projected[1, 0]) / scale
    qy = 0.25 * scale
    qz = (projected[1, 2] + projected[2, 1]) / scale
  else:
    scale = math.sqrt(1.0 + projected[2, 2] - projected[0, 0] - projected[1, 1]) * 2.0
    qw = (projected[1, 0] - projected[0, 1]) / scale
    qx = (projected[0, 2] + projected[2, 0]) / scale
    qy = (projected[1, 2] + projected[2, 1]) / scale
    qz = 0.25 * scale

  quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
  norm = float(np.linalg.norm(quaternion))
  if norm <= 1e-12:
    raise ValueError("Rotation matrix produced a zero-length quaternion")
  quaternion /= norm
  if quaternion[3] < 0.0:
    quaternion *= -1.0
  return tuple(float(value) for value in quaternion)


def compose_pose(parent_from_child: np.ndarray, child_from_object: np.ndarray) -> np.ndarray:
  """Compose rigid transforms and return parent_from_object."""
  parent_from_child_matrix = _validate_pose_matrix(parent_from_child)
  child_from_object_matrix = _validate_pose_matrix(child_from_object)
  return _validate_pose_matrix(parent_from_child_matrix @ child_from_object_matrix)


def pose_matrix_to_wheelarm_target(
    base_pose: np.ndarray,
    previous_quaternion_xyzw=None,
) -> tuple[list[float], tuple[float, float, float, float]]:
  """Return [x, y, z, qw, qx, qy, qz] and the (x, y, z, w) quaternion."""
  matrix = _validate_pose_matrix(base_pose)
  quaternion = np.asarray(rotation_matrix_to_quaternion(matrix[:3, :3]), dtype=np.float64)
  if previous_quaternion_xyzw is not None:
    previous = np.asarray(previous_quaternion_xyzw, dtype=np.float64).reshape(4)
    if np.isfinite(previous).all() and float(np.dot(previous, quaternion)) < 0.0:
      quaternion *= -1.0
  qx, qy, qz, qw = (float(value) for value in quaternion)
  data = np.asarray(
      (matrix[0, 3], matrix[1, 3], matrix[2, 3], qw, qx, qy, qz),
      dtype=np.float32,
  )
  if not np.isfinite(data).all():
    raise ValueError("WheelArm target contains NaN or Inf")
  return [float(value) for value in data], (qx, qy, qz, qw)
