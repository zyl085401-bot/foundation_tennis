import math
from types import SimpleNamespace
import threading
import time
import unittest

import numpy as np

from realtime_foundation.mros_pose_publisher import (
    MrosPosePublisher,
    _validate_pose_matrix,
    compose_pose,
    pose_matrix_to_wheelarm_target,
    rotation_matrix_to_quaternion,
)


BASE_FROM_CAMERA = np.array([
    [0.0, -0.406749098, -0.913539912, 0.0968],
    [-1.0, 0.0, 0.0, 0.01759],
    [0.0, 0.913539912, -0.406749098, -0.00409],
    [0.0, 0.0, 0.0, 1.0],
])


class FakeTime:
  def __init__(self, sec=0, nsec=0):
    self.sec = sec
    self.nsec = nsec

  @staticmethod
  def now():
    return FakeTime(123, 456)


class FakeMros:
  Time = FakeTime


class FakeFloat32MultiArray:
  def __init__(self):
    self.data = []


class FakeString:
  def __init__(self):
    self.data = ""


class FakeHeader:
  def __init__(self):
    self.seq = 0
    self.stamp = None
    self.frame_id = ""


class FakePoseStamped:
  def __init__(self):
    self.header = FakeHeader()
    self.pose = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


class FakeImage:
  def __init__(self):
    self.header = FakeHeader()
    self.height = 0
    self.width = 0
    self.encoding = ""
    self.is_bigendian = 0
    self.step = 0
    self.data = b""


class FakeTransformStamped:
  def __init__(self):
    self.header = FakeHeader()
    self.child_frame_id = ""
    self.transform = SimpleNamespace(
        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


class FakePublisher:
  def __init__(self, subscriber_count=0):
    self.messages = []
    self.subscriber_count = subscriber_count

  def publish(self, message):
    self.messages.append(message)

  def getNumSubscribers(self):
    return self.subscriber_count


class BlockingFakePublisher(FakePublisher):
  def __init__(self):
    super().__init__()
    self.publish_started = threading.Event()
    self.publish_release = threading.Event()

  def publish(self, message):
    self.publish_started.set()
    self.publish_release.wait(timeout=1.0)
    super().publish(message)


class FakeTransformBroadcaster:
  def __init__(self):
    self.transform_batches = []

  def sendTransform(self, transforms):
    self.transform_batches.append(transforms)


class MrosPosePublisherTest(unittest.TestCase):
  @staticmethod
  def mark_started(publisher):
    publisher._mros = FakeMros
    publisher._started = True

  def test_identity_rotation_converts_to_identity_quaternion(self):
    quaternion = rotation_matrix_to_quaternion(np.eye(3))

    np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1e-7)

  def test_quarter_turn_about_z_uses_xyzw_quaternion_order(self):
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    quaternion = rotation_matrix_to_quaternion(rotation)

    expected = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    np.testing.assert_allclose(quaternion, expected, atol=1e-7)

  def test_pose_validation_accepts_flat_homogeneous_matrix(self):
    pose = np.eye(4)
    pose[:3, 3] = (0.1, -0.2, 0.3)

    validated = _validate_pose_matrix(pose.reshape(-1))

    self.assertEqual(validated.shape, (4, 4))
    np.testing.assert_allclose(validated[:3, 3], (0.1, -0.2, 0.3))

  def test_pose_validation_rejects_reflection(self):
    pose = np.eye(4)
    pose[0, 0] = -1.0

    with self.assertRaisesRegex(ValueError, "positive determinant"):
      _validate_pose_matrix(pose)

  def test_disabled_publisher_does_not_import_mros(self):
    publisher = MrosPosePublisher({"enabled": False})

    publisher.start()
    publisher.publish_status("TRACKING")
    publisher.publish_pose(np.eye(4))
    publisher.stop()

  def test_invalid_queue_size_is_rejected(self):
    for value in (0, -1, True, "invalid"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "queue_size"):
          MrosPosePublisher({"enabled": True, "queue_size": value})

  def test_status_preserves_detail_text_and_deduplicates_full_message(self):
    publisher = MrosPosePublisher({"enabled": True})
    self.mark_started(publisher)
    publisher._String = FakeString
    publisher._status_publisher = FakePublisher()

    publisher.publish_status("searching: wait yolo seg mask")
    publisher.publish_status("SEARCHING: wait yolo seg mask")
    publisher.publish_status("SEARCHING: confidence 0.400 < 0.700")

    self.assertEqual(
        [message.data for message in publisher._status_publisher.messages],
        [
            "SEARCHING: wait yolo seg mask",
            "SEARCHING: confidence 0.400 < 0.700",
        ],
    )

  def test_status_accepts_only_searching_tracking_and_lost(self):
    publisher = MrosPosePublisher({"enabled": True})
    self.mark_started(publisher)
    publisher._String = FakeString
    publisher._status_publisher = FakePublisher()

    publisher.publish_status("SEARCHING: wait target")
    publisher.publish_status("TRACKING: target visible")
    publisher.publish_status("LOST: target missing")

    self.assertEqual(
        [message.data for message in publisher._status_publisher.messages],
        ["SEARCHING: wait target", "TRACKING: target visible", "LOST: target missing"],
    )

  def test_status_requires_detail_text(self):
    publisher = MrosPosePublisher({"enabled": True})
    self.mark_started(publisher)
    publisher._String = FakeString
    publisher._status_publisher = FakePublisher()

    for status in ("SEARCHING", "TRACKING:", "LOST:   "):
      with self.subTest(status=status):
        with self.assertRaisesRegex(ValueError, "include detail"):
          publisher.publish_status(status)

  def test_static_tf_rejects_identical_parent_and_child_frames(self):
    with self.assertRaisesRegex(ValueError, "source_frame and target_frame must differ"):
      MrosPosePublisher({
          "enabled": True,
          "wheelarm_target": {
              "enabled": True,
              "publish_static_tf": True,
              "source_frame": "base_Link",
              "target_frame": "base_Link",
          },
      })

  def test_compose_pose_applies_base_from_camera_on_the_left(self):
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = (0.0, 0.0, 0.6)

    base_pose = compose_pose(BASE_FROM_CAMERA, camera_pose)

    np.testing.assert_allclose(
        base_pose[:3, 3],
        (-0.4513239472, 0.01759, -0.2481394588),
        atol=1e-8,
    )

  def test_static_tf_uses_base_as_parent_and_camera_as_child(self):
    publisher = MrosPosePublisher({
        "enabled": True,
        "wheelarm_target": {
            "enabled": True,
            "publish_static_tf": True,
            "source_frame": "camera_color_optical_frame",
            "target_frame": "base_Link",
            "base_from_camera": BASE_FROM_CAMERA,
        },
    })
    self.mark_started(publisher)
    publisher._TransformStamped = FakeTransformStamped
    publisher._wheelarm_static_tf_broadcaster = FakeTransformBroadcaster()

    publisher._publish_wheelarm_static_transform()

    self.assertEqual(len(publisher._wheelarm_static_tf_broadcaster.transform_batches), 1)
    transform = publisher._wheelarm_static_tf_broadcaster.transform_batches[0][0]
    self.assertEqual(transform.header.frame_id, "base_Link")
    self.assertEqual(transform.child_frame_id, "camera_color_optical_frame")
    np.testing.assert_allclose(
        (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ),
        BASE_FROM_CAMERA[:3, 3],
        atol=1e-8,
    )

  def test_publish_pose_builds_mros_pose_and_dynamic_tf(self):
    publisher = MrosPosePublisher({"enabled": True, "publish_tf": True})
    self.mark_started(publisher)
    publisher._PoseStamped = FakePoseStamped
    publisher._TransformStamped = FakeTransformStamped
    publisher._pose_publisher = FakePublisher()
    publisher._tf_broadcaster = FakeTransformBroadcaster()
    pose = np.eye(4)
    pose[:3, 3] = (0.1, -0.2, 0.3)

    publisher.publish_pose(pose, stamp=12.25)

    message = publisher._pose_publisher.messages[0]
    self.assertEqual(message.header.seq, 0)
    self.assertEqual((message.header.stamp.sec, message.header.stamp.nsec), (12, 250_000_000))
    self.assertEqual(message.header.frame_id, "camera_color_optical_frame")
    np.testing.assert_allclose(
        (message.pose.position.x, message.pose.position.y, message.pose.position.z),
        (0.1, -0.2, 0.3),
    )
    transform = publisher._tf_broadcaster.transform_batches[0][0]
    self.assertEqual(transform.child_frame_id, "detected_object")

  def test_visualization_uses_mros_subscriber_count_and_nsec_stamp(self):
    publisher = MrosPosePublisher({
        "enabled": True,
        "visualization": {"enabled": True, "only_with_subscribers": True},
    })
    self.mark_started(publisher)
    publisher._Image = FakeImage
    publisher._visualization_publisher = FakePublisher(subscriber_count=0)
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    self.assertFalse(publisher.publish_visualization(image, timestamp_seconds=5.5))
    publisher._visualization_publisher.subscriber_count = 1
    self.assertTrue(publisher.publish_visualization(image, timestamp_seconds=5.5))

    message = publisher._visualization_publisher.messages[0]
    self.assertEqual((message.header.stamp.sec, message.header.stamp.nsec), (5, 500_000_000))
    self.assertEqual((message.height, message.width, message.step), (2, 3, 9))
    self.assertEqual(message.encoding, "rgb8")

  def test_wheelarm_target_uses_wxyz_quaternion_order(self):
    pose = np.eye(4)
    pose[:3, 3] = (0.6, 0.2, 0.4)

    data, quaternion_xyzw = pose_matrix_to_wheelarm_target(pose)

    np.testing.assert_allclose(data, (0.6, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0), atol=1e-7)
    np.testing.assert_allclose(quaternion_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1e-7)

  def test_wheelarm_update_publishes_float32_data_and_base_pose(self):
    publisher = MrosPosePublisher({
        "enabled": True,
        "wheelarm_target": {
            "enabled": True,
            "target_frame": "base_Link",
            "base_from_camera": np.eye(4),
        },
    })
    self.mark_started(publisher)
    publisher._Float32MultiArray = FakeFloat32MultiArray
    publisher._PoseStamped = FakePoseStamped
    publisher._wheelarm_publisher = FakePublisher()
    publisher._wheelarm_base_pose_publisher = FakePublisher()
    pose = np.eye(4)
    pose[:3, 3] = (0.6, 0.2, 0.4)

    data = publisher.update_wheelarm_target(pose, stamp=FakeTime(123, 456))

    target_message = publisher._wheelarm_publisher.messages[0]
    self.assertEqual(target_message.data, [float(np.float32(value)) for value in data])
    pose_message = publisher._wheelarm_base_pose_publisher.messages[0]
    self.assertEqual(pose_message.header.frame_id, "base_Link")
    np.testing.assert_allclose(
        (pose_message.pose.position.x, pose_message.pose.position.y, pose_message.pose.position.z),
        data[:3],
        atol=1e-7,
    )

  def test_clear_wheelarm_target_stops_republishing_latest_pose(self):
    publisher = MrosPosePublisher({
        "enabled": True,
        "wheelarm_target": {
            "enabled": True,
        },
    })
    publisher._latest_wheelarm_data = [1.0] * 7
    publisher._latest_wheelarm_update_monotonic = 10.0
    publisher._latest_wheelarm_quaternion_xyzw = (0.0, 0.0, 0.0, 1.0)
    publisher._wheelarm_stale_reported = True

    publisher.clear_wheelarm_target()

    self.assertIsNone(publisher._latest_wheelarm_data)
    self.assertIsNone(publisher._latest_wheelarm_update_monotonic)
    self.assertIsNone(publisher._latest_wheelarm_quaternion_xyzw)
    self.assertFalse(publisher._wheelarm_stale_reported)

  def test_clear_waits_for_inflight_publish_and_prevents_later_stale_publish(self):
    publisher = MrosPosePublisher({
        "enabled": True,
        "wheelarm_target": {
            "enabled": True,
            "publish_rate_hz": 1000.0,
            "stale_timeout_sec": 10.0,
        },
    })
    publisher._Float32MultiArray = FakeFloat32MultiArray
    publisher._wheelarm_publisher = BlockingFakePublisher()
    publisher._latest_wheelarm_data = [1.0] * 7
    publisher._latest_wheelarm_update_monotonic = time.monotonic()
    publisher._wheelarm_stop_event.clear()
    publish_thread = threading.Thread(target=publisher._wheelarm_publish_loop)
    clear_finished = threading.Event()
    clear_thread = threading.Thread(
        target=lambda: (publisher.clear_wheelarm_target(), clear_finished.set())
    )
    publish_thread.start()
    try:
      self.assertTrue(publisher._wheelarm_publisher.publish_started.wait(timeout=1.0))
      clear_thread.start()
      self.assertFalse(clear_finished.wait(timeout=0.02))
      publisher._wheelarm_publisher.publish_release.set()
      self.assertTrue(clear_finished.wait(timeout=1.0))
      published_after_clear = len(publisher._wheelarm_publisher.messages)
      time.sleep(0.02)
      self.assertEqual(len(publisher._wheelarm_publisher.messages), published_after_clear)
    finally:
      publisher._wheelarm_publisher.publish_release.set()
      publisher._wheelarm_stop_event.set()
      publish_thread.join(timeout=1.0)
      if clear_thread.is_alive():
        clear_thread.join(timeout=1.0)


if __name__ == "__main__":
  unittest.main()
