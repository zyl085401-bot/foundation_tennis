import math
from types import SimpleNamespace
import unittest

import numpy as np

from realtime_foundation.ros2_pose_publisher import (
    Ros2PosePublisher,
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


class FakeFloat32MultiArray:
  def __init__(self):
    self.data = []


class FakeString:
  def __init__(self):
    self.data = ""


class FakePoseStamped:
  def __init__(self):
    self.header = SimpleNamespace(stamp=None, frame_id="")
    self.pose = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


class FakeTransformStamped:
  def __init__(self):
    self.header = SimpleNamespace(stamp=None, frame_id="")
    self.child_frame_id = ""
    self.transform = SimpleNamespace(
        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


class FakePublisher:
  def __init__(self):
    self.messages = []

  def publish(self, message):
    self.messages.append(message)


class FakeStaticTransformBroadcaster:
  def __init__(self):
    self.transforms = []

  def sendTransform(self, transform):
    self.transforms.append(transform)


class FakeNode:
  def get_clock(self):
    return self

  def now(self):
    return self

  def to_msg(self):
    return SimpleNamespace(sec=123, nanosec=456)


class Ros2PosePublisherTest(unittest.TestCase):
  def test_identity_rotation_converts_to_identity_quaternion(self):
    quaternion = rotation_matrix_to_quaternion(np.eye(3))

    np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1e-7)

  def test_quarter_turn_about_z_uses_ros_quaternion_order(self):
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

  def test_disabled_publisher_does_not_import_ros(self):
    publisher = Ros2PosePublisher({"enabled": False})

    publisher.start()
    publisher.publish_status("TRACKING")
    publisher.publish_pose(np.eye(4))
    publisher.stop()

  def test_invalid_reliability_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "reliability"):
      Ros2PosePublisher({
          "enabled": True,
          "qos": {"reliability": "sometimes"},
      })

  def test_status_preserves_detail_text_and_deduplicates_full_message(self):
    publisher = Ros2PosePublisher({"enabled": True})
    publisher._node = object()
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
    publisher = Ros2PosePublisher({"enabled": True})
    publisher._node = object()
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
    publisher = Ros2PosePublisher({"enabled": True})
    publisher._node = object()
    publisher._String = FakeString
    publisher._status_publisher = FakePublisher()

    for status in ("SEARCHING", "TRACKING:", "LOST:   "):
      with self.subTest(status=status):
        with self.assertRaisesRegex(ValueError, "include detail"):
          publisher.publish_status(status)

  def test_status_rejects_removed_states(self):
    publisher = Ros2PosePublisher({"enabled": True})
    publisher._node = object()
    publisher._String = FakeString
    publisher._status_publisher = FakePublisher()

    for state in ("WAITING_FOR_DETECTION", "REGISTERING", "ERROR", "STOPPED"):
      with self.subTest(state=state):
        with self.assertRaisesRegex(ValueError, "must be one of"):
          publisher.publish_status(state)

  def test_static_tf_rejects_identical_parent_and_child_frames(self):
    with self.assertRaisesRegex(ValueError, "source_frame and target_frame must differ"):
      Ros2PosePublisher({
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
    publisher = Ros2PosePublisher({
        "enabled": True,
        "wheelarm_target": {
            "enabled": True,
            "publish_static_tf": True,
            "source_frame": "camera_color_optical_frame",
            "target_frame": "base_Link",
            "base_from_camera": BASE_FROM_CAMERA,
        },
    })
    publisher._node = FakeNode()
    publisher._TransformStamped = FakeTransformStamped
    publisher._wheelarm_static_tf_broadcaster = FakeStaticTransformBroadcaster()

    publisher._publish_wheelarm_static_transform()

    self.assertEqual(len(publisher._wheelarm_static_tf_broadcaster.transforms), 1)
    transform = publisher._wheelarm_static_tf_broadcaster.transforms[0]
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
    np.testing.assert_allclose(
        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ),
        rotation_matrix_to_quaternion(BASE_FROM_CAMERA[:3, :3]),
        atol=1e-8,
    )

  def test_wheelarm_target_uses_wxyz_quaternion_order(self):
    pose = np.eye(4)
    pose[:3, 3] = (0.6, 0.2, 0.4)

    data, quaternion_xyzw = pose_matrix_to_wheelarm_target(pose)

    np.testing.assert_allclose(data, (0.6, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0), atol=1e-7)
    np.testing.assert_allclose(quaternion_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1e-7)

  def test_wheelarm_target_keeps_quaternion_sign_continuous(self):
    data, quaternion_xyzw = pose_matrix_to_wheelarm_target(
        np.eye(4),
        previous_quaternion_xyzw=(0.0, 0.0, 0.0, -1.0),
    )

    np.testing.assert_allclose(data[3:], (-1.0, 0.0, 0.0, 0.0), atol=1e-7)
    np.testing.assert_allclose(quaternion_xyzw, (0.0, 0.0, 0.0, -1.0), atol=1e-7)

  def test_wheelarm_update_publishes_float32_data_immediately(self):
    publisher = Ros2PosePublisher({
        "enabled": True,
        "wheelarm_target": {
            "enabled": True,
            "base_from_camera": np.eye(4),
        },
    })
    publisher._node = object()
    publisher._Float32MultiArray = FakeFloat32MultiArray
    publisher._wheelarm_publisher = FakePublisher()
    pose = np.eye(4)
    pose[:3, 3] = (0.6, 0.2, 0.4)

    data = publisher.update_wheelarm_target(pose)

    self.assertEqual(len(publisher._wheelarm_publisher.messages), 1)
    message = publisher._wheelarm_publisher.messages[0]
    self.assertEqual(len(message.data), 7)
    self.assertEqual(message.data, [float(np.float32(value)) for value in data])

  def test_wheelarm_update_publishes_matching_base_pose_for_rviz(self):
    publisher = Ros2PosePublisher({
      "enabled": True,
      "wheelarm_target": {
        "enabled": True,
        "target_frame": "base_Link",
        "base_from_camera": np.eye(4),
      },
    })
    publisher._node = object()
    publisher._Float32MultiArray = FakeFloat32MultiArray
    publisher._PoseStamped = FakePoseStamped
    publisher._wheelarm_publisher = FakePublisher()
    publisher._wheelarm_base_pose_publisher = FakePublisher()
    stamp = SimpleNamespace(sec=123, nanosec=456)
    pose = np.eye(4)
    pose[:3, 3] = (0.6, 0.2, 0.4)

    data = publisher.update_wheelarm_target(pose, stamp=stamp)

    self.assertEqual(len(publisher._wheelarm_base_pose_publisher.messages), 1)
    message = publisher._wheelarm_base_pose_publisher.messages[0]
    self.assertIs(message.header.stamp, stamp)
    self.assertEqual(message.header.frame_id, "base_Link")
    np.testing.assert_allclose(
      (message.pose.position.x, message.pose.position.y, message.pose.position.z),
      data[:3],
      atol=1e-7,
    )
    np.testing.assert_allclose(
      (
        message.pose.orientation.w,
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
      ),
      data[3:],
      atol=1e-7,
    )


if __name__ == "__main__":
  unittest.main()
