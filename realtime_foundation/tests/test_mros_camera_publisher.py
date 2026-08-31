from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


CAMERA_DIR = Path(__file__).resolve().parents[1] / "camera"
sys.path.insert(0, str(CAMERA_DIR))
publisher_module = SourceFileLoader(
    "mros_depth_color_publish",
    str(CAMERA_DIR / "depth_color_publish"),
).load_module()


class FakeHeader:
  def __init__(self):
    self.seq = 0
    self.stamp = None
    self.frame_id = ""


class FakeImage:
  def __init__(self):
    self.header = FakeHeader()
    self.height = 0
    self.width = 0
    self.encoding = ""
    self.is_bigendian = 0
    self.step = 0
    self.data = b""


class FakeCameraInfo:
  def __init__(self):
    self.header = FakeHeader()
    self.height = 0
    self.width = 0
    self.distortion_model = ""
    self.D = []
    self.K = []
    self.R = []
    self.P = []
    self.binning_x = 0
    self.binning_y = 0


class MrosCameraPublisherTest(unittest.TestCase):
  def setUp(self):
    self.mros = SimpleNamespace(Time=SimpleNamespace(now=lambda: "now"))
    self.stamp = SimpleNamespace(sec=12, nsec=34)

  def test_color_message_preserves_topic_payload_format(self):
    color = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    message = publisher_module.build_color_message(
        FakeImage, self.mros, color, 7, "camera_color_optical_frame", self.stamp
    )

    self.assertEqual(message.header.seq, 7)
    self.assertIs(message.header.stamp, self.stamp)
    self.assertEqual(message.header.frame_id, "camera_color_optical_frame")
    self.assertEqual((message.height, message.width), (2, 3))
    self.assertEqual(message.encoding, "rgb8")
    self.assertEqual(message.step, 9)
    self.assertEqual(message.data, color.tobytes())

  def test_depth_message_uses_float32_meters(self):
    depth = np.array([[0.5, 1.25, 2.0]], dtype=np.float32)

    message = publisher_module.build_depth_message(
        FakeImage, self.mros, depth, 8, "camera_color_optical_frame", self.stamp
    )

    self.assertEqual(message.header.seq, 8)
    self.assertIs(message.header.stamp, self.stamp)
    self.assertEqual(message.encoding, "32FC1")
    self.assertEqual(message.step, 12)
    np.testing.assert_array_equal(
        np.frombuffer(message.data, dtype="<f4"),
        depth.reshape(-1),
    )

  def test_camera_info_uses_same_stamp_and_standard_intrinsics(self):
    K = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 601.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    message = publisher_module.build_camera_info_message(
        FakeCameraInfo,
        self.mros,
        K,
        640,
        480,
        9,
        "camera_color_optical_frame",
        self.stamp,
    )

    self.assertEqual(message.header.seq, 9)
    self.assertIs(message.header.stamp, self.stamp)
    self.assertEqual((message.width, message.height), (640, 480))
    np.testing.assert_array_equal(np.asarray(message.K).reshape(3, 3), K)
    self.assertEqual(len(message.P), 12)

  def test_runtime_config_keeps_existing_topic_names(self):
    config = {
        "camera": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "mros": {
                "color_topic": "/camera/color/image_raw",
                "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
                "publisher_cpu_cores": [min(publisher_module.os.sched_getaffinity(0))],
            },
        },
    }

    runtime = publisher_module.build_runtime_config(config)

    self.assertEqual(runtime["color_topic"], "/camera/color/image_raw")
    self.assertEqual(runtime["depth_topic"], "/camera/aligned_depth_to_color/image_raw")
    self.assertEqual(runtime["camera_info_topic"], "/camera/color/camera_info")


if __name__ == "__main__":
  unittest.main()
