from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from realtime_foundation.camera.mros_rgbd_reader import (
    ApproximateImageSynchronizer,
    DepthToColorAlignment,
    MrosRgbdReader,
    _camera_matrix,
    _color_image_to_rgb,
    _depth_image_to_meters,
    _stamp_seconds,
)


def make_alignment_config(
    width: int,
    height: int,
    *,
    enabled: bool = True,
    depth_K=None,
    color_K=None,
    rotation=None,
    translation_m=None,
):
  default_K = [
      [100.0, 0.0, width / 2.0],
      [0.0, 100.0, height / 2.0],
      [0.0, 0.0, 1.0],
  ]
  return {
      "enabled": enabled,
      "backend": "opencv_rgbd",
      "depth_dilation": False,
      "calibration_width": width,
      "calibration_height": height,
      "calibration_fps": 30,
      "validation": {
        "require_zero_distortion": True,
        "require_resolution_match": True,
      },
      "diagnostics": {"enabled": False},
      "depth": {
        "K": default_K if depth_K is None else depth_K,
        "distortion_model": "brown_conrady",
        "distortion": [0.0] * 5,
      },
      "color": {
        "K": default_K if color_K is None else color_K,
        "distortion_model": "inverse_brown_conrady",
        "distortion": [0.0] * 5,
      },
      "depth_to_color": {
        "rotation": np.eye(3).tolist() if rotation is None else rotation,
        "translation_m": [0.0, 0.0, 0.0] if translation_m is None else translation_m,
      },
  }


def make_stamp(timestamp: float):
  seconds = int(timestamp)
  nanoseconds = int(round((timestamp - seconds) * 1_000_000_000))
  return SimpleNamespace(sec=seconds, nsec=nanoseconds)


def make_message(timestamp: float, **fields):
  return SimpleNamespace(
      header=SimpleNamespace(stamp=make_stamp(timestamp), frame_id="camera"),
      **fields,
  )


class MrosRgbdConversionTest(unittest.TestCase):
  def test_stamp_uses_mros_nsec_field(self):
    message = make_message(123.25)

    self.assertAlmostEqual(_stamp_seconds(message), 123.25)

  def test_bgr_image_with_row_padding_converts_to_rgb(self):
    data = np.array([
        1, 2, 3, 4, 5, 6, 99, 99,
        7, 8, 9, 10, 11, 12, 99, 99,
    ], dtype=np.uint8)
    message = make_message(
        1.0,
        encoding="bgr8",
        width=2,
        height=2,
        step=8,
        is_bigendian=0,
        data=data,
    )

    image = _color_image_to_rgb(message)

    np.testing.assert_array_equal(
        image,
        np.array([
            [[3, 2, 1], [6, 5, 4]],
            [[9, 8, 7], [12, 11, 10]],
        ], dtype=np.uint8),
    )
    self.assertTrue(image.flags.c_contiguous)

  def test_big_endian_depth_with_padding_converts_to_meters(self):
    rows = np.array([
        [1000, 2000, 65535],
        [3000, 4000, 65535],
    ], dtype=">u2")
    message = make_message(
        1.0,
        encoding="16UC1",
        width=2,
        height=2,
        step=6,
        is_bigendian=1,
        data=rows.tobytes(),
    )

    depth = _depth_image_to_meters(message, 0.001)

    np.testing.assert_allclose(
        depth,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    self.assertEqual(depth.dtype, np.float32)
    self.assertTrue(depth.flags.c_contiguous)

  def test_camera_info_uses_uppercase_K(self):
    message = make_message(
        1.0,
        width=640,
        height=480,
        K=[600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0],
    )

    matrix = _camera_matrix(message)

    np.testing.assert_array_equal(
        matrix,
        np.array([
            [600.0, 0.0, 320.0],
            [0.0, 601.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
    )

  def test_reader_filters_invalid_depth_values(self):
    reader = MrosRgbdReader(
        color_topic="color",
        depth_topic="depth",
        camera_info_topic="info",
        width=2,
        height=1,
        depth_min=0.1,
        depth_max=2.0,
    )
    color = make_message(
        5.0,
        encoding="rgb8",
        width=2,
        height=1,
        step=6,
        is_bigendian=0,
        data=bytes((1, 2, 3, 4, 5, 6)),
    )
    depth = make_message(
        5.0,
        encoding="32FC1",
        width=2,
        height=1,
        step=8,
        is_bigendian=0,
        data=np.array([np.nan, 3.0], dtype="<f4").tobytes(),
    )
    camera_info = make_message(
        5.0,
        width=2,
        height=1,
        K=[100.0, 0.0, 1.0, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0],
    )

    _, converted_depth, _, timestamp = reader._convert_frame(color, depth, camera_info)

    np.testing.assert_array_equal(converted_depth, np.zeros((1, 2), dtype=np.float32))
    self.assertEqual(timestamp, 5.0)


class DepthToColorAlignmentTest(unittest.TestCase):
  DEPTH_K = [
      [389.875305175781, 0.0, 320.413757324219],
      [0.0, 389.875305175781, 236.457885742188],
      [0.0, 0.0, 1.0],
  ]
  COLOR_K = [
      [607.86669921875, 0.0, 324.967041015625],
      [0.0, 607.938354492188, 247.910064697266],
      [0.0, 0.0, 1.0],
  ]
  ROTATION = [
      [0.999917, 0.0117287, 0.00531798],
      [-0.0117292, 0.999931, 0.000057999],
      [-0.00531694, -0.00012037, 0.999986],
  ]
  TRANSLATION = [
      0.0147585337981582,
      -0.00018205500964541,
      -0.0000490047350467648,
  ]

  def actual_alignment_config(self, enabled=True):
    return make_alignment_config(
        640,
        480,
        enabled=enabled,
        depth_K=self.DEPTH_K,
        color_K=self.COLOR_K,
        rotation=self.ROTATION,
        translation_m=self.TRANSLATION,
    )

  def test_actual_extrinsics_match_manual_depth_to_color_projection(self):
    alignment = DepthToColorAlignment(self.actual_alignment_config(), 640, 480)
    source = np.zeros((480, 640), dtype=np.uint16)
    source[240, 320] = 1000

    registered = alignment.register(source, 0.001)

    depth_K = np.asarray(self.DEPTH_K, dtype=np.float64)
    color_K = np.asarray(self.COLOR_K, dtype=np.float64)
    rotation = np.asarray(self.ROTATION, dtype=np.float64)
    translation = np.asarray(self.TRANSLATION, dtype=np.float64)
    z = 1.0
    point_depth = np.array([
        (320.0 - depth_K[0, 2]) / depth_K[0, 0] * z,
        (240.0 - depth_K[1, 2]) / depth_K[1, 1] * z,
        z,
    ])
    point_color = rotation @ point_depth + translation
    expected_uv = np.array([
        color_K[0, 0] * point_color[0] / point_color[2] + color_K[0, 2],
        color_K[1, 1] * point_color[1] / point_color[2] + color_K[1, 2],
    ])
    y, x = np.argwhere(registered > 0.0)[0]

    self.assertLessEqual(abs(float(x) - expected_uv[0]), 0.5)
    self.assertLessEqual(abs(float(y) - expected_uv[1]), 0.5)
    self.assertAlmostEqual(float(registered[y, x]), float(point_color[2]), places=3)

  def test_registration_preserves_uint16_millimeter_scale_as_float_meters(self):
    alignment = DepthToColorAlignment(
        make_alignment_config(240, 240, translation_m=[0.1, 0.0, 0.0]),
        240,
        240,
    )
    source = np.zeros((240, 240), dtype=np.uint16)
    source[120, 120] = 1000

    registered = alignment.register(source, 0.001)

    self.assertEqual(registered.dtype, np.float32)
    self.assertAlmostEqual(float(registered[120, 130]), 1.0, places=6)

  def test_registration_keeps_nearest_depth_on_projection_collision(self):
    alignment = DepthToColorAlignment(
        make_alignment_config(240, 240, translation_m=[0.1, 0.0, 0.0]),
        240,
        240,
    )
    source = np.zeros((240, 240), dtype=np.uint16)
    source[100, 100] = 1000
    source[100, 105] = 2000

    registered = alignment.register(source, 0.001)

    self.assertAlmostEqual(float(registered[100, 110]), 1.0, places=6)
    self.assertEqual(np.count_nonzero(registered > 0.0), 1)

  def test_disabled_alignment_passes_raw_depth_through(self):
    alignment = DepthToColorAlignment(
        make_alignment_config(2, 1, enabled=False),
        2,
        1,
    )
    source = np.array([[1000, 2000]], dtype=np.uint16)

    registered = alignment.register(source, 0.001)

    np.testing.assert_allclose(registered, [[1.0, 2.0]])

  def test_reader_uses_static_color_K_without_camera_info(self):
    config = make_alignment_config(2, 1, enabled=False)
    reader = MrosRgbdReader(
        color_topic="color",
        depth_topic="depth",
        camera_info_topic=None,
        width=2,
        height=1,
        alignment_config=config,
    )
    color = make_message(
        5.0,
        encoding="rgb8",
        width=2,
        height=1,
        step=6,
        is_bigendian=0,
        data=bytes((1, 2, 3, 4, 5, 6)),
    )
    depth = make_message(
        5.0,
        encoding="16UC1",
        width=2,
        height=1,
        step=4,
        is_bigendian=0,
        data=np.array([500, 1000], dtype="<u2").tobytes(),
    )

    _, converted_depth, K, _ = reader._convert_frame(color, depth)

    np.testing.assert_allclose(converted_depth, [[0.5, 1.0]])
    np.testing.assert_array_equal(K, np.asarray(config["color"]["K"], dtype=np.float32))

  def test_reader_delivers_synchronized_static_calibration_frame_without_camera_info(self):
    reader = MrosRgbdReader(
        color_topic="color",
        depth_topic="depth",
        camera_info_topic=None,
        width=2,
        height=1,
        alignment_config=make_alignment_config(2, 1, enabled=False),
    )
    reader._started = True
    reader._mros = SimpleNamespace(ok=lambda: True)
    reader._handle_color_message(make_message(
        5.0,
        encoding="rgb8",
        width=2,
        height=1,
        step=6,
        is_bigendian=0,
        data=bytes((1, 2, 3, 4, 5, 6)),
    ))
    reader._handle_depth_message(make_message(
        5.0,
        encoding="16UC1",
        width=2,
        height=1,
        step=4,
        is_bigendian=0,
        data=np.array([500, 1000], dtype="<u2").tobytes(),
    ))

    with patch(
        "realtime_foundation.camera.mros_rgbd_reader.time.monotonic",
        side_effect=(10.0, 10.1),
    ):
      frame = reader.get_frame()

    self.assertEqual(frame[3], 5.0)
    self.assertIsNone(reader._camera_info_subscriber)

  def test_camera_config_accepts_static_calibration_without_camera_info_topic(self):
    reader = MrosRgbdReader.from_camera_config({
        "width": 2,
        "height": 1,
        "mros": {
            "color_topic": "/tron2/color",
            "depth_topic": "/tron2/depth",
            "camera_info_topic": None,
            "depth_to_color_alignment": make_alignment_config(2, 1, enabled=False),
        },
    })

    self.assertIsNone(reader.camera_info_topic)
    self.assertIsNotNone(reader.alignment)
    self.assertFalse(reader.alignment.enabled)

  def test_rejects_nonzero_distortion(self):
    config = make_alignment_config(640, 480)
    config["color"]["distortion"][0] = 0.01

    with self.assertRaisesRegex(RuntimeError, "Non-zero depth/color distortion"):
      DepthToColorAlignment(config, 640, 480)

  def test_rejects_invalid_rotation(self):
    config = make_alignment_config(640, 480)
    config["depth_to_color"]["rotation"][0][0] = 2.0

    with self.assertRaisesRegex(RuntimeError, "not a valid rotation matrix"):
      DepthToColorAlignment(config, 640, 480)

  def test_rejects_calibration_resolution_mismatch(self):
    config = make_alignment_config(640, 480)

    with self.assertRaisesRegex(RuntimeError, "resolution does not match"):
      DepthToColorAlignment(config, 848, 480)


class MrosRgbdDisconnectTest(unittest.TestCase):
  def make_reader(self, disconnect_timeout_sec=3.0, **kwargs):
    return MrosRgbdReader(
        color_topic="color",
        depth_topic="depth",
        camera_info_topic="info",
        disconnect_timeout_sec=disconnect_timeout_sec,
        **kwargs,
    )

  def test_callbacks_deliver_one_synchronized_frame_without_polling_subscribers(self):
    reader = self.make_reader(width=2, height=1)
    reader._started = True
    reader._mros = SimpleNamespace(ok=lambda: True)
    reader._color_subscriber = SimpleNamespace(
        readMsgRT=lambda: self.fail("get_frame must not poll the color subscriber")
    )
    reader._depth_subscriber = SimpleNamespace(
        readMsgRT=lambda: self.fail("get_frame must not poll the depth subscriber")
    )
    reader._camera_info_subscriber = SimpleNamespace(
        readMsgRT=lambda: self.fail("get_frame must not poll the CameraInfo subscriber")
    )
    color = make_message(
        5.0,
        encoding="rgb8",
        width=2,
        height=1,
        step=6,
        is_bigendian=0,
        data=bytes((1, 2, 3, 4, 5, 6)),
    )
    depth = make_message(
        5.0,
        encoding="32FC1",
        width=2,
        height=1,
        step=8,
        is_bigendian=0,
        data=np.array([0.5, 1.0], dtype="<f4").tobytes(),
    )
    camera_info = make_message(
        5.0,
        width=2,
        height=1,
        K=[100.0, 0.0, 1.0, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0],
    )
    reader._handle_color_message(color)
    reader._handle_depth_message(depth)
    reader._handle_camera_info_message(camera_info)

    with patch(
        "realtime_foundation.camera.mros_rgbd_reader.time.monotonic",
        side_effect=(10.0, 10.1),
    ):
      frame = reader.get_frame()

    self.assertEqual(frame[3], 5.0)
    np.testing.assert_array_equal(
        frame[0],
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(frame[1], np.array([[0.5, 1.0]], dtype=np.float32))
    self.assertEqual(reader._last_synchronized_frame_monotonic, 10.1)

  def test_does_not_disconnect_before_first_synchronized_frame(self):
    reader = self.make_reader()

    with patch(
        "realtime_foundation.camera.mros_rgbd_reader.time.monotonic",
        return_value=100.0,
    ):
      reader._raise_if_disconnected()

  def test_disconnects_when_stream_stalls_at_configured_limit(self):
    reader = self.make_reader(disconnect_timeout_sec=3.0)
    reader._last_synchronized_frame_monotonic = 10.0

    with patch(
        "realtime_foundation.camera.mros_rgbd_reader.time.monotonic",
        return_value=12.9,
    ):
      reader._raise_if_disconnected()

    with patch(
        "realtime_foundation.camera.mros_rgbd_reader.time.monotonic",
        return_value=13.0,
    ):
      with self.assertRaisesRegex(TimeoutError, "stream disconnected"):
        reader._raise_if_disconnected()

  def test_get_frame_raises_after_continuous_timeout(self):
    reader = self.make_reader(disconnect_timeout_sec=3.0)
    reader.frame_timeout_sec = 0.2
    reader._started = True
    reader._mros = SimpleNamespace(ok=lambda: True)
    empty_subscriber = SimpleNamespace(readMsgRT=lambda: None)
    reader._color_subscriber = empty_subscriber
    reader._depth_subscriber = empty_subscriber
    reader._camera_info_subscriber = empty_subscriber
    reader._last_synchronized_frame_monotonic = 10.0

    with patch(
        "realtime_foundation.camera.mros_rgbd_reader.time.monotonic",
        side_effect=(13.0, 13.2, 13.2),
    ):
      with self.assertRaisesRegex(TimeoutError, "stream disconnected"):
        reader.get_frame()

  def test_camera_config_sets_disconnect_timeout(self):
    reader = MrosRgbdReader.from_camera_config({
        "mros": {
            "color_topic": "color",
            "depth_topic": "depth",
            "camera_info_topic": "info",
            "disconnect_timeout_sec": 4.5,
        },
    })

    self.assertEqual(reader.disconnect_timeout_sec, 4.5)

  def test_rejects_non_positive_disconnect_timeout(self):
    with self.assertRaisesRegex(ValueError, "disconnect_timeout_sec"):
      self.make_reader(disconnect_timeout_sec=0.0)


class ApproximateImageSynchronizerTest(unittest.TestCase):
  def test_pairs_nearest_messages_within_slop(self):
    synchronizer = ApproximateImageSynchronizer(queue_size=4, slop_sec=0.03)
    color_early = make_message(1.000)
    color_near = make_message(1.020)
    depth = make_message(1.018)
    synchronizer.add_color(color_early)
    synchronizer.add_color(color_near)
    synchronizer.add_depth(depth)

    match = synchronizer.pop_match()

    self.assertIsNotNone(match)
    self.assertIs(match[0], color_near)
    self.assertIs(match[1], depth)

  def test_does_not_pair_messages_outside_slop(self):
    synchronizer = ApproximateImageSynchronizer(queue_size=4, slop_sec=0.03)
    synchronizer.add_color(make_message(1.0))
    synchronizer.add_depth(make_message(1.1))

    self.assertIsNone(synchronizer.pop_match())

  def test_matched_messages_are_not_reused(self):
    synchronizer = ApproximateImageSynchronizer(queue_size=4, slop_sec=0.03)
    synchronizer.add_color(make_message(1.0))
    synchronizer.add_depth(make_message(1.0))

    self.assertIsNotNone(synchronizer.pop_match())
    self.assertIsNone(synchronizer.pop_match())

  def test_records_match_delta_and_queue_drops(self):
    synchronizer = ApproximateImageSynchronizer(queue_size=1, slop_sec=0.01)
    synchronizer.add_color(make_message(1.0))
    synchronizer.add_color(make_message(2.0))
    synchronizer.add_depth(make_message(2.005))

    self.assertIsNotNone(synchronizer.pop_match())
    self.assertAlmostEqual(synchronizer.last_match_delta_sec, 0.005)
    self.assertEqual(synchronizer.matched_pair_count, 1)
    self.assertEqual(synchronizer.dropped_color_count, 1)


if __name__ == "__main__":
  unittest.main()
