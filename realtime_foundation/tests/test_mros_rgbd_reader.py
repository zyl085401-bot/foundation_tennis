from types import SimpleNamespace
import unittest

import numpy as np

from realtime_foundation.camera.mros_rgbd_reader import (
    ApproximateImageSynchronizer,
    MrosRgbdReader,
    _camera_matrix,
    _color_image_to_rgb,
    _depth_image_to_meters,
    _stamp_seconds,
)


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


if __name__ == "__main__":
  unittest.main()
