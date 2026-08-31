from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


def load_selector_module():
  module_path = Path(__file__).resolve().parents[1] / "tracking" / "rotation_candidate_selector.py"
  spec = importlib.util.spec_from_file_location("rotation_candidate_selector_test", module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {module_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


selector = load_selector_module()


def axis_angle_rotation(axis, angle):
  axis = np.asarray(axis, dtype=np.float64)
  axis /= np.linalg.norm(axis)
  x, y, z = axis
  skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
  return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def sample_rotation_grid():
  axes = (
      (1.0, 0.0, 0.0),
      (0.0, 1.0, 0.0),
      (0.0, 0.0, 1.0),
      (1.0, 1.0, 0.0),
      (1.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
  )
  rotations = [np.eye(3)]
  for axis in axes:
    rotations.extend(axis_angle_rotation(axis, angle) for angle in (np.pi / 2.0, np.pi))
  grid = np.tile(np.eye(4), (len(rotations), 1, 1))
  grid[:, :3, :3] = np.asarray(rotations)
  return grid


class RotationCandidateSelectorTests(unittest.TestCase):
  def test_selection_is_deterministic_unique_and_starts_near_identity(self):
    grid = sample_rotation_grid()
    first = selector.select_so3_farthest_candidates(grid, 5)
    second = selector.select_so3_farthest_candidates(grid, 5)

    np.testing.assert_array_equal(first.indices, second.indices)
    self.assertEqual(5, len(first.indices))
    self.assertEqual(5, len(set(first.indices.tolist())))
    self.assertEqual(0, int(first.indices[0]))
    self.assertGreater(first.min_pairwise_angle_deg, 0.0)

  def test_selector_accepts_three_by_three_rotation_grid(self):
    grid = sample_rotation_grid()[:, :3, :3]
    selection = selector.select_so3_farthest_candidates(grid, 5)
    self.assertEqual(5, len(selection.indices))

  def test_selector_rejects_invalid_counts(self):
    grid = sample_rotation_grid()
    with self.assertRaisesRegex(ValueError, "at least 1"):
      selector.select_so3_farthest_candidates(grid, 0)
    with self.assertRaisesRegex(ValueError, "exceeds rotation grid size"):
      selector.select_so3_farthest_candidates(grid, len(grid) + 1)

  def test_selector_rejects_invalid_rotation_matrix(self):
    grid = sample_rotation_grid()
    grid[1, 0, 0] = 2.0
    with self.assertRaisesRegex(ValueError, "invalid rotation matrices"):
      selector.select_so3_farthest_candidates(grid, 5)

  def test_explicit_selector_preserves_requested_source_index(self):
    grid = sample_rotation_grid()
    selection = selector.select_explicit_candidates(grid, [7])
    np.testing.assert_array_equal(np.asarray([7]), selection.indices)
    self.assertEqual(0.0, selection.min_pairwise_angle_deg)

  def test_explicit_selector_rejects_duplicates_and_out_of_range_indices(self):
    grid = sample_rotation_grid()
    with self.assertRaisesRegex(ValueError, "unique"):
      selector.select_explicit_candidates(grid, [2, 2])
    with self.assertRaisesRegex(ValueError, "outside rotation grid"):
      selector.select_explicit_candidates(grid, [len(grid)])


if __name__ == "__main__":
  unittest.main()
