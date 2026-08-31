from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


def load_predictability_module():
  module_path = Path(__file__).resolve().parents[2] / "FoundationPose" / "candidate_predictability.py"
  spec = importlib.util.spec_from_file_location("candidate_predictability_test", module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {module_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


predictability = load_predictability_module()


def pose(translation):
  value = np.eye(4, dtype=np.float32)
  value[:3, 3] = translation
  return value


class CandidatePredictabilityTests(unittest.TestCase):
  def test_translation_errors_and_candidate_identity_are_preserved(self):
    poses = np.stack([
        pose([0.003, 0.004, 0.610]),
        pose([0.000, 0.000, 0.600]),
        pose([-0.006, 0.008, 0.580]),
    ])
    result = predictability.evaluate_candidate_predictability(
        coarse_poses=poses,
        reference_pose=pose([0.000, 0.000, 0.600]),
        candidate_ids=np.asarray([0, 1, 2]),
        source_candidate_ids=np.asarray([7, 19, 31]),
        selected_candidate_id=1,
    )

    self.assertEqual(3, result["candidate_count"])
    self.assertEqual([7, 19, 31], [item["source_candidate_id"] for item in result["candidates"]])
    self.assertAlmostEqual(0.005, result["candidates"][0]["error_xy_m"], places=7)
    self.assertAlmostEqual(0.010, result["candidates"][0]["error_z_m"], places=7)
    self.assertAlmostEqual(np.sqrt(0.000125), result["candidates"][0]["error_translation_m"], places=7)
    self.assertTrue(result["candidates"][1]["selected_by_phase2"])
    self.assertEqual(0.0, result["candidates"][1]["error_translation_m"])
    self.assertFalse(result["candidates"][2]["selected_by_phase2"])

  def test_metrics_are_deterministic(self):
    poses = np.stack([pose([0.0, 0.0, 0.5]), pose([0.001, 0.0, 0.5])])
    first = predictability.evaluate_candidate_predictability(poses, poses[0], np.asarray([0, 1]))
    second = predictability.evaluate_candidate_predictability(poses, poses[0], np.asarray([0, 1]))
    self.assertEqual(first, second)

  def test_invalid_inputs_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "shape"):
      predictability.evaluate_candidate_predictability(
          np.zeros((3, 3, 3)), np.eye(4), np.asarray([0, 1, 2])
      )
    with self.assertRaisesRegex(ValueError, "identity count"):
      predictability.evaluate_candidate_predictability(
          np.stack([np.eye(4), np.eye(4)]), np.eye(4), np.asarray([0])
      )
    with self.assertRaisesRegex(ValueError, "unique"):
      predictability.evaluate_candidate_predictability(
          np.stack([np.eye(4), np.eye(4)]), np.eye(4), np.asarray([1, 1])
      )


if __name__ == "__main__":
  unittest.main()
