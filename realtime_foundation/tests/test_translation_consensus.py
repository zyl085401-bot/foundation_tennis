from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


def load_consensus_module():
  module_path = Path(__file__).resolve().parents[2] / "FoundationPose" / "translation_consensus.py"
  spec = importlib.util.spec_from_file_location("translation_consensus_test", module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {module_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


consensus = load_consensus_module()


class TranslationConsensusTests(unittest.TestCase):
  def test_medoid_consensus_tolerates_one_outlier(self):
    translations = np.asarray([
        [0.000, 0.000, 0.600],
        [0.001, 0.000, 0.600],
        [-0.001, 0.001, 0.599],
        [0.000, -0.001, 0.602],
        [0.030, 0.000, 0.640],
    ])
    result = consensus.evaluate_translation_consensus(
        translations,
        median_distance_threshold_m=0.004,
        inlier_distance_threshold_m=0.006,
        min_inlier_count=4,
    )

    self.assertTrue(result.passed)
    self.assertIn(result.medoid_index, (0, 1, 2, 3))
    self.assertEqual(4, result.inlier_count)
    self.assertLess(result.median_distance_m, 0.004)
    self.assertGreater(result.max_distance_m, 0.04)

  def test_consensus_rejects_split_candidates(self):
    translations = np.asarray([
        [0.000, 0.000, 0.600],
        [0.001, 0.000, 0.601],
        [0.020, 0.000, 0.620],
        [0.021, 0.000, 0.621],
        [0.040, 0.000, 0.640],
    ])
    result = consensus.evaluate_translation_consensus(
        translations,
        median_distance_threshold_m=0.004,
        inlier_distance_threshold_m=0.006,
        min_inlier_count=4,
    )

    self.assertFalse(result.passed)
    self.assertLess(result.inlier_count, 4)

  def test_selection_is_deterministic_for_ties(self):
    translations = np.asarray([
        [-0.001, 0.000, 0.600],
        [0.001, 0.000, 0.600],
        [0.000, -0.001, 0.600],
        [0.000, 0.001, 0.600],
        [0.000, 0.000, 0.600],
    ])
    first = consensus.evaluate_translation_consensus(translations, 0.004, 0.006, 4)
    second = consensus.evaluate_translation_consensus(translations, 0.004, 0.006, 4)
    self.assertEqual(first.medoid_index, second.medoid_index)

  def test_invalid_inputs_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "shape"):
      consensus.evaluate_translation_consensus(np.zeros((5, 2)), 0.004, 0.006, 4)
    with self.assertRaisesRegex(ValueError, "at least two"):
      consensus.evaluate_translation_consensus(np.zeros((1, 3)), 0.004, 0.006, 1)
    with self.assertRaisesRegex(ValueError, "min_inlier_count"):
      consensus.evaluate_translation_consensus(np.zeros((5, 3)), 0.004, 0.006, 6)


if __name__ == "__main__":
  unittest.main()
