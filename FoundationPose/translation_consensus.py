from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TranslationConsensusResult:
  medoid_index: int
  distances_m: np.ndarray
  median_distance_m: float
  mean_distance_m: float
  max_distance_m: float
  inlier_count: int
  passed: bool


def evaluate_translation_consensus(
    translations: np.ndarray,
    median_distance_threshold_m: float,
    inlier_distance_threshold_m: float,
    min_inlier_count: int,
) -> TranslationConsensusResult:
  """Select a translation medoid and evaluate robust candidate agreement."""
  values = np.asarray(translations, dtype=np.float64)
  if values.ndim != 2 or values.shape[1] != 3:
    raise ValueError(f"translations must have shape (N,3), got {values.shape}")
  if len(values) < 2:
    raise ValueError("translation consensus requires at least two candidates")
  if not np.isfinite(values).all():
    raise ValueError("translations contain non-finite values")

  median_threshold = float(median_distance_threshold_m)
  inlier_threshold = float(inlier_distance_threshold_m)
  min_inliers = int(min_inlier_count)
  if median_threshold < 0 or inlier_threshold < 0:
    raise ValueError("translation consensus distance thresholds must be non-negative")
  if not 1 <= min_inliers <= len(values):
    raise ValueError(
        f"min_inlier_count must be between 1 and {len(values)}, got {min_inliers}"
    )

  pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
  medoid_index = int(np.argmin(pairwise.sum(axis=1)))
  medoid_distances = pairwise[medoid_index]
  other_distances = np.delete(medoid_distances, medoid_index)
  median_distance = float(np.median(other_distances))
  mean_distance = float(np.mean(other_distances))
  max_distance = float(np.max(other_distances))
  inlier_count = int(np.count_nonzero(medoid_distances <= inlier_threshold))
  passed = median_distance <= median_threshold and inlier_count >= min_inliers

  return TranslationConsensusResult(
      medoid_index=medoid_index,
      distances_m=medoid_distances.astype(np.float32),
      median_distance_m=median_distance,
      mean_distance_m=mean_distance,
      max_distance_m=max_distance,
      inlier_count=inlier_count,
      passed=passed,
  )
