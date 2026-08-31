from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RotationCandidateSelection:
  indices: np.ndarray
  min_pairwise_angle_deg: float


def _rotation_matrices(rotation_grid: np.ndarray) -> np.ndarray:
  grid = np.asarray(rotation_grid)
  if grid.ndim != 3 or grid.shape[1:] not in ((3, 3), (4, 4)):
    raise ValueError(f"rotation_grid must have shape (N,3,3) or (N,4,4), got {grid.shape}")
  if len(grid) == 0:
    raise ValueError("rotation_grid must contain at least one candidate")
  rotations = np.asarray(grid[:, :3, :3], dtype=np.float64)
  if not np.isfinite(rotations).all():
    raise ValueError("rotation_grid contains non-finite values")
  identity = np.eye(3, dtype=np.float64)
  orthogonality_error = np.max(np.abs(np.swapaxes(rotations, 1, 2) @ rotations - identity))
  determinants = np.linalg.det(rotations)
  if orthogonality_error > 1e-3 or np.max(np.abs(determinants - 1.0)) > 1e-3:
    raise ValueError("rotation_grid contains invalid rotation matrices")
  return rotations


def _angular_distances(rotations: np.ndarray, reference: np.ndarray) -> np.ndarray:
  relative = np.swapaxes(rotations, 1, 2) @ reference
  cosine = (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5
  return np.arccos(np.clip(cosine, -1.0, 1.0))


def _minimum_pairwise_angle_deg(rotations: np.ndarray, selected_indices: np.ndarray) -> float:
  if len(selected_indices) < 2:
    return 0.0
  pairwise_angles = []
  for position, selected_index in enumerate(selected_indices[:-1]):
    distances = _angular_distances(
        rotations[selected_indices[position + 1:]],
        rotations[selected_index],
    )
    pairwise_angles.extend(distances.tolist())
  return float(np.rad2deg(min(pairwise_angles)))


def select_so3_farthest_candidates(rotation_grid: np.ndarray, count: int) -> RotationCandidateSelection:
  """Select a deterministic, well-spaced subset of an SO(3) rotation grid."""
  rotations = _rotation_matrices(rotation_grid)
  count = int(count)
  if count < 1:
    raise ValueError(f"candidate count must be at least 1, got {count}")
  if count > len(rotations):
    raise ValueError(f"candidate count {count} exceeds rotation grid size {len(rotations)}")

  # Start from the candidate nearest the canonical identity rotation.
  identity_distance = _angular_distances(rotations, np.eye(3, dtype=np.float64))
  first_index = int(np.argmin(identity_distance))
  selected = [first_index]
  selected_mask = np.zeros(len(rotations), dtype=bool)
  selected_mask[first_index] = True
  min_distance = _angular_distances(rotations, rotations[first_index])

  while len(selected) < count:
    available_distance = np.where(selected_mask, -np.inf, min_distance)
    next_index = int(np.argmax(available_distance))
    selected.append(next_index)
    selected_mask[next_index] = True
    min_distance = np.minimum(
        min_distance,
        _angular_distances(rotations, rotations[next_index]),
    )

  selected_indices = np.asarray(selected, dtype=np.int64)
  return RotationCandidateSelection(
      indices=selected_indices,
      min_pairwise_angle_deg=_minimum_pairwise_angle_deg(rotations, selected_indices),
  )


def select_explicit_candidates(rotation_grid: np.ndarray, indices) -> RotationCandidateSelection:
  """Select fixed source indices from the original rotation grid."""
  rotations = _rotation_matrices(rotation_grid)
  selected_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
  if len(selected_indices) == 0:
    raise ValueError("explicit candidate indices must not be empty")
  if len(np.unique(selected_indices)) != len(selected_indices):
    raise ValueError("explicit candidate indices must be unique")
  if np.any(selected_indices < 0) or np.any(selected_indices >= len(rotations)):
    raise ValueError(
        f"explicit candidate index outside rotation grid size {len(rotations)}: {selected_indices.tolist()}"
    )
  return RotationCandidateSelection(
      indices=selected_indices,
      min_pairwise_angle_deg=_minimum_pairwise_angle_deg(rotations, selected_indices),
  )
