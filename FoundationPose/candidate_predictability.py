from __future__ import annotations

import numpy as np


def evaluate_candidate_predictability(
    coarse_poses: np.ndarray,
    reference_pose: np.ndarray,
    candidate_ids: np.ndarray,
    source_candidate_ids: np.ndarray | None = None,
    selected_candidate_id: int | None = None,
) -> dict:
  """Compare each coarse candidate translation with the unchanged Phase 2 result."""
  poses = np.asarray(coarse_poses, dtype=np.float64)
  reference = np.asarray(reference_pose, dtype=np.float64)
  ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
  source_ids = ids if source_candidate_ids is None else np.asarray(source_candidate_ids, dtype=np.int64).reshape(-1)

  if poses.ndim != 3 or poses.shape[1:] != (4, 4):
    raise ValueError(f'coarse_poses must have shape (N,4,4), got {poses.shape}')
  if reference.shape != (4, 4):
    raise ValueError(f'reference_pose must have shape (4,4), got {reference.shape}')
  if len(poses) == 0:
    raise ValueError('coarse_poses must contain at least one candidate')
  if len(ids) != len(poses) or len(source_ids) != len(poses):
    raise ValueError(
        f'candidate identity count must match poses: poses={len(poses)}, ids={len(ids)}, source_ids={len(source_ids)}'
    )
  if len(np.unique(ids)) != len(ids):
    raise ValueError('candidate_ids must be unique')
  if not np.isfinite(poses).all() or not np.isfinite(reference).all():
    raise ValueError('poses contain non-finite values')

  translation_delta = poses[:, :3, 3] - reference[:3, 3]
  error_xy = np.linalg.norm(translation_delta[:, :2], axis=1)
  error_z = np.abs(translation_delta[:, 2])
  error_translation = np.linalg.norm(translation_delta, axis=1)
  selected_id = None if selected_candidate_id is None else int(selected_candidate_id)

  return {
      'schema_version': 1,
      'reference': 'phase2_final_pose',
      'candidate_count': int(len(poses)),
      'reference_centered_pose': reference.astype(np.float32).tolist(),
      'reference_translation_m': reference[:3, 3].astype(np.float32).tolist(),
      'selected_candidate_id': selected_id,
      'candidates': [
          {
              'candidate_id': int(ids[index]),
              'source_candidate_id': int(source_ids[index]),
              'selected_by_phase2': selected_id is not None and int(ids[index]) == selected_id,
              'centered_pose': poses[index].astype(np.float32).tolist(),
              'translation_m': poses[index, :3, 3].astype(np.float32).tolist(),
              'error_xy_m': float(error_xy[index]),
              'error_z_m': float(error_z[index]),
              'error_translation_m': float(error_translation[index]),
          }
          for index in range(len(poses))
      ],
  }
