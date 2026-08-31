from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np


DISTILLATION_SCHEMA_VERSION = 1
DEFAULT_FEATURE_VERSION = "refinenet_ab_mean_v1"


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if isinstance(value, np.generic):
    return _json_safe(value.item())
  if isinstance(value, float) and not np.isfinite(value):
    return None
  if isinstance(value, Path):
    return str(value)
  return value


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _safe_identifier(value: object) -> str:
  text = str(value)
  normalized = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in text)
  return normalized.strip("_") or "unknown"


class DistillationCaptureWriter:
  """Persist validated Fine-candidate groups without changing Legacy ranking."""

  def __init__(self, config: dict[str, Any] | None, version_metadata: dict[str, Any]):
    config = dict(config or {})
    self.enabled = bool(config.get("enabled", False))
    self.output_root = Path(
        config.get("output_dir", "realtime_foundation/outputs/distillation_capture")
    ).expanduser().resolve()
    self.expected_candidate_count = max(1, int(config.get("expected_candidate_count", 5)))
    self.max_groups = max(0, int(config.get("max_groups", 200)))
    self.stride = max(1, int(config.get("stride", 1)))
    self.strict = bool(config.get("strict", False))
    self.sequence_id = config.get("sequence_id")
    self.feature_version = str(config.get("feature_version", DEFAULT_FEATURE_VERSION))
    self.capture_attempts = 0
    self.saved_groups = 0
    self.last_error: str | None = None
    self._lock = threading.Lock()
    self.run_id = str(config.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    self.run_dir = self.output_root / _safe_identifier(self.run_id)
    self.groups_dir = self.run_dir / "groups"
    self.manifest_path = self.run_dir / "manifest.jsonl"
    self.version_metadata = dict(version_metadata)

    if not self.enabled:
      return
    self.groups_dir.mkdir(parents=True, exist_ok=False)
    dataset_metadata = {
        "schema_version": DISTILLATION_SCHEMA_VERSION,
        "feature_version": self.feature_version,
        "created_at_utc": _utc_now(),
        "run_id": self.run_id,
        "expected_candidate_count": self.expected_candidate_count,
        "stride": self.stride,
        "max_groups": self.max_groups,
        "sequence_id": self.sequence_id,
        "version_metadata": self.version_metadata,
    }
    self._write_json_atomic(self.run_dir / "dataset_metadata.json", dataset_metadata)

  def write_group(
      self,
      group: dict[str, Any] | None,
      *,
      frame_id: int | None,
      timestamp: float | None,
      object_id: object | None,
      sequence_id: object | None = None,
      source: str = "runtime",
  ) -> dict[str, Any] | None:
    if not self.enabled or group is None:
      return None

    with self._lock:
      self.capture_attempts += 1
      if (self.capture_attempts - 1) % self.stride != 0:
        return None
      if self.max_groups > 0 and self.saved_groups >= self.max_groups:
        return None
      try:
        arrays, group_summary = self._validate_and_flatten(group)
        sample_index = self.saved_groups + 1
        frame_text = "unknown" if frame_id is None else f"{int(frame_id):06d}"
        sample_id = f"group_{sample_index:06d}_frame_{frame_text}"
        npz_path = self.groups_dir / f"{sample_id}.npz"
        json_path = self.groups_dir / f"{sample_id}.json"
        resolved_sequence_id = sequence_id if sequence_id is not None else self.sequence_id
        metadata = {
            "schema_version": DISTILLATION_SCHEMA_VERSION,
            "feature_version": self.feature_version,
            "sample_id": sample_id,
            "run_id": self.run_id,
            "sequence_id": resolved_sequence_id,
            "frame_id": None if frame_id is None else int(frame_id),
            "timestamp": None if timestamp is None else float(timestamp),
            "object_id": object_id,
            "source": str(source),
            "created_at_utc": _utc_now(),
            "array_file": npz_path.name,
            **group_summary,
        }
        self._write_npz_atomic(npz_path, arrays)
        self._write_json_atomic(json_path, metadata)
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
          manifest.write(json.dumps(_json_safe(metadata), ensure_ascii=False, sort_keys=True))
          manifest.write("\n")
        self.saved_groups = sample_index
        self.last_error = None
        return {
            "sample_id": sample_id,
            "npz": str(npz_path),
            "json": str(json_path),
        }
      except Exception as error:
        self.last_error = f"{type(error).__name__}: {error}"
        if self.strict:
          raise
        return None

  def _validate_and_flatten(self, group: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    schema_version = int(group.get("schema_version", 0))
    if schema_version != DISTILLATION_SCHEMA_VERSION:
      raise ValueError(f"Unsupported distillation group schema {schema_version}")
    feature_version = str(group.get("feature_version", ""))
    if feature_version != self.feature_version:
      raise ValueError(
          f"Feature version mismatch: group={feature_version!r}, writer={self.feature_version!r}"
      )

    candidate_ids = np.asarray(group["candidate_ids"], dtype=np.int64).reshape(-1)
    candidate_count = len(candidate_ids)
    if int(group.get("candidate_count", candidate_count)) != candidate_count:
      raise ValueError("candidate_count does not match candidate_ids")
    if candidate_count != self.expected_candidate_count:
      raise ValueError(
          f"Expected {self.expected_candidate_count} candidates, got {candidate_count}"
      )
    if len(np.unique(candidate_ids)) != candidate_count:
      raise ValueError("candidate_ids must be unique within a group")

    arrays: dict[str, np.ndarray] = {
        "candidate_ids": candidate_ids,
        "teacher_raw_logits": self._candidate_array(group["teacher_raw_logits"], candidate_count, "teacher_raw_logits"),
        "teacher_public_scores": self._candidate_array(group["teacher_public_scores"], candidate_count, "teacher_public_scores"),
        "teacher_order_positions": np.asarray(group["teacher_order_positions"], dtype=np.int64).reshape(candidate_count),
        "teacher_order_candidate_ids": np.asarray(group["teacher_order_candidate_ids"], dtype=np.int64).reshape(candidate_count),
        "final_poses": self._candidate_array(group["final_poses"], candidate_count, "final_poses", trailing_shape=(4, 4)),
    }
    expected_order = np.argsort(arrays["teacher_raw_logits"])[::-1]
    if not np.array_equal(arrays["teacher_order_positions"], expected_order):
      raise ValueError("teacher_order_positions does not match teacher_raw_logits")
    if not np.array_equal(arrays["teacher_order_candidate_ids"], candidate_ids[expected_order]):
      raise ValueError("teacher_order_candidate_ids does not match candidate_ids")
    if int(group["teacher_top1_candidate_id"]) != int(candidate_ids[expected_order[0]]):
      raise ValueError("teacher_top1_candidate_id does not match teacher_raw_logits")
    if not np.allclose(
        arrays["teacher_public_scores"],
        arrays["teacher_raw_logits"] + 100.0,
        rtol=1e-5,
        atol=1e-5,
    ):
      raise ValueError("teacher_public_scores does not match raw logits plus 100")

    iterations = list(group.get("fine_iterations", ()))
    if not iterations:
      raise ValueError("Fine Refiner iteration data is empty")
    iteration_backends = []
    for position, iteration in enumerate(iterations, start=1):
      prefix = f"fine_iter_{position:02d}"
      iteration_number = int(iteration.get("iteration", position))
      if iteration_number != position:
        raise ValueError(f"Unexpected Fine iteration number {iteration_number} at position {position}")
      arrays[f"{prefix}__poses_before"] = self._candidate_array(
          iteration["poses_before"], candidate_count, "poses_before", trailing_shape=(4, 4)
      )
      arrays[f"{prefix}__poses_after"] = self._candidate_array(
          iteration["poses_after"], candidate_count, "poses_after", trailing_shape=(4, 4)
      )
      arrays[f"{prefix}__raw_trans"] = self._candidate_array(
          iteration["raw_trans"], candidate_count, "raw_trans"
      )
      arrays[f"{prefix}__raw_rot"] = self._candidate_array(
          iteration["raw_rot"], candidate_count, "raw_rot"
      )
      arrays[f"{prefix}__trans_applied"] = self._candidate_array(
          iteration["trans_applied"], candidate_count, "trans_applied", trailing_shape=(3,)
      )
      arrays[f"{prefix}__rot_applied"] = self._candidate_array(
          iteration["rot_applied"], candidate_count, "rot_applied", trailing_shape=(3, 3)
      )
      shared_feature = self._candidate_array(
          iteration["shared_feature"], candidate_count, "shared_feature"
      )
      if shared_feature.ndim != 2:
        raise ValueError(f"shared_feature must have shape [N,D], got {shared_feature.shape}")
      arrays[f"{prefix}__shared_feature"] = shared_feature
      input_size = np.asarray(iteration["input_size"], dtype=np.int32).reshape(2)
      arrays[f"{prefix}__input_size"] = input_size
      iteration_backends.append(str(iteration.get("network_backend", "unknown")))

    if not np.allclose(
        arrays[f"fine_iter_{len(iterations):02d}__poses_after"],
        arrays["final_poses"],
        rtol=1e-5,
        atol=1e-6,
    ):
      raise ValueError("Final Fine poses do not match Final Scorer input poses")

    teacher_margin = float(group["teacher_margin"])
    if candidate_count > 1:
      recomputed_margin = float(
          arrays["teacher_raw_logits"][expected_order[0]]
          - arrays["teacher_raw_logits"][expected_order[1]]
      )
      if not np.isclose(teacher_margin, recomputed_margin, rtol=1e-5, atol=1e-7):
        raise ValueError("teacher_margin does not match teacher_raw_logits")

    summary = {
        "candidate_count": candidate_count,
        "candidate_ids": candidate_ids.tolist(),
        "fine_iteration_count": len(iterations),
        "fine_iteration_backends": iteration_backends,
        "teacher_backend": str(group.get("teacher_backend", "unknown")),
        "teacher_top1_candidate_id": int(group["teacher_top1_candidate_id"]),
        "teacher_margin": teacher_margin,
        "teacher_order_candidate_ids": arrays["teacher_order_candidate_ids"].tolist(),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
    }
    return arrays, summary

  @staticmethod
  def _candidate_array(
      value: Any,
      candidate_count: int,
      name: str,
      trailing_shape: tuple[int, ...] | None = None,
  ) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
      raise ValueError(f"{name} must have a candidate dimension")
    if array.shape[0] != candidate_count:
      raise ValueError(f"{name} candidate dimension is {array.shape[0]}, expected {candidate_count}")
    if trailing_shape is not None and tuple(array.shape[1:]) != trailing_shape:
      raise ValueError(f"{name} shape is {array.shape}, expected {(candidate_count, *trailing_shape)}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
      raise ValueError(f"{name} must be finite numeric data")
    return np.ascontiguousarray(array)

  @staticmethod
  def _write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
      json.dump(_json_safe(value), output, indent=2, ensure_ascii=False, allow_nan=False)
      output.write("\n")
    os.replace(temporary_path, path)

  @staticmethod
  def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output:
      np.savez_compressed(output, **arrays)
    os.replace(temporary_path, path)
