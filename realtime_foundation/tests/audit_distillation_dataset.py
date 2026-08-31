from __future__ import annotations

# All paths are relative to the repository root. Edit these two values when needed.
INPUT_ROOT_RELATIVE = "realtime_foundation/outputs/distillation_capture"
OUTPUT_DIR_RELATIVE = "realtime_foundation/outputs/distillation_audit"

EXPECTED_CANDIDATE_COUNT = 5
EXPECTED_SHARED_FEATURE_DIM = 512
TIE_MARGIN_EPSILON = 1e-6
POSE_RTOL = 1e-5
POSE_ATOL = 1e-6
RIGID_TRANSFORM_TOLERANCE = 1e-3
HISTOGRAM_BIN_COUNT = 30

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPOSITORY_ROOT / INPUT_ROOT_RELATIVE
OUTPUT_DIR = REPOSITORY_ROOT / OUTPUT_DIR_RELATIVE
ITERATION_KEY = re.compile(r"^fine_iter_(\d{2})__shared_feature$")


@dataclass
class Issue:
  severity: str
  run_id: str
  sample_id: str
  check_name: str
  expected: Any
  actual: Any
  message: str


@dataclass
class RunContext:
  run_dir: Path
  run_id: str
  metadata: dict[str, Any]
  compatibility_signature: str
  compatibility_payload: dict[str, Any]


def json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple, set)):
    return [json_safe(item) for item in value]
  if isinstance(value, np.ndarray):
    return json_safe(value.tolist())
  if isinstance(value, np.generic):
    return json_safe(value.item())
  if isinstance(value, Path):
    return value.as_posix()
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def canonical_hash(value: Any) -> str:
  encoded = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def nested_get(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
  current: Any = value
  for key in keys:
    if not isinstance(current, dict) or key not in current:
      return default
    current = current[key]
  return current


def read_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as input_file:
    value = json.load(input_file)
  if not isinstance(value, dict):
    raise ValueError(f"Expected JSON object, got {type(value).__name__}")
  return value


def add_issue(
    issues: list[Issue],
    severity: str,
    run_id: str,
    sample_id: str,
    check_name: str,
    expected: Any,
    actual: Any,
    message: str,
) -> None:
  issues.append(Issue(severity, run_id, sample_id, check_name, expected, actual, message))


def artifact_sha(metadata: dict[str, Any], *keys: str) -> Any:
  value = nested_get(metadata, *keys, default={})
  return value.get("sha256") if isinstance(value, dict) else None


def build_compatibility_payload(metadata: dict[str, Any]) -> dict[str, Any]:
  version = metadata.get("version_metadata", {})
  implementation = version.get("implementation", {}) if isinstance(version, dict) else {}
  implementation_hashes = {
      path: artifact.get("sha256")
      for path, artifact in sorted(implementation.items())
      if isinstance(artifact, dict)
  }
  return {
      "schema_version": metadata.get("schema_version"),
      "feature_version": metadata.get("feature_version"),
      "expected_candidate_count": metadata.get("expected_candidate_count"),
      "pipeline_config_sha256": nested_get(version, "pipeline_config_sha256"),
      "refiner_input_sizes": nested_get(version, "pipeline_config", "refiner_input_sizes"),
      "mesh_sha256": artifact_sha(version, "mesh"),
      "refiner_checkpoint_sha256": artifact_sha(version, "refiner", "checkpoint"),
      "refiner_config_sha256": artifact_sha(version, "refiner", "config"),
      "scorer_checkpoint_sha256": artifact_sha(version, "scorer", "checkpoint"),
      "scorer_config_sha256": artifact_sha(version, "scorer", "config"),
      "refiner_tensorrt_engines": nested_get(version, "refiner", "tensorrt_engines"),
      "scorer_tensorrt_engines": nested_get(version, "scorer", "tensorrt_engines"),
      "implementation_hashes": implementation_hashes,
  }


def discover_runs(input_root: Path, issues: list[Issue]) -> list[RunContext]:
  if not input_root.exists():
    add_issue(
        issues, "error", "", "", "input_root", "existing directory", input_root,
        "Distillation input root does not exist.",
    )
    return []

  metadata_paths = sorted(input_root.rglob("dataset_metadata.json"))
  if not metadata_paths:
    add_issue(
        issues, "error", "", "", "run_discovery", "at least one dataset_metadata.json", 0,
        "No distillation runs were found.",
    )
    return []

  runs: list[RunContext] = []
  for metadata_path in metadata_paths:
    try:
      metadata = read_json(metadata_path)
    except Exception as error:
      add_issue(
          issues, "error", metadata_path.parent.name, "", "dataset_metadata_json", "valid JSON object",
          f"{type(error).__name__}: {error}", "Failed to parse dataset metadata.",
      )
      continue
    run_id = str(metadata.get("run_id") or metadata_path.parent.name)
    payload = build_compatibility_payload(metadata)
    runs.append(RunContext(metadata_path.parent, run_id, metadata, canonical_hash(payload), payload))
  return runs


def load_manifest(run: RunContext, issues: list[Issue]) -> list[dict[str, Any]]:
  manifest_path = run.run_dir / "manifest.jsonl"
  if not manifest_path.is_file():
    add_issue(
        issues, "error", run.run_id, "", "manifest_exists", "manifest.jsonl", "missing",
        "Run manifest is missing.",
    )
    return []

  rows: list[dict[str, Any]] = []
  with manifest_path.open("r", encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, start=1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
        if not isinstance(row, dict):
          raise ValueError("manifest row is not a JSON object")
        rows.append(row)
      except Exception as error:
        add_issue(
            issues, "error", run.run_id, "", "manifest_json", "valid JSON object",
            f"line {line_number}: {type(error).__name__}: {error}", "Failed to parse manifest row.",
        )
  return rows


def check_numeric_array(
    name: str,
    array: np.ndarray,
    expected_shape: tuple[int, ...] | tuple[tuple[int, ...], ...],
    run_id: str,
    sample_id: str,
    issues: list[Issue],
) -> bool:
  valid = True
  allowed_shapes = expected_shape if expected_shape and isinstance(expected_shape[0], tuple) else (expected_shape,)
  if tuple(array.shape) not in allowed_shapes:
    add_issue(
        issues, "error", run_id, sample_id, f"shape:{name}", allowed_shapes, tuple(array.shape),
        f"Unexpected shape for {name}.",
    )
    valid = False
  if not np.issubdtype(array.dtype, np.number):
    add_issue(
        issues, "error", run_id, sample_id, f"dtype:{name}", "numeric", str(array.dtype),
        f"Array {name} is not numeric.",
    )
    return False
  if not np.isfinite(array).all():
    add_issue(
        issues, "error", run_id, sample_id, f"finite:{name}", "all finite", int(np.size(array) - np.isfinite(array).sum()),
        f"Array {name} contains NaN or Inf.",
    )
    valid = False
  return valid


def rigid_transform_error(matrices: np.ndarray) -> tuple[float, float, float]:
  rotations = matrices[..., :3, :3]
  identities = np.eye(3, dtype=np.float64)
  orthogonality = float(np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - identities)))
  determinants = np.linalg.det(rotations)
  determinant_error = float(np.max(np.abs(determinants - 1.0)))
  homogeneous_error = float(np.max(np.abs(matrices[..., 3, :] - np.array([0.0, 0.0, 0.0, 1.0]))))
  return orthogonality, determinant_error, homogeneous_error


def content_fingerprint(arrays: dict[str, np.ndarray]) -> str:
  digest = hashlib.sha256()
  for name in ("candidate_ids", "teacher_raw_logits", "final_poses", "fine_iter_01__shared_feature"):
    if name not in arrays:
      continue
    array = np.ascontiguousarray(arrays[name])
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
  return digest.hexdigest()


def audit_group(
    run: RunContext,
    json_path: Path,
    npz_path: Path,
    issues: list[Issue],
) -> dict[str, Any]:
  sample_id = json_path.stem
  issue_start = len(issues)
  row: dict[str, Any] = {
      "run_id": run.run_id,
      "sample_id": sample_id,
      "sequence_id": None,
      "frame_id": None,
      "timestamp": None,
      "valid": False,
      "teacher_margin": None,
      "teacher_top1_position": None,
      "input_size": None,
      "shared_feature_dim": None,
      "teacher_backend": None,
      "fine_backends": None,
      "content_sha256": None,
      "error_count": 0,
      "warning_count": 0,
  }

  try:
    group_metadata = read_json(json_path)
  except Exception as error:
    add_issue(
        issues, "error", run.run_id, sample_id, "group_json", "valid JSON object",
        f"{type(error).__name__}: {error}", "Failed to parse group metadata.",
    )
    return finish_group_row(row, issues, issue_start)

  sample_id = str(group_metadata.get("sample_id") or sample_id)
  row.update({
      "sample_id": sample_id,
      "sequence_id": group_metadata.get("sequence_id") or run.metadata.get("sequence_id"),
      "frame_id": group_metadata.get("frame_id"),
      "timestamp": group_metadata.get("timestamp"),
      "teacher_backend": group_metadata.get("teacher_backend"),
      "fine_backends": ";".join(map(str, group_metadata.get("fine_iteration_backends", []))),
  })

  if sample_id != json_path.stem:
    add_issue(
        issues, "error", run.run_id, sample_id, "sample_id_filename", json_path.stem, sample_id,
        "sample_id does not match the metadata filename.",
    )
  if str(group_metadata.get("run_id")) != run.run_id:
    add_issue(
        issues, "error", run.run_id, sample_id, "group_run_id", run.run_id, group_metadata.get("run_id"),
        "Group run_id does not match dataset metadata.",
    )
  if str(group_metadata.get("feature_version")) != str(run.metadata.get("feature_version")):
    add_issue(
        issues, "error", run.run_id, sample_id, "feature_version", run.metadata.get("feature_version"),
        group_metadata.get("feature_version"), "Group feature version differs from the run.",
    )
  if group_metadata.get("array_file") != npz_path.name:
    add_issue(
        issues, "error", run.run_id, sample_id, "array_file", npz_path.name, group_metadata.get("array_file"),
        "Group metadata points to a different NPZ file.",
    )

  try:
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {name: archive[name] for name in archive.files}
  except Exception as error:
    add_issue(
        issues, "error", run.run_id, sample_id, "group_npz", "readable NPZ",
        f"{type(error).__name__}: {error}", "Failed to load group arrays.",
    )
    return finish_group_row(row, issues, issue_start)

  declared_shapes = group_metadata.get("array_shapes", {})
  declared_dtypes = group_metadata.get("array_dtypes", {})
  for name, array in arrays.items():
    if name in declared_shapes and list(array.shape) != declared_shapes[name]:
      add_issue(
          issues, "error", run.run_id, sample_id, f"declared_shape:{name}", list(array.shape),
          declared_shapes[name], "Persisted array shape differs from group metadata.",
      )
    if name in declared_dtypes and str(array.dtype) != declared_dtypes[name]:
      add_issue(
          issues, "error", run.run_id, sample_id, f"declared_dtype:{name}", str(array.dtype),
          declared_dtypes[name], "Persisted array dtype differs from group metadata.",
      )

  expected_n = int(run.metadata.get("expected_candidate_count", EXPECTED_CANDIDATE_COUNT))
  if expected_n != EXPECTED_CANDIDATE_COUNT:
    add_issue(
        issues, "warning", run.run_id, sample_id, "configured_candidate_count", EXPECTED_CANDIDATE_COUNT,
        expected_n, "Run uses a non-default candidate count.",
    )

  required_shapes: dict[str, tuple[int, ...] | tuple[tuple[int, ...], ...]] = {
      "candidate_ids": (expected_n,),
      "teacher_raw_logits": (expected_n,),
      "teacher_public_scores": (expected_n,),
      "teacher_order_positions": (expected_n,),
      "teacher_order_candidate_ids": (expected_n,),
      "final_poses": (expected_n, 4, 4),
  }
  for name, expected_shape in required_shapes.items():
    if name not in arrays:
      add_issue(
          issues, "error", run.run_id, sample_id, f"required_array:{name}", "present", "missing",
          f"Required array {name} is missing.",
      )
    else:
      check_numeric_array(name, arrays[name], expected_shape, run.run_id, sample_id, issues)

  iteration_numbers = sorted(
      int(match.group(1))
      for name in arrays
      if (match := ITERATION_KEY.match(name)) is not None
  )
  if not iteration_numbers:
    add_issue(
        issues, "error", run.run_id, sample_id, "fine_iterations", "at least one", 0,
        "No Fine Refiner iteration was found.",
    )
  elif iteration_numbers != list(range(1, max(iteration_numbers) + 1)):
    add_issue(
        issues, "error", run.run_id, sample_id, "fine_iteration_sequence", "contiguous from 1",
        iteration_numbers, "Fine iteration indices are not contiguous.",
    )

  for iteration in iteration_numbers:
    prefix = f"fine_iter_{iteration:02d}"
    iteration_shapes: dict[str, tuple[int, ...] | tuple[tuple[int, ...], ...]] = {
        f"{prefix}__poses_before": (expected_n, 4, 4),
        f"{prefix}__poses_after": (expected_n, 4, 4),
        f"{prefix}__raw_trans": (expected_n, 3),
        f"{prefix}__raw_rot": ((expected_n, 3), (expected_n, 6)),
        f"{prefix}__trans_applied": (expected_n, 3),
        f"{prefix}__rot_applied": (expected_n, 3, 3),
        f"{prefix}__shared_feature": (expected_n, EXPECTED_SHARED_FEATURE_DIM),
        f"{prefix}__input_size": (2,),
    }
    for name, expected_shape in iteration_shapes.items():
      if name not in arrays:
        add_issue(
            issues, "error", run.run_id, sample_id, f"required_array:{name}", "present", "missing",
            f"Required array {name} is missing.",
        )
      else:
        check_numeric_array(name, arrays[name], expected_shape, run.run_id, sample_id, issues)

  if "candidate_ids" in arrays and arrays["candidate_ids"].shape == (expected_n,):
    candidate_ids = arrays["candidate_ids"].astype(np.int64, copy=False)
    if len(np.unique(candidate_ids)) != expected_n:
      add_issue(
          issues, "error", run.run_id, sample_id, "candidate_id_unique", expected_n,
          len(np.unique(candidate_ids)), "Candidate IDs are not unique within the group.",
      )
  else:
    candidate_ids = None

  logits = arrays.get("teacher_raw_logits")
  if logits is not None and logits.shape == (expected_n,) and np.isfinite(logits).all():
    expected_order = np.argsort(logits)[::-1]
    margin = float(logits[expected_order[0]] - logits[expected_order[1]]) if expected_n > 1 else 0.0
    row["teacher_margin"] = margin
    row["teacher_top1_position"] = int(expected_order[0])
    if "teacher_order_positions" in arrays and not np.array_equal(arrays["teacher_order_positions"], expected_order):
      add_issue(
          issues, "error", run.run_id, sample_id, "teacher_order", expected_order,
          arrays["teacher_order_positions"], "Stored Teacher order does not match raw logits.",
      )
    if candidate_ids is not None and "teacher_order_candidate_ids" in arrays:
      expected_ids = candidate_ids[expected_order]
      if not np.array_equal(arrays["teacher_order_candidate_ids"], expected_ids):
        add_issue(
            issues, "error", run.run_id, sample_id, "teacher_order_candidate_ids", expected_ids,
            arrays["teacher_order_candidate_ids"], "Stored ordered candidate IDs do not match raw logits.",
        )
      if group_metadata.get("teacher_top1_candidate_id") != int(expected_ids[0]):
        add_issue(
            issues, "error", run.run_id, sample_id, "teacher_top1_candidate_id", int(expected_ids[0]),
            group_metadata.get("teacher_top1_candidate_id"), "Stored Teacher Top1 candidate is inconsistent.",
        )
    stored_margin = group_metadata.get("teacher_margin")
    if stored_margin is None or not np.isclose(float(stored_margin), margin, rtol=1e-5, atol=1e-7):
      add_issue(
          issues, "error", run.run_id, sample_id, "teacher_margin", margin, stored_margin,
          "Stored Teacher margin does not match raw logits.",
      )
    if margin <= TIE_MARGIN_EPSILON:
      add_issue(
          issues, "warning", run.run_id, sample_id, "teacher_tie", f"> {TIE_MARGIN_EPSILON}", margin,
          "Teacher Top1 and Top2 logits are tied or nearly tied.",
      )

  public_scores = arrays.get("teacher_public_scores")
  if logits is not None and public_scores is not None and logits.shape == public_scores.shape:
    if not np.allclose(public_scores, logits + 100.0, rtol=1e-5, atol=1e-5):
      add_issue(
          issues, "error", run.run_id, sample_id, "teacher_public_scores", "raw logits + 100",
          float(np.max(np.abs(public_scores - logits - 100.0))), "Public scores do not match raw logits plus 100.",
      )

  if iteration_numbers and "final_poses" in arrays:
    last_prefix = f"fine_iter_{iteration_numbers[-1]:02d}"
    poses_after = arrays.get(f"{last_prefix}__poses_after")
    final_poses = arrays["final_poses"]
    if poses_after is not None and poses_after.shape == final_poses.shape:
      maximum_pose_difference = float(np.max(np.abs(poses_after - final_poses)))
      if not np.allclose(poses_after, final_poses, rtol=POSE_RTOL, atol=POSE_ATOL):
        add_issue(
            issues, "error", run.run_id, sample_id, "fine_pose_alignment", "last Fine poses_after ~= final_poses",
            maximum_pose_difference, "Final Fine poses do not match Final Scorer input poses.",
        )

  for pose_name in ("final_poses",):
    pose_array = arrays.get(pose_name)
    if pose_array is not None and pose_array.shape == (expected_n, 4, 4) and np.isfinite(pose_array).all():
      orthogonality, determinant, homogeneous = rigid_transform_error(pose_array.astype(np.float64))
      if max(orthogonality, determinant, homogeneous) > RIGID_TRANSFORM_TOLERANCE:
        add_issue(
            issues, "warning", run.run_id, sample_id, f"rigid_transform:{pose_name}",
            f"max error <= {RIGID_TRANSFORM_TOLERANCE}",
            {"orthogonality": orthogonality, "determinant": determinant, "homogeneous": homogeneous},
            "Pose matrix deviates from a rigid homogeneous transform.",
        )

  for iteration in iteration_numbers:
    input_size = arrays.get(f"fine_iter_{iteration:02d}__input_size")
    shared_feature = arrays.get(f"fine_iter_{iteration:02d}__shared_feature")
    if input_size is not None and input_size.shape == (2,):
      row["input_size"] = "x".join(str(int(value)) for value in input_size)
    if shared_feature is not None and shared_feature.ndim == 2:
      row["shared_feature_dim"] = int(shared_feature.shape[1])
      variances = np.var(shared_feature.astype(np.float64), axis=0)
      near_zero_channels = int(np.sum(variances <= 1e-12))
      if near_zero_channels > 0:
        add_issue(
            issues, "warning", run.run_id, sample_id, "shared_feature_variance", "all channels > 1e-12",
            near_zero_channels, "Shared Feature contains channels with near-zero within-group variance.",
        )

  row["content_sha256"] = content_fingerprint(arrays)
  return finish_group_row(row, issues, issue_start)


def finish_group_row(row: dict[str, Any], issues: list[Issue], issue_start: int) -> dict[str, Any]:
  group_issues = issues[issue_start:]
  row["error_count"] = sum(issue.severity == "error" for issue in group_issues)
  row["warning_count"] = sum(issue.severity == "warning" for issue in group_issues)
  row["valid"] = row["error_count"] == 0
  return row


def audit_run(run: RunContext, issues: list[Issue]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  if run.metadata.get("schema_version") != 1:
    add_issue(
        issues, "error", run.run_id, "", "schema_version", 1, run.metadata.get("schema_version"),
        "Unsupported or missing dataset schema version.",
    )
  if not run.metadata.get("feature_version"):
    add_issue(
        issues, "error", run.run_id, "", "feature_version", "non-empty feature version",
        run.metadata.get("feature_version"), "Dataset feature version is missing.",
    )
  if run.metadata.get("sequence_id") is None:
    add_issue(
        issues, "warning", run.run_id, "", "sequence_id", "stable rosbag sequence ID", None,
        "Run has no sequence_id; cross-run duplicate detection will rely on content hashes.",
    )
  required_version_fields = {
      "pipeline_config_sha256": nested_get(run.metadata, "version_metadata", "pipeline_config_sha256"),
      "mesh_sha256": artifact_sha(run.metadata.get("version_metadata", {}), "mesh"),
      "refiner_checkpoint_sha256": artifact_sha(run.metadata.get("version_metadata", {}), "refiner", "checkpoint"),
      "scorer_checkpoint_sha256": artifact_sha(run.metadata.get("version_metadata", {}), "scorer", "checkpoint"),
  }
  for field_name, field_value in required_version_fields.items():
    if not field_value:
      add_issue(
          issues, "warning", run.run_id, "", f"version_metadata:{field_name}", "recorded SHA-256",
          field_value, "Version metadata is incomplete, reducing compatibility-check confidence.",
      )

  groups_dir = run.run_dir / "groups"
  if not groups_dir.is_dir():
    add_issue(
        issues, "error", run.run_id, "", "groups_directory", "existing groups directory", "missing",
        "Run groups directory is missing.",
    )
    return make_run_summary(run, [], [], issues), []

  manifest_rows = load_manifest(run, issues)
  json_paths = {path.stem: path for path in groups_dir.glob("*.json")}
  npz_paths = {path.stem: path for path in groups_dir.glob("*.npz")}
  manifest_ids = [str(row.get("sample_id")) for row in manifest_rows if row.get("sample_id") is not None]

  for sample_id in sorted(set(json_paths) - set(npz_paths)):
    add_issue(
        issues, "error", run.run_id, sample_id, "npz_pair", "matching NPZ", "missing",
        "Group JSON has no matching NPZ.",
    )
  for sample_id in sorted(set(npz_paths) - set(json_paths)):
    add_issue(
        issues, "error", run.run_id, sample_id, "json_pair", "matching JSON", "missing",
        "Group NPZ has no matching JSON.",
    )
  for sample_id in sorted(set(manifest_ids) - set(json_paths)):
    add_issue(
        issues, "error", run.run_id, sample_id, "manifest_group", "group JSON/NPZ", "missing",
        "Manifest references a missing group.",
    )
  for sample_id in sorted(set(json_paths) - set(manifest_ids)):
    add_issue(
        issues, "warning", run.run_id, sample_id, "unlisted_group", "listed in manifest", "not listed",
        "Group exists on disk but is absent from the manifest.",
    )
  if len(manifest_ids) != len(set(manifest_ids)):
    add_issue(
        issues, "error", run.run_id, "", "manifest_sample_ids", "unique", len(manifest_ids) - len(set(manifest_ids)),
        "Manifest contains duplicate sample IDs.",
    )

  group_rows = [
      audit_group(run, json_paths[sample_id], npz_paths[sample_id], issues)
      for sample_id in sorted(set(json_paths) & set(npz_paths))
  ]
  input_sizes = {row["input_size"] for row in group_rows if row.get("input_size")}
  feature_dims = {row["shared_feature_dim"] for row in group_rows if row.get("shared_feature_dim") is not None}
  teacher_backends = {row["teacher_backend"] for row in group_rows if row.get("teacher_backend")}
  if len(input_sizes) > 1:
    add_issue(
        issues, "error", run.run_id, "", "run_input_sizes", "one Fine input size",
        sorted(input_sizes), "Fine Refiner input size changes within one run.",
    )
  if len(feature_dims) > 1:
    add_issue(
        issues, "error", run.run_id, "", "run_shared_feature_dims", "one feature dimension",
        sorted(feature_dims), "Shared Feature dimension changes within one run.",
    )
  if len(teacher_backends) > 1:
    add_issue(
        issues, "warning", run.run_id, "", "run_teacher_backends", "one Teacher backend",
        sorted(teacher_backends), "Teacher backend changes within one run.",
    )
  return make_run_summary(run, manifest_rows, group_rows, issues), group_rows


def make_run_summary(
    run: RunContext,
    manifest_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    issues: list[Issue],
) -> dict[str, Any]:
  run_issues = [issue for issue in issues if issue.run_id == run.run_id]
  margins = [
      float(row["teacher_margin"])
      for row in group_rows
      if row.get("teacher_margin") is not None and row.get("valid")
  ]
  return {
      "run_id": run.run_id,
      "run_dir": run.run_dir.relative_to(REPOSITORY_ROOT).as_posix(),
      "sequence_id": run.metadata.get("sequence_id"),
      "schema_version": run.metadata.get("schema_version"),
      "feature_version": run.metadata.get("feature_version"),
      "expected_candidate_count": run.metadata.get("expected_candidate_count"),
      "manifest_groups": len(manifest_rows),
      "disk_groups": len(group_rows),
      "valid_groups": sum(bool(row["valid"]) for row in group_rows),
      "invalid_groups": sum(not bool(row["valid"]) for row in group_rows),
      "error_count": sum(issue.severity == "error" for issue in run_issues),
      "warning_count": sum(issue.severity == "warning" for issue in run_issues),
      "teacher_backends": sorted({str(row["teacher_backend"]) for row in group_rows}),
      "fine_backends": sorted({str(row["fine_backends"]) for row in group_rows}),
      "input_sizes": sorted({str(row["input_size"]) for row in group_rows}),
      "shared_feature_dims": sorted({str(row["shared_feature_dim"]) for row in group_rows}),
      "margin_min": min(margins) if margins else None,
      "margin_median": float(np.median(margins)) if margins else None,
      "margin_max": max(margins) if margins else None,
      "compatibility_signature": run.compatibility_signature,
  }


def add_duplicate_issues(group_rows: list[dict[str, Any]], issues: list[Issue]) -> list[dict[str, Any]]:
  duplicates: list[dict[str, Any]] = []
  indexes: dict[str, dict[Any, list[dict[str, Any]]]] = {
      "sequence_frame": {},
      "sequence_timestamp": {},
      "content_sha256": {},
  }
  for row in group_rows:
    sequence = row.get("sequence_id")
    if sequence is not None and row.get("frame_id") is not None:
      indexes["sequence_frame"].setdefault((sequence, row["frame_id"]), []).append(row)
    if sequence is not None and row.get("timestamp") is not None:
      indexes["sequence_timestamp"].setdefault((sequence, row["timestamp"]), []).append(row)
    if row.get("content_sha256"):
      indexes["content_sha256"].setdefault(row["content_sha256"], []).append(row)

  seen_groups: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
  for check_name, index in indexes.items():
    for key, rows in index.items():
      identities = tuple(sorted((str(row["run_id"]), str(row["sample_id"])) for row in rows))
      dedupe_key = (check_name, identities)
      if len(rows) < 2 or dedupe_key in seen_groups:
        continue
      seen_groups.add(dedupe_key)
      duplicate = {
          "check_name": check_name,
          "key": str(key),
          "samples": [f"{row['run_id']}/{row['sample_id']}" for row in rows],
      }
      duplicates.append(duplicate)
      for row in rows:
        add_issue(
            issues, "warning", str(row["run_id"]), str(row["sample_id"]), f"duplicate:{check_name}",
            "unique", str(key), "Sample appears in a duplicate group.",
        )
  return duplicates


def distribution_summary(values: list[float]) -> dict[str, Any]:
  if not values:
    return {"count": 0}
  array = np.asarray(values, dtype=np.float64)
  quantiles = (0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99)
  return {
      "count": int(array.size),
      "min": float(array.min()),
      "max": float(array.max()),
      "mean": float(array.mean()),
      "std": float(array.std()),
      "median": float(np.median(array)),
      "tie_count": int(np.sum(array <= TIE_MARGIN_EPSILON)),
      "tie_fraction": float(np.mean(array <= TIE_MARGIN_EPSILON)),
      "quantiles": {f"p{int(q * 100):02d}": float(np.quantile(array, q)) for q in quantiles},
  }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
  if fieldnames is None:
    fieldnames = sorted({key for row in rows for key in row})
  with path.open("w", encoding="utf-8", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow({key: json.dumps(json_safe(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def write_json(path: Path, value: Any) -> None:
  with path.open("w", encoding="utf-8") as output:
    json.dump(json_safe(value), output, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True)
    output.write("\n")


def write_issues(path: Path, issues: list[Issue]) -> None:
  with path.open("w", encoding="utf-8") as output:
    for issue in issues:
      output.write(json.dumps(json_safe(asdict(issue)), ensure_ascii=False, sort_keys=True))
      output.write("\n")


def write_margin_histogram(output_dir: Path, margins: list[float]) -> None:
  csv_path = output_dir / "teacher_margin_histogram.csv"
  svg_path = output_dir / "teacher_margin_histogram.svg"
  if not margins:
    write_csv(csv_path, [], ["bin_left", "bin_right", "count"])
    svg_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"120\"><text x=\"20\" y=\"60\">No Teacher margins</text></svg>\n", encoding="utf-8")
    return

  counts, edges = np.histogram(np.asarray(margins, dtype=np.float64), bins=HISTOGRAM_BIN_COUNT)
  histogram_rows = [
      {"bin_left": float(edges[index]), "bin_right": float(edges[index + 1]), "count": int(count)}
      for index, count in enumerate(counts)
  ]
  write_csv(csv_path, histogram_rows)

  width, height = 900, 420
  left, right, top, bottom = 70, 20, 30, 70
  plot_width, plot_height = width - left - right, height - top - bottom
  maximum = max(int(counts.max()), 1)
  bar_width = plot_width / max(len(counts), 1)
  bars = []
  for index, count in enumerate(counts):
    bar_height = plot_height * int(count) / maximum
    x = left + index * bar_width
    y = top + plot_height - bar_height
    bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 1, 1):.2f}" height="{bar_height:.2f}" fill="#4c78a8"/>')
  minimum_label = f"{edges[0]:.6g}"
  maximum_label = f"{edges[-1]:.6g}"
  svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="20" text-anchor="middle" font-family="sans-serif" font-size="16">Teacher Margin Histogram</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>
{''.join(bars)}
<text x="{left}" y="{height - 42}" text-anchor="start" font-family="sans-serif" font-size="12">{minimum_label}</text>
<text x="{left + plot_width}" y="{height - 42}" text-anchor="end" font-family="sans-serif" font-size="12">{maximum_label}</text>
<text x="{left + plot_width / 2}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="13">Teacher Top1 - Top2 Logit Margin</text>
<text x="18" y="{top + plot_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 {top + plot_height / 2})">Group count</text>
</svg>
'''
  svg_path.write_text(svg, encoding="utf-8")


def main() -> int:
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  issues: list[Issue] = []
  runs = discover_runs(INPUT_ROOT, issues)
  run_rows: list[dict[str, Any]] = []
  group_rows: list[dict[str, Any]] = []

  for run in runs:
    run_row, rows = audit_run(run, issues)
    run_rows.append(run_row)
    group_rows.extend(rows)

  duplicates = add_duplicate_issues(group_rows, issues)
  for run_row in run_rows:
    run_issues = [issue for issue in issues if issue.run_id == run_row["run_id"]]
    run_row["error_count"] = sum(issue.severity == "error" for issue in run_issues)
    run_row["warning_count"] = sum(issue.severity == "warning" for issue in run_issues)
  margins = [
      float(row["teacher_margin"])
      for row in group_rows
      if row.get("teacher_margin") is not None and row.get("valid")
  ]
  margin_by_run = {
      run.run_id: distribution_summary([
          float(row["teacher_margin"])
          for row in group_rows
          if row["run_id"] == run.run_id and row.get("teacher_margin") is not None and row.get("valid")
      ])
      for run in runs
  }

  compatibility_groups: dict[str, dict[str, Any]] = {}
  for run in runs:
    group = compatibility_groups.setdefault(run.compatibility_signature, {
        "signature": run.compatibility_signature,
        "runs": [],
        "payload": run.compatibility_payload,
    })
    group["runs"].append(run.run_id)
  compatibility_rows = [
      {
          "compatibility_group": f"G{index:02d}",
          "signature": value["signature"],
          "run_count": len(value["runs"]),
          "runs": value["runs"],
          "payload": value["payload"],
      }
      for index, value in enumerate(sorted(compatibility_groups.values(), key=lambda item: item["signature"]), start=1)
  ]

  error_count = sum(issue.severity == "error" for issue in issues)
  warning_count = sum(issue.severity == "warning" for issue in issues)
  summary = {
      "input_root": INPUT_ROOT_RELATIVE,
      "output_dir": OUTPUT_DIR_RELATIVE,
      "run_count": len(runs),
      "group_count": len(group_rows),
      "valid_group_count": sum(bool(row["valid"]) for row in group_rows),
      "invalid_group_count": sum(not bool(row["valid"]) for row in group_rows),
      "error_count": error_count,
      "warning_count": warning_count,
      "duplicate_set_count": len(duplicates),
      "compatibility_group_count": len(compatibility_rows),
      "teacher_margin": distribution_summary(margins),
      "teacher_margin_by_run": margin_by_run,
      "note": "Global margin quantiles are descriptive only. Training split thresholds must be computed from training runs only.",
  }

  write_json(OUTPUT_DIR / "audit_summary.json", summary)
  write_csv(OUTPUT_DIR / "run_summary.csv", run_rows)
  write_csv(OUTPUT_DIR / "group_summary.csv", group_rows)
  write_issues(OUTPUT_DIR / "issues.jsonl", issues)
  write_json(OUTPUT_DIR / "compatibility_groups.json", compatibility_rows)
  write_csv(OUTPUT_DIR / "compatibility_matrix.csv", compatibility_rows)
  write_csv(OUTPUT_DIR / "duplicate_groups.csv", duplicates, ["check_name", "key", "samples"])
  margin_rows = [{"run_id": "ALL", **distribution_summary(margins)}]
  margin_rows.extend({"run_id": run_id, **stats} for run_id, stats in margin_by_run.items())
  write_csv(OUTPUT_DIR / "teacher_margin_summary.csv", margin_rows)
  write_margin_histogram(OUTPUT_DIR, margins)

  print(f"Input:  {INPUT_ROOT}")
  print(f"Output: {OUTPUT_DIR}")
  print(f"Runs: {len(runs)}, groups: {len(group_rows)}, valid: {summary['valid_group_count']}, invalid: {summary['invalid_group_count']}")
  print(f"Errors: {error_count}, warnings: {warning_count}, compatibility groups: {len(compatibility_rows)}")
  return 1 if error_count else 0


if __name__ == "__main__":
  sys.exit(main())
