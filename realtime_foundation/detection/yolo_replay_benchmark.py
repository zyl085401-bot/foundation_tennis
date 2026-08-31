from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

from yolo_segmenter import YoloMaskResult, YoloSegmenter


# ==================== 可直接修改的默认路径 ====================
# 相对路径以仓库根目录为基准，也可以改成绝对路径。
BENCHMARK_CONFIG_PATH = Path("realtime_foundation/config.yaml")
BENCHMARK_INPUT_DIR = Path("realtime_foundation/outputs/frame_records/frame_data")
BENCHMARK_OUTPUT_ROOT = Path("realtime_foundation/outputs/yolo_benchmark")
# =============================================================


VISUALIZATION_MASK_ALPHA = 0.45
VISUALIZATION_JPEG_QUALITY = 95
BASELINE_MASK_COLOR_RGB = (0, 220, 80)
CURRENT_MASK_COLOR_RGB = (255, 120, 0)


SCRIPT_DIR = Path(__file__).resolve().parent
REALTIME_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REALTIME_ROOT.parent
DEFAULT_CONFIG = (
  BENCHMARK_CONFIG_PATH
  if BENCHMARK_CONFIG_PATH.is_absolute()
  else REPO_ROOT / BENCHMARK_CONFIG_PATH
)
DEFAULT_INPUT_DIR = (
  BENCHMARK_INPUT_DIR
  if BENCHMARK_INPUT_DIR.is_absolute()
  else REPO_ROOT / BENCHMARK_INPUT_DIR
)
DEFAULT_OUTPUT_ROOT = (
  BENCHMARK_OUTPUT_ROOT
  if BENCHMARK_OUTPUT_ROOT.is_absolute()
  else REPO_ROOT / BENCHMARK_OUTPUT_ROOT
)

TIMING_FIELDS = {
    "yolo_total_ms": "total",
    "model_predict_ms": "model_predict",
    "preprocess_ms": "model_preprocess",
    "tensorrt_inference_ms": "model_inference",
    "ultralytics_postprocess_ms": "model_postprocess",
    "framework_overhead_ms": "model_framework_overhead",
    "tensor_to_cpu_ms": "tensor_to_cpu",
    "mask_clean_ms": "mask_resize_clean",
    "select_best_mask_ms": "select_best_mask",
}


@dataclass(frozen=True)
class FrameInput:
  key: str
  relative_path: str
  sha256: str
  rgb: np.ndarray


@dataclass(frozen=True)
class PredictionObservation:
  result: YoloMaskResult | None
  timing: dict[str, Any]
  wall_ms: float
  candidate_count: int
  execution_path: str


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Benchmark YOLO segmentation with recorded RGB frames and save comparable reference outputs."
  )
  parser.add_argument("--mode", choices=("baseline", "compare"), default="baseline")
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
  parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
  parser.add_argument("--run-name", type=str, default=None)
  parser.add_argument("--reference", type=Path, default=None, help="Baseline run directory or manifest.json.")
  parser.add_argument("--iterations", type=int, default=None, help="Measured iterations; default 100 or reference value.")
  parser.add_argument("--warmup", type=int, default=None, help="Warmup iterations; default 10 or reference value.")
  parser.add_argument("--max-frames", type=int, default=0, help="Use at most this many sorted NPZ frames; 0 means all.")
  parser.add_argument("--progress-every", type=int, default=10)
  parser.add_argument(
      "--execution-path",
      choices=("legacy", "fast"),
      default=None,
      help="Override yolo.execution_path from config.",
  )
  parser.add_argument(
      "--profile-stages",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Synchronize CUDA around fast-path stages; use --no-profile-stages for production wall timing.",
  )
  parser.add_argument(
      "--allow-fast-fallback",
      action="store_true",
      help="Allow a requested fast path to fall back to legacy; disabled by default to prevent invalid comparisons.",
  )
  parser.add_argument(
      "--postprocess-backend",
      choices=("gpu", "cpu"),
      default=None,
      help="Override yolo.postprocess_backend from config.",
  )
  parser.add_argument(
      "--allow-engine-difference",
      action="store_true",
      help="Permit compare mode to use an engine SHA256 different from the reference; recorded in metadata.",
  )
  args = parser.parse_args()
  if args.mode == "compare" and args.reference is None:
    parser.error("--reference is required in compare mode")
  if args.mode == "baseline" and args.reference is not None:
    parser.error("--reference is only valid in compare mode")
  if args.iterations is not None and args.iterations < 1:
    parser.error("--iterations must be positive")
  if args.warmup is not None and args.warmup < 0:
    parser.error("--warmup must be non-negative")
  if args.max_frames < 0:
    parser.error("--max-frames must be non-negative")
  if args.progress_every < 0:
    parser.error("--progress-every must be non-negative")
  if args.run_name is not None and Path(args.run_name).name != args.run_name:
    parser.error("--run-name must be a single directory name")
  return args


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def resolve_cli_path(path: Path) -> Path:
  return path.expanduser().resolve()


def resolve_repo_path(value: str | os.PathLike[str]) -> Path:
  path = Path(value).expanduser()
  return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
  try:
    return path.resolve().relative_to(REPO_ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_json(value: Any) -> str:
  payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def comparison_yolo_config(yolo_config: dict[str, Any]) -> dict[str, Any]:
  runtime_only = {
      "weights",
  "int8_weights",
  "int8_io_weights",
  "precision",
  "io_precision",
  "fallback_to_fp32_io",
  "fallback_to_fp16",
      "execution_path",
      "profile_stages",
      "fallback_to_legacy",
      "postprocess_backend",
  }
  return {key: value for key, value in yolo_config.items() if key not in runtime_only}


def requested_yolo_weights(yolo_config: dict[str, Any]) -> str:
  precision = str(yolo_config.get("precision", "fp16")).strip().lower()
  if precision == "fp16":
    key = "weights"
  elif precision == "int8":
    io_precision = str(yolo_config.get("io_precision", "fp32")).strip().lower()
    if io_precision == "int8":
      key = "int8_io_weights"
    elif io_precision == "fp32":
      key = "int8_weights"
    else:
      raise ValueError(f"Unsupported YOLO I/O precision: {io_precision!r}")
  else:
    raise ValueError(f"Unsupported YOLO precision: {precision!r}")
  value = yolo_config.get(key)
  if not value:
    raise ValueError(f"Config yolo.{key} is required for precision={precision!r}")
  return str(value)


def json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(item) for item in value]
  if isinstance(value, np.generic):
    return json_safe(value.item())
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  if not path.is_file():
    raise FileNotFoundError(f"Config does not exist: {path}")
  with path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}
  if not isinstance(config, dict):
    raise ValueError(f"Config must be a YAML mapping: {path}")
  yolo_config = config.get("yolo") or {}
  if not isinstance(yolo_config, dict):
    raise ValueError(f"Config yolo section must be a mapping: {path}")
  if not yolo_config.get("weights"):
    raise ValueError(f"Config yolo.weights is required: {path}")
  return config, yolo_config


def load_reference_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
  candidate = resolve_cli_path(path)
  manifest_path = candidate / "manifest.json" if candidate.is_dir() else candidate
  if not manifest_path.is_file():
    raise FileNotFoundError(f"Reference manifest does not exist: {manifest_path}")
  with manifest_path.open("r", encoding="utf-8") as file:
    manifest = json.load(file)
  if int(manifest.get("format_version", 0)) != 1:
    raise ValueError(f"Unsupported reference manifest format: {manifest_path}")
  return manifest_path.parent, manifest


def discover_input_paths(input_dir: Path, max_frames: int) -> list[Path]:
  if not input_dir.is_dir():
    raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
  paths = sorted(path for path in input_dir.glob("*.npz") if path.is_file())
  if max_frames > 0:
    paths = paths[:max_frames]
  if not paths:
    raise RuntimeError(f"No NPZ frames found in: {input_dir}")
  return paths


def paths_from_manifest(input_dir: Path, manifest: dict[str, Any]) -> list[Path]:
  entries = manifest.get("inputs") or []
  if not entries:
    raise ValueError("Reference manifest has no inputs")
  paths = []
  for entry in entries:
    relative_path = Path(str(entry["relative_path"]))
    path = (input_dir / relative_path).resolve()
    if not path.is_file():
      raise FileNotFoundError(f"Reference input does not exist: {path}")
    paths.append(path)
  return paths


def load_frame(path: Path, input_dir: Path, index: int) -> FrameInput:
  with np.load(path, allow_pickle=False) as data:
    if "rgb" not in data:
      raise KeyError(f"Recorded frame has no 'rgb' array: {path}")
    rgb = np.asarray(data["rgb"])
  if rgb.dtype != np.uint8:
    raise ValueError(f"Expected uint8 RGB, got {rgb.dtype}: {path}")
  if rgb.ndim != 3 or rgb.shape[2] != 3:
    raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}: {path}")
  rgb = np.ascontiguousarray(rgb)
  relative_path = path.resolve().relative_to(input_dir.resolve()).as_posix()
  return FrameInput(
      key=f"{index:04d}_{path.stem}",
      relative_path=relative_path,
      sha256=sha256_file(path),
      rgb=rgb,
  )


def load_frames(paths: list[Path], input_dir: Path) -> list[FrameInput]:
  return [load_frame(path, input_dir, index) for index, path in enumerate(paths)]


def build_detector(
    yolo_config: dict[str, Any],
    weights_path: Path,
    execution_path: str,
    profile_stages: bool,
    fallback_to_legacy: bool,
    postprocess_backend: str,
) -> YoloSegmenter:
  return YoloSegmenter(
      weights=str(weights_path),
      target_class=yolo_config.get("target_class"),
      target_class_id=yolo_config.get("target_class_id"),
      conf=float(yolo_config.get("conf", 0.35)),
      imgsz=yolo_config.get("imgsz", 640),
      device=yolo_config.get("device"),
      half=bool(yolo_config.get("half", True)),
      min_mask_area=int(yolo_config.get("min_mask_area", 100)),
      morph_kernel=int(yolo_config.get("morph_kernel", 5)),
      gpu_best_mask_topk=int(yolo_config.get("gpu_best_mask_topk", 1)),
      input_is_rgb=True,
      execution_path=execution_path,
      profile_stages=profile_stages,
      fallback_to_legacy=fallback_to_legacy,
      postprocess_backend=postprocess_backend,
        int8_input_dynamic_range=float(yolo_config.get("int8_input_dynamic_range", 1.0)),
  )


def run_prediction(
    detector: YoloSegmenter,
    frame: FrameInput,
) -> PredictionObservation:
  started = time.perf_counter()
  result = detector.predict_mask(frame.rgb)
  wall_ms = (time.perf_counter() - started) * 1000.0
  candidate_count = getattr(detector, "last_candidate_count", None)
  if candidate_count is None:
    raise RuntimeError("YOLO executor did not report a candidate count")
  timing = dict(detector.last_timing)
  validate_timing(timing)
  return PredictionObservation(
      result=result,
      timing=timing,
      wall_ms=wall_ms,
      candidate_count=int(candidate_count),
      execution_path=str(getattr(detector, "last_execution_path", "unknown")),
  )


def timing_to_ms(value: Any) -> float | None:
  if value is None:
    return None
  return float(value) * 1000.0


def validate_timing(timing: dict[str, Any]) -> None:
  required_keys = (
      "total",
      "model_predict",
      "model_preprocess",
      "model_inference",
      "model_postprocess",
      "tensor_to_cpu",
      "mask_resize_clean",
      "select_best_mask",
  )
  missing = [name for name in required_keys if name not in timing]
  if missing:
    raise RuntimeError(f"YOLO timing keys are unavailable: {missing}")
  if timing["total"] is None:
    raise RuntimeError("YOLO total timing is unavailable")


def shape_text(shape: tuple[int, ...] | list[int]) -> str:
  return "x".join(str(int(value)) for value in shape)


def result_snapshot(result: YoloMaskResult | None, image_shape: tuple[int, int]) -> dict[str, Any]:
  if result is None:
    return {
        "detected": False,
        "mask": np.zeros(image_shape, dtype=np.uint8),
        "box_xyxy": np.full(4, np.nan, dtype=np.float32),
        "confidence": float("nan"),
        "class_id": -1,
        "area": 0,
    }
  mask = np.ascontiguousarray(result.mask)
  if mask.ndim != 2:
    raise ValueError(f"Expected 2D result mask, got {mask.shape}")
  return {
      "detected": True,
      "mask": mask,
      "box_xyxy": np.asarray(result.box_xyxy, dtype=np.float32).reshape(4),
      "confidence": float(result.confidence),
      "class_id": int(result.class_id),
      "area": int(result.area),
  }


def snapshot_descriptor(snapshot: dict[str, Any]) -> dict[str, Any]:
  mask = np.asarray(snapshot["mask"])
  detected = bool(snapshot["detected"])
  box = np.asarray(snapshot["box_xyxy"], dtype=np.float32).reshape(4)
  return {
      "detected": detected,
      "output_mask_shape": shape_text(mask.shape),
      "output_mask_dtype": str(mask.dtype),
      "output_mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
      "mask_area": int(snapshot["area"]),
      "class_id": int(snapshot["class_id"]) if detected else None,
      "confidence": float(snapshot["confidence"]) if detected else None,
      "box_xyxy": json.dumps(box.tolist()) if detected else None,
  }


def compare_snapshots(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
  current_detected = bool(current["detected"])
  reference_detected = bool(reference["detected"])
  if current_detected and reference_detected:
    status = "both_detected"
  elif not current_detected and not reference_detected:
    status = "both_missing"
  elif current_detected:
    status = "current_only"
  else:
    status = "reference_only"

  current_mask = np.asarray(current["mask"])
  reference_mask = np.asarray(reference["mask"])
  shape_equal = current_mask.shape == reference_mask.shape
  dtype_equal = current_mask.dtype == reference_mask.dtype
  if shape_equal:
    current_bool = current_mask.astype(bool, copy=False)
    reference_bool = reference_mask.astype(bool, copy=False)
    pixel_equal_count = int(np.count_nonzero(current_mask == reference_mask))
    pixel_total = int(current_mask.size)
    pixel_equal_ratio = pixel_equal_count / max(pixel_total, 1)
    intersection = int(np.logical_and(current_bool, reference_bool).sum())
    union = int(np.logical_or(current_bool, reference_bool).sum())
    mask_iou = 1.0 if union == 0 else intersection / union
    exact_match = bool(np.array_equal(current_mask, reference_mask))
  else:
    pixel_equal_count = 0
    pixel_total = max(int(current_mask.size), int(reference_mask.size), 1)
    pixel_equal_ratio = 0.0
    mask_iou = 0.0
    exact_match = False

  area_abs_diff = abs(int(current["area"]) - int(reference["area"]))
  area_rel_diff = area_abs_diff / max(int(reference["area"]), 1)
  class_id_equal = int(current["class_id"]) == int(reference["class_id"])
  bbox_max_abs_diff = None
  confidence_abs_diff = None
  if current_detected and reference_detected:
    current_box = np.asarray(current["box_xyxy"], dtype=np.float32).reshape(4)
    reference_box = np.asarray(reference["box_xyxy"], dtype=np.float32).reshape(4)
    bbox_max_abs_diff = float(np.max(np.abs(current_box - reference_box)))
    confidence_abs_diff = abs(float(current["confidence"]) - float(reference["confidence"]))

  return {
      "cmp_status": status,
      "cmp_detection_equal": current_detected == reference_detected,
      "cmp_exact_mask": exact_match,
      "cmp_mask_shape_equal": shape_equal,
      "cmp_mask_dtype_equal": dtype_equal,
      "cmp_pixel_equal_count": pixel_equal_count,
      "cmp_pixel_total": pixel_total,
      "cmp_pixel_equal_ratio": pixel_equal_ratio,
      "cmp_mask_iou": mask_iou,
      "cmp_area_abs_diff": area_abs_diff,
      "cmp_area_rel_diff": area_rel_diff,
      "cmp_bbox_max_abs_diff": bbox_max_abs_diff,
      "cmp_confidence_abs_diff": confidence_abs_diff,
      "cmp_class_id_equal": class_id_equal,
  }


def build_sample_row(
    phase: str,
    iteration: int,
    frame: FrameInput,
    observation: PredictionObservation,
    snapshot: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
  row = {
      "phase": phase,
      "iteration": iteration,
      "frame_key": frame.key,
      "input_relative_path": frame.relative_path,
      "input_sha256": frame.sha256,
      "input_shape": shape_text(frame.rgb.shape),
      "input_dtype": str(frame.rgb.dtype),
      "input_c_contiguous": bool(frame.rgb.flags.c_contiguous),
      "wall_ms": observation.wall_ms,
      "candidate_count": observation.candidate_count,
      "execution_path": observation.execution_path,
  }
  for output_name, timing_name in TIMING_FIELDS.items():
    row[output_name] = timing_to_ms(observation.timing.get(timing_name))
  row.update(snapshot_descriptor(snapshot))
  if reference is not None:
    row.update(compare_snapshots(snapshot, reference))
  return row


def save_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
  np.savez_compressed(
      path,
      detected=np.asarray(int(bool(snapshot["detected"])), dtype=np.uint8),
      mask=np.asarray(snapshot["mask"]),
      box_xyxy=np.asarray(snapshot["box_xyxy"], dtype=np.float32),
      confidence=np.asarray(snapshot["confidence"], dtype=np.float32),
      class_id=np.asarray(snapshot["class_id"], dtype=np.int32),
      area=np.asarray(snapshot["area"], dtype=np.int64),
  )


def load_snapshot(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(f"Reference output does not exist: {path}")
  with np.load(path, allow_pickle=False) as data:
    return {
        "detected": bool(int(data["detected"])),
        "mask": np.asarray(data["mask"]).copy(),
        "box_xyxy": np.asarray(data["box_xyxy"], dtype=np.float32).reshape(4),
        "confidence": float(data["confidence"]),
        "class_id": int(data["class_id"]),
        "area": int(data["area"]),
    }


def draw_visualization_label(image: np.ndarray, text: str) -> None:
  import cv2

  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = max(0.5, min(image.shape[1] / 900.0, 0.8))
  thickness = 2
  (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
  padding = 8
  cv2.rectangle(
      image,
      (0, 0),
      (min(image.shape[1] - 1, text_width + padding * 2), text_height + baseline + padding * 2),
      (0, 0, 0),
      thickness=-1,
  )
  cv2.putText(
      image,
      text,
      (padding, text_height + padding),
      font,
      font_scale,
      (255, 255, 255),
      thickness,
      cv2.LINE_AA,
  )


def render_segmentation_overlay(
    rgb: np.ndarray,
    snapshot: dict[str, Any],
    title: str,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
  import cv2

  image = np.ascontiguousarray(rgb.copy())
  mask = np.asarray(snapshot["mask"]).astype(bool, copy=False)
  if mask.shape != image.shape[:2]:
    raise ValueError(f"Visualization mask shape {mask.shape} differs from RGB shape {image.shape[:2]}")

  if bool(snapshot["detected"]):
    color = np.asarray(color_rgb, dtype=np.float32)
    image[mask] = np.clip(
        image[mask].astype(np.float32) * (1.0 - VISUALIZATION_MASK_ALPHA)
        + color * VISUALIZATION_MASK_ALPHA,
        0,
        255,
    ).astype(np.uint8)
    mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color_rgb, 2, cv2.LINE_AA)

    box = np.asarray(snapshot["box_xyxy"], dtype=np.float32).reshape(4)
    if np.all(np.isfinite(box)):
      height, width = image.shape[:2]
      x1, y1, x2, y2 = np.rint(box).astype(int)
      x1 = int(np.clip(x1, 0, width - 1))
      y1 = int(np.clip(y1, 0, height - 1))
      x2 = int(np.clip(x2, 0, width - 1))
      y2 = int(np.clip(y2, 0, height - 1))
      cv2.rectangle(image, (x1, y1), (x2, y2), color_rgb, 2, cv2.LINE_AA)
    label = (
        f"{title} | class={int(snapshot['class_id'])} "
        f"conf={float(snapshot['confidence']):.3f} area={int(snapshot['area'])}"
    )
  else:
    label = f"{title} | no detection"

  draw_visualization_label(image, label)
  return image


def save_rgb_jpeg(path: Path, rgb: np.ndarray) -> None:
  import cv2

  bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
  written = cv2.imwrite(
      str(path),
      bgr,
      [cv2.IMWRITE_JPEG_QUALITY, VISUALIZATION_JPEG_QUALITY],
  )
  if not written:
    raise RuntimeError(f"Failed to save visualization image: {path}")


def export_visualizations(
    output_dir: Path,
    frames: list[FrameInput],
    current_snapshots: dict[str, dict[str, Any]],
    reference_snapshots: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
  visualization_dir = output_dir / "visualizations"
  visualization_dir.mkdir()
  files = []
  for frame in frames:
    current = current_snapshots[frame.key]
    if mode == "compare":
      reference = reference_snapshots[frame.key]
      original = frame.rgb.copy()
      draw_visualization_label(original, "Original RGB")
      baseline = render_segmentation_overlay(
          frame.rgb,
          reference,
          "Baseline",
          BASELINE_MASK_COLOR_RGB,
      )
      comparison = compare_snapshots(current, reference)
      current_overlay = render_segmentation_overlay(
          frame.rgb,
          current,
          f"Current | mask IoU={float(comparison['cmp_mask_iou']):.4f}",
          CURRENT_MASK_COLOR_RGB,
      )
      separator = np.full((frame.rgb.shape[0], 4, 3), 255, dtype=np.uint8)
      rendered = np.concatenate(
          (original, separator, baseline, separator, current_overlay),
          axis=1,
      )
      layout = "original|baseline|current"
    else:
      rendered = render_segmentation_overlay(
          frame.rgb,
          current,
          "YOLO segmentation",
          BASELINE_MASK_COLOR_RGB,
      )
      layout = "mask_overlay"

    filename = f"{frame.key}.jpg"
    save_rgb_jpeg(visualization_dir / filename, rendered)
    files.append({
        "frame_key": frame.key,
        "input_relative_path": frame.relative_path,
        "image": f"visualizations/{filename}",
    })

  return {
      "directory": "visualizations",
      "image_count": len(files),
      "layout": layout,
      "mask_alpha": VISUALIZATION_MASK_ALPHA,
      "jpeg_quality": VISUALIZATION_JPEG_QUALITY,
      "excluded_from_inference_timing": True,
      "files": files,
  }


def numeric_stats(values: list[Any]) -> dict[str, float | int] | None:
  numeric = np.asarray(
      [float(value) for value in values if value is not None and np.isfinite(float(value))],
      dtype=np.float64,
  )
  if numeric.size == 0:
    return None
  return {
      "count": int(numeric.size),
      "mean": float(np.mean(numeric)),
      "std": float(np.std(numeric)),
      "min": float(np.min(numeric)),
      "p10": float(np.percentile(numeric, 10)),
      "median": float(np.percentile(numeric, 50)),
      "p90": float(np.percentile(numeric, 90)),
      "p99": float(np.percentile(numeric, 99)),
      "max": float(np.max(numeric)),
  }


def distribution(values: list[Any]) -> dict[str, int]:
  result: dict[str, int] = {}
  for value in values:
    key = "null" if value is None else str(value)
    result[key] = result.get(key, 0) + 1
  return dict(sorted(result.items()))


def comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
  compared = [row for row in rows if "cmp_status" in row]
  if not compared:
    return None
  count = len(compared)
  pixel_total = sum(int(row["cmp_pixel_total"]) for row in compared)
  return {
      "count": count,
      "status_counts": distribution([row["cmp_status"] for row in compared]),
      "detection_equal_rate": sum(bool(row["cmp_detection_equal"]) for row in compared) / count,
      "exact_mask_frame_rate": sum(bool(row["cmp_exact_mask"]) for row in compared) / count,
      "mask_shape_equal_rate": sum(bool(row["cmp_mask_shape_equal"]) for row in compared) / count,
      "mask_dtype_equal_rate": sum(bool(row["cmp_mask_dtype_equal"]) for row in compared) / count,
      "class_id_equal_rate": sum(bool(row["cmp_class_id_equal"]) for row in compared) / count,
      "global_pixel_equal_ratio": (
          sum(int(row["cmp_pixel_equal_count"]) for row in compared) / max(pixel_total, 1)
      ),
      "pixel_equal_ratio": numeric_stats([row["cmp_pixel_equal_ratio"] for row in compared]),
      "mask_iou": numeric_stats([row["cmp_mask_iou"] for row in compared]),
      "area_abs_diff": numeric_stats([row["cmp_area_abs_diff"] for row in compared]),
      "area_rel_diff": numeric_stats([row["cmp_area_rel_diff"] for row in compared]),
      "bbox_max_abs_diff": numeric_stats([row["cmp_bbox_max_abs_diff"] for row in compared]),
      "confidence_abs_diff": numeric_stats([row["cmp_confidence_abs_diff"] for row in compared]),
  }


def build_summary(
    rows: list[dict[str, Any]],
    cold_row: dict[str, Any],
    detector_construction_ms: float,
    warmup: int,
) -> dict[str, Any]:
  return {
      "detector_construction_ms": detector_construction_ms,
      "first_inference_after_construction": cold_row,
      "cold_first_inference": cold_row,
      "warmup_iterations_excluded": warmup,
      "measured_iterations": len(rows),
      "timing_ms": {
          field: numeric_stats([row.get(field) for row in rows])
          for field in ("wall_ms", *TIMING_FIELDS.keys())
      },
      "candidate_count": numeric_stats([row["candidate_count"] for row in rows]),
      "execution_paths": distribution([row["execution_path"] for row in rows]),
      "detected_count": sum(bool(row["detected"]) for row in rows),
      "missing_count": sum(not bool(row["detected"]) for row in rows),
      "output_mask_shapes": distribution([row["output_mask_shape"] for row in rows]),
      "output_mask_dtypes": distribution([row["output_mask_dtype"] for row in rows]),
      "comparison": comparison_summary(rows),
  }


def write_json(path: Path, value: Any) -> None:
  with path.open("w", encoding="utf-8") as file:
    json.dump(json_safe(value), file, indent=2, ensure_ascii=False, allow_nan=False)
    file.write("\n")


def write_samples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames: list[str] = []
  seen = set()
  for row in rows:
    for key in row:
      if key not in seen:
        seen.add(key)
        fieldnames.append(key)
  with path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def optional_module_version(name: str) -> str | None:
  try:
    module = importlib.import_module(name)
  except Exception:
    return None
  return str(getattr(module, "__version__", "unknown"))


def capture_command(command: list[str]) -> dict[str, Any] | None:
  executable = shutil.which(command[0])
  if executable is None:
    return None
  try:
    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
  except Exception as exc:
    return {"error": str(exc)}
  return {
      "returncode": completed.returncode,
      "stdout": completed.stdout.strip(),
      "stderr": completed.stderr.strip(),
  }


def read_thermal_zones() -> dict[str, float]:
  temperatures: dict[str, float] = {}
  for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
    try:
      name = (zone / "type").read_text(encoding="utf-8").strip()
      raw = float((zone / "temp").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
      continue
    temperatures[f"{zone.name}:{name}"] = raw / 1000.0 if abs(raw) >= 1000 else raw
  return temperatures


def backend_metadata(detector: YoloSegmenter) -> dict[str, Any]:
  backend = getattr(detector, "_fast_backend", None)
  if backend is None:
    predictor = getattr(detector.model, "predictor", None)
    backend = getattr(predictor, "model", None)
  if backend is None:
    raise RuntimeError("YOLO TensorRT backend was not initialized by the cold inference")
  model_format = str(getattr(backend, "format", ""))
  if model_format != "engine":
    raise RuntimeError(f"Expected TensorRT engine backend, got format={model_format!r}")

  bindings = {}
  for name, binding in dict(getattr(backend, "bindings", {})).items():
    dtype = getattr(binding, "dtype", None)
    shape = getattr(binding, "shape", None)
    bindings[str(name)] = {
        "dtype": None if dtype is None else str(dtype),
        "shape": None if shape is None else [int(value) for value in shape],
    }
  device = getattr(backend, "device", None)
  return {
      "format": model_format,
      "device": None if device is None else str(device),
      "fp16": bool(getattr(backend, "fp16", False)),
      "dynamic": bool(getattr(backend, "dynamic", False)),
      "end2end": bool(getattr(backend, "end2end", False)),
      "bindings": bindings,
  }


def environment_metadata() -> dict[str, Any]:
  metadata: dict[str, Any] = {
      "python": platform.python_version(),
      "platform": platform.platform(),
      "machine": platform.machine(),
      "versions": {
          name: optional_module_version(name)
          for name in ("numpy", "torch", "ultralytics", "tensorrt", "cv2")
      },
      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
      "nvpmodel": capture_command(["nvpmodel", "-q"]),
      "jetson_clocks": capture_command(["jetson_clocks", "--show"]),
  }
  try:
    import torch

    metadata["torch_cuda_version"] = torch.version.cuda
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
      metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
      metadata["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
  except Exception as exc:
    metadata["torch_cuda_error"] = str(exc)
  return metadata


def validate_reference_compatibility(
    manifest: dict[str, Any],
    engine_sha256: str,
    yolo_config_sha256: str,
    frames: list[FrameInput],
    allow_engine_difference: bool,
) -> None:
  compatibility = manifest.get("compatibility") or {}
  expected_engine = compatibility.get("engine_sha256")
  expected_yolo_config = compatibility.get("yolo_config_sha256")
  if expected_engine != engine_sha256 and not allow_engine_difference:
    raise RuntimeError(f"TensorRT engine SHA256 differs: reference={expected_engine}, current={engine_sha256}")
  if expected_yolo_config != yolo_config_sha256:
    raise RuntimeError(
        f"YOLO config SHA256 differs: reference={expected_yolo_config}, current={yolo_config_sha256}"
    )
  entries = manifest.get("inputs") or []
  if len(entries) != len(frames):
    raise RuntimeError(f"Input count differs: reference={len(entries)}, current={len(frames)}")
  for entry, frame in zip(entries, frames):
    if entry.get("key") != frame.key or entry.get("relative_path") != frame.relative_path:
      raise RuntimeError(f"Input order differs at {frame.relative_path}")
    if entry.get("sha256") != frame.sha256:
      raise RuntimeError(f"Input SHA256 differs: {frame.relative_path}")


def print_timing_summary(summary: dict[str, Any]) -> None:
  print("[YOLO-BENCH] Measured timing (ms)")
  print(f"{'stage':<34}{'median':>12}{'p90':>12}{'p99':>12}")
  print("-" * 70)
  for field, stats in summary["timing_ms"].items():
    if stats is None:
      continue
    print(f"{field:<34}{stats['median']:>12.3f}{stats['p90']:>12.3f}{stats['p99']:>12.3f}")


def main() -> None:
  args = parse_args()
  config_path = resolve_cli_path(args.config)
  input_dir = resolve_cli_path(args.input_dir)
  output_root = resolve_cli_path(args.output_root)
  _config, yolo_config = load_config(config_path)
  weights_path = resolve_repo_path(requested_yolo_weights(yolo_config))
  if not weights_path.is_file():
    raise FileNotFoundError(f"YOLO weights do not exist: {weights_path}")
  if weights_path.suffix.lower() != ".engine":
    raise RuntimeError(f"This benchmark requires a TensorRT .engine file, got: {weights_path}")

  postprocess_backend = str(
      args.postprocess_backend or yolo_config.get("postprocess_backend", "gpu")
  ).lower()

  reference_root = None
  reference_manifest = None
  if args.mode == "compare":
    reference_root, reference_manifest = load_reference_manifest(args.reference)
    iterations = int(args.iterations or reference_manifest["benchmark"]["iterations"])
    warmup = int(args.warmup if args.warmup is not None else reference_manifest["benchmark"]["warmup"])
    input_paths = paths_from_manifest(input_dir, reference_manifest)
  else:
    iterations = int(args.iterations or 100)
    warmup = int(args.warmup if args.warmup is not None else 10)
    input_paths = discover_input_paths(input_dir, args.max_frames)
    if len(input_paths) > iterations:
      print(
          f"[YOLO-BENCH] Input count {len(input_paths)} exceeds iterations {iterations}; "
          f"using the first {iterations} sorted frames"
      )
      input_paths = input_paths[:iterations]

  print(f"[YOLO-BENCH] Loading {len(input_paths)} recorded RGB frame(s) into memory")
  frames = load_frames(input_paths, input_dir)
  comparable_yolo_config = comparison_yolo_config(yolo_config)
  yolo_config_sha256 = sha256_json(comparable_yolo_config)

  if reference_manifest is not None:
    expected_iterations = int(reference_manifest["benchmark"]["iterations"])
    if iterations != expected_iterations:
      raise RuntimeError(f"Measured iterations differ: reference={expected_iterations}, current={iterations}")

  run_name = args.run_name or f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  output_dir = output_root / run_name
  output_dir.mkdir(parents=True, exist_ok=False)
  reference_output_dir = output_dir / "references"
  if args.mode == "baseline":
    reference_output_dir.mkdir()

  temperature_start = read_thermal_zones()
  execution_path = str(args.execution_path or yolo_config.get("execution_path", "legacy")).lower()
  if execution_path not in ("legacy", "fast"):
    raise ValueError(f"Unsupported YOLO execution path: {execution_path!r}")
  fallback_to_legacy = bool(args.allow_fast_fallback) if execution_path == "fast" else True
  print(f"[YOLO-BENCH] Constructing detector: {display_path(weights_path)}")
  print(
      f"[YOLO-BENCH] Execution path={execution_path} "
      f"postprocess={postprocess_backend} profile_stages={args.profile_stages} "
      f"legacy_fallback={fallback_to_legacy}"
  )
  model_started = time.perf_counter()
  detector = build_detector(
      yolo_config,
      weights_path,
      execution_path=execution_path,
      profile_stages=bool(args.profile_stages),
      fallback_to_legacy=fallback_to_legacy,
      postprocess_backend=postprocess_backend,
  )
  detector_construction_ms = (time.perf_counter() - model_started) * 1000.0

  print("[YOLO-BENCH] Running one separately reported first inference after construction")
  cold_observation = run_prediction(detector, frames[0])
  cold_snapshot = result_snapshot(cold_observation.result, frames[0].rgb.shape[:2])
  cold_row = build_sample_row(
      phase="cold",
      iteration=0,
      frame=frames[0],
      observation=cold_observation,
      snapshot=cold_snapshot,
      reference=None,
  )

  print(f"[YOLO-BENCH] Running {warmup} warmup iteration(s), excluded from statistics")
  for index in range(warmup):
    run_prediction(detector, frames[index % len(frames)])

  actual_backend = backend_metadata(detector)
  actual_weights_path = Path(detector.selected_weights).expanduser().resolve()
  engine_sha256 = sha256_file(actual_weights_path)
  print(f"[YOLO-BENCH] Measured engine={display_path(actual_weights_path)}")
  if reference_manifest is not None:
    validate_reference_compatibility(
        reference_manifest,
        engine_sha256,
        yolo_config_sha256,
        frames,
        allow_engine_difference=bool(args.allow_engine_difference),
    )

  reference_snapshots: dict[str, dict[str, Any]] = {}
  reference_entries = {}
  if reference_manifest is not None:
    reference_entries = {str(entry["key"]): entry for entry in reference_manifest["inputs"]}
    for frame in frames:
      entry = reference_entries[frame.key]
      reference_snapshots[frame.key] = load_snapshot(reference_root / str(entry["reference_output"]))

  print(f"[YOLO-BENCH] Running {iterations} measured iteration(s)")
  rows: list[dict[str, Any]] = []
  current_snapshots: dict[str, dict[str, Any]] = {}
  for index in range(iterations):
    frame = frames[index % len(frames)]
    observation = run_prediction(detector, frame)
    snapshot = result_snapshot(observation.result, frame.rgb.shape[:2])
    current_snapshots.setdefault(frame.key, snapshot)
    if args.mode == "baseline":
      reference = reference_snapshots.setdefault(frame.key, snapshot)
    else:
      reference = reference_snapshots[frame.key]
    rows.append(
        build_sample_row(
            phase="measured",
            iteration=index + 1,
            frame=frame,
            observation=observation,
            snapshot=snapshot,
            reference=reference,
        )
    )
    if args.progress_every > 0 and ((index + 1) % args.progress_every == 0 or index + 1 == iterations):
      print(f"[YOLO-BENCH] Progress {index + 1}/{iterations}")

  final_weights_path = Path(detector.selected_weights).expanduser().resolve()
  if final_weights_path != actual_weights_path:
    raise RuntimeError(
        "YOLO engine changed during measured iterations; refusing to save a mixed-engine run: "
        f"measured_start={actual_weights_path}, final={final_weights_path}"
    )

  manifest_inputs = []
  if args.mode == "baseline":
    for frame in frames:
      reference_name = f"{frame.key}.npz"
      save_snapshot(reference_output_dir / reference_name, reference_snapshots[frame.key])
      manifest_inputs.append({
          "key": frame.key,
          "relative_path": frame.relative_path,
          "sha256": frame.sha256,
          "reference_output": f"references/{reference_name}",
      })
  else:
    manifest_inputs = list(reference_manifest["inputs"])

  print("[YOLO-BENCH] Exporting segmentation visualizations outside measured timing")
  visualization = export_visualizations(
      output_dir,
      frames,
      current_snapshots,
      reference_snapshots,
      args.mode,
  )
  temperature_end = read_thermal_zones()
  summary = build_summary(rows, cold_row, detector_construction_ms, warmup)
  summary["visualization"] = visualization
  metadata = {
      "format_version": 1,
      "created_at_utc": utc_now(),
      "mode": args.mode,
      "command": sys.argv,
      "paths": {
          "config": display_path(config_path),
          "input_dir": display_path(input_dir),
          "requested_weights": display_path(weights_path),
          "actual_weights": display_path(actual_weights_path),
          "output_dir": display_path(output_dir),
          "reference": None if args.reference is None else str(resolve_cli_path(args.reference)),
      },
      "benchmark": {
          "iterations": iterations,
          "warmup": warmup,
          "input_count": len(frames),
          "input_schedule": "sorted_round_robin",
          "cold_inference_excluded": True,
          "input_preloaded": True,
          "candidate_count_source": "yolo_segmenter.last_candidate_count",
          "requested_execution_path": execution_path,
          "postprocess_backend": postprocess_backend,
          "profile_stages": bool(args.profile_stages),
          "fallback_to_legacy": fallback_to_legacy,
          "allow_engine_difference": bool(args.allow_engine_difference),
      },
      "compatibility": {
          "engine_sha256": engine_sha256,
          "engine_size_bytes": actual_weights_path.stat().st_size,
          "config_sha256": sha256_file(config_path),
          "yolo_config_sha256": yolo_config_sha256,
          "yolo_config": yolo_config,
          "comparison_yolo_config": comparable_yolo_config,
      },
      "implementation": {
          "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
          "yolo_segmenter_sha256": sha256_file(SCRIPT_DIR / "yolo_segmenter.py"),
      },
      "actual_backend": actual_backend,
      "executor": detector.execution_metadata(),
      "environment": environment_metadata(),
      "temperature_c": {
          "start": temperature_start,
          "end": temperature_end,
      },
      "visualization": visualization,
  }
  manifest = {
      "format_version": 1,
      "created_at_utc": metadata["created_at_utc"],
      "benchmark": metadata["benchmark"],
      "compatibility": metadata["compatibility"],
      "inputs": manifest_inputs,
  }

  write_samples_csv(output_dir / "samples.csv", rows)
  write_json(output_dir / "summary.json", summary)
  write_json(output_dir / "metadata.json", metadata)
  write_json(output_dir / "manifest.json", manifest)
  print_timing_summary(summary)
  comparison = summary.get("comparison")
  if comparison is not None:
    print(
        f"[YOLO-BENCH] Output comparison: detection_equal={comparison['detection_equal_rate']:.6f} "
        f"exact_mask_frames={comparison['exact_mask_frame_rate']:.6f} "
        f"median_iou={comparison['mask_iou']['median']:.6f}"
    )
  print(f"[YOLO-BENCH] Visualizations saved to: {output_dir / 'visualizations'}")
  print(f"[YOLO-BENCH] Results saved to: {output_dir}")


if __name__ == "__main__":
  main()