#!/usr/bin/env python3 候选可预测性统计。
"""Replay recorded register frames and evaluate fixed-candidate predictability offline."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


REALTIME_ROOT = Path(__file__).resolve().parents[1]
if str(REALTIME_ROOT) not in sys.path:
  sys.path.insert(0, str(REALTIME_ROOT))

from run_realtime import build_tracker, evaluate_register_quality, load_config
from simulate_recorded_camera import RecordedFramePublisher


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--config", type=Path, default=REALTIME_ROOT / "config.yaml")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--pattern", default="*_register.npz")
  parser.add_argument("--max-samples", type=int)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument("--canonical-source-index", type=int)
  parser.add_argument("--refiner-fp16-engine", type=Path)
  parser.add_argument("--baseline-records", type=Path)
  return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
  if not values:
    return None
  return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def mask_iou(rendered_mask: np.ndarray, observed_mask: np.ndarray) -> float:
  rendered = np.asarray(rendered_mask) > 0
  observed = np.asarray(observed_mask) > 0
  intersection = int(np.logical_and(rendered, observed).sum())
  union = int(np.logical_or(rendered, observed).sum())
  return float(intersection / union) if union else 0.0


def summarize(
    records: list[dict],
    render_iou_threshold: float,
    *,
    accepted_only: bool = True,
) -> dict:
  aggregates: dict[int, dict[str, object]] = {}
  for record in records:
    if accepted_only and not record.get("phase2_accepted", False):
      continue
    for candidate in record["candidate_predictability"]["candidates"]:
      source_id = int(candidate["source_candidate_id"])
      aggregate = aggregates.setdefault(source_id, {
          "candidate_id": int(candidate["candidate_id"]),
          "error_xy_m": [],
          "error_z_m": [],
          "error_translation_m": [],
          "render_iou": [],
          "selected_count": 0,
      })
      aggregate["selected_count"] = int(aggregate["selected_count"]) + int(candidate["selected_by_phase2"])
      for name in ("error_xy_m", "error_z_m", "error_translation_m", "render_iou"):
        aggregate[name].append(float(candidate[name]))

  candidates = []
  for source_id in sorted(aggregates):
    values = aggregates[source_id]
    translation_errors = values["error_translation_m"]
    render_ious = values["render_iou"]
    sample_count = len(translation_errors)
    candidates.append({
        "candidate_id": int(values["candidate_id"]),
        "source_candidate_id": source_id,
        "sample_count": sample_count,
        "selected_count": int(values["selected_count"]),
        "selected_fraction": int(values["selected_count"]) / sample_count,
        "error_xy_p50_m": percentile(values["error_xy_m"], 0.50),
        "error_xy_p95_m": percentile(values["error_xy_m"], 0.95),
        "error_z_p50_m": percentile(values["error_z_m"], 0.50),
        "error_z_p95_m": percentile(values["error_z_m"], 0.95),
        "error_translation_p50_m": percentile(translation_errors, 0.50),
        "error_translation_p95_m": percentile(translation_errors, 0.95),
        "error_translation_p99_m": percentile(translation_errors, 0.99),
        "error_translation_max_m": max(translation_errors),
        "over_5mm_fraction": sum(value > 0.005 for value in translation_errors) / sample_count,
        "over_10mm_fraction": sum(value > 0.010 for value in translation_errors) / sample_count,
        "over_20mm_fraction": sum(value > 0.020 for value in translation_errors) / sample_count,
        "render_iou_p50": percentile(render_ious, 0.50),
        "render_iou_p05": percentile(render_ious, 0.05),
        "render_iou_mean": statistics.fmean(render_ious),
        "render_iou_pass_fraction": sum(value >= render_iou_threshold for value in render_ious) / sample_count,
    })

  accepted = sum(record.get("phase2_accepted", False) for record in records)
  return {
      "schema_version": 1,
      "record_count": len(records),
      "phase2_accepted_count": accepted,
      "phase2_rejected_count": len(records) - accepted,
      "render_iou_threshold": render_iou_threshold,
      "candidates": candidates,
  }


def summarize_baseline_comparison(records: list[dict]) -> dict | None:
  compared = [record for record in records if record.get("baseline_translation_error_m") is not None]
  if not compared:
    return None
  translation_errors = [float(record["baseline_translation_error_m"]) for record in compared]
  xy_errors = [float(record["baseline_xy_error_m"]) for record in compared]
  z_errors = [float(record["baseline_z_error_m"]) for record in compared]
  refiner_times = [float(record["refiner_coarse_seconds"]) for record in compared]
  register_times = [float(record["register_seconds"]) for record in compared]
  n1_accepted = sum(record.get("phase2_accepted", False) for record in compared)
  baseline_accepted = sum(record.get("baseline_phase2_accepted", False) for record in compared)
  recoverable = sum(
      not record.get("phase2_accepted", False) and record.get("baseline_phase2_accepted", False)
      for record in compared
  )
  result = {
      "sample_count": len(compared),
      "translation_error_p50_m": percentile(translation_errors, 0.50),
      "translation_error_p95_m": percentile(translation_errors, 0.95),
      "translation_error_p99_m": percentile(translation_errors, 0.99),
      "translation_error_max_m": max(translation_errors),
      "xy_error_p50_m": percentile(xy_errors, 0.50),
      "xy_error_p95_m": percentile(xy_errors, 0.95),
      "z_error_p50_m": percentile(z_errors, 0.50),
      "z_error_p95_m": percentile(z_errors, 0.95),
      "refiner_coarse_seconds_p50": percentile(refiner_times, 0.50),
      "refiner_coarse_seconds_p95": percentile(refiner_times, 0.95),
      "register_seconds_p50": percentile(register_times, 0.50),
      "register_seconds_p95": percentile(register_times, 0.95),
      "refiner_coarse_backends": dict(Counter(record.get("refiner_coarse_backend") for record in compared)),
      "single_candidate_statuses": dict(Counter(record.get("single_candidate_status") for record in compared)),
      "coarse_scorer_statuses": dict(Counter(record.get("coarse_scorer_status") for record in compared)),
      "n1_accepted_count": n1_accepted,
      "n1_rejected_count": len(compared) - n1_accepted,
      "n1_direct_accept_fraction": n1_accepted / len(compared),
      "baseline_phase2_accepted_count": baseline_accepted,
      "projected_n5_recovery_count": recoverable,
      "projected_n5_recovery_fraction": recoverable / len(compared),
  }
  timed = [record for record in compared if record.get("baseline_register_seconds") is not None]
  if timed:
    baseline_refiner_times = [float(record["baseline_refiner_coarse_seconds"]) for record in timed]
    baseline_register_times = [float(record["baseline_register_seconds"]) for record in timed]
    result.update({
        "baseline_refiner_coarse_seconds_p50": percentile(baseline_refiner_times, 0.50),
        "baseline_refiner_coarse_seconds_p95": percentile(baseline_refiner_times, 0.95),
        "baseline_register_seconds_p50": percentile(baseline_register_times, 0.50),
        "baseline_register_seconds_p95": percentile(baseline_register_times, 0.95),
        "register_p50_speedup": (
            percentile(baseline_register_times, 0.50) / percentile(register_times, 0.50)
        ),
        "refiner_coarse_p50_speedup": (
            percentile(baseline_refiner_times, 0.50) / percentile(refiner_times, 0.50)
        ),
        "baseline_refiner_coarse_backends": dict(
            Counter(record.get("baseline_refiner_coarse_backend") for record in timed)
        ),
    })
  return result


def print_summary(summary: dict) -> None:
  print(
      f"[Phase3-0][SUMMARY] records={summary['record_count']} "
      f"phase2_accepted={summary['phase2_accepted_count']} "
      f"phase2_rejected={summary['phase2_rejected_count']}"
  )
  print("candidate source selected et_p50 et_p95 et_p99 max_mm iou_p50 iou_pass")
  for item in summary["candidates"]:
    print(
        f"{item['candidate_id']:>9} {item['source_candidate_id']:>6} "
        f"{item['selected_count']:>4}/{item['sample_count']:<4} "
        f"{item['error_translation_p50_m'] * 1000.0:>7.3f} "
        f"{item['error_translation_p95_m'] * 1000.0:>7.3f} "
        f"{item['error_translation_p99_m'] * 1000.0:>7.3f} "
        f"{item['error_translation_max_m'] * 1000.0:>6.3f} "
        f"{item['render_iou_p50']:>7.4f} "
        f"{item['render_iou_pass_fraction']:>8.3f}"
    )


def main() -> None:
  args = parse_args()
  if args.stride < 1:
    raise ValueError("--stride must be positive")
  if args.max_samples is not None and args.max_samples < 1:
    raise ValueError("--max-samples must be positive")

  input_dir = args.input_dir.expanduser().resolve()
  config_path = args.config.expanduser().resolve()
  output_dir = args.output_dir.expanduser().resolve()
  paths = sorted(input_dir.glob(args.pattern))[::args.stride]
  if args.max_samples is not None:
    paths = paths[:args.max_samples]
  if not paths:
    raise RuntimeError(f"No inputs match {args.pattern!r} in {input_dir}")

  cfg = load_config(str(config_path))
  foundation_cfg = cfg.get("foundationpose", {})
  run_mode = "phase2_n5"
  if args.canonical_source_index is not None:
    if args.refiner_fp16_engine is None:
      raise ValueError("--canonical-source-index requires --refiner-fp16-engine")
    engine_path = args.refiner_fp16_engine.expanduser().resolve()
    foundation_cfg["initial_candidate_selection"] = {
        "enabled": True,
        "mode": "explicit_indices",
        "indices": [int(args.canonical_source_index)],
    }
    foundation_cfg["translation_consensus"] = {"enabled": False}
    foundation_cfg["single_candidate_mode"] = {"enabled": True}
    foundation_cfg["fine_stage_enabled"] = False
    foundation_cfg["candidate_predictability_shadow"] = {
        **dict(foundation_cfg.get("candidate_predictability_shadow", {}) or {}),
        "enabled": True,
        "expected_candidate_count": 1,
    }
    refiner_backend = dict(dict(foundation_cfg.get("tensorrt", {}) or {}).get("refiner", {}) or {})
    refiner_backend.update({
        "enabled": True,
        "precision": "fp16",
        "fallback_to_fp16": False,
        "engine_paths": {"refiner_coarse": str(engine_path)},
        "min_candidates": 1,
        "max_candidates": 1,
        "stages": ["refiner_coarse"],
    })
    foundation_cfg["tensorrt"] = {
        **dict(foundation_cfg.get("tensorrt", {}) or {}),
        "refiner": refiner_backend,
    }
    run_mode = f"phase3a_n1_source_{args.canonical_source_index}_fp16"
  shadow_cfg = dict(foundation_cfg.get("candidate_predictability_shadow", {}) or {})
  if not shadow_cfg.get("enabled", False):
    raise RuntimeError("foundationpose.candidate_predictability_shadow.enabled must be true")
  runtime_cfg = cfg.get("runtime", {})
  render_iou_threshold = float(runtime_cfg.get("register_min_render_iou", 0.0))
  output_dir.mkdir(parents=True, exist_ok=True)
  records_path = output_dir / "records.jsonl"
  summary_path = output_dir / "summary.json"

  baseline_by_name = {}
  if args.baseline_records is not None:
    with args.baseline_records.expanduser().resolve().open("r", encoding="utf-8") as baseline_file:
      for line in baseline_file:
        if not line.strip():
          continue
        baseline_record = json.loads(line)
        baseline_by_name[Path(baseline_record["input"]).name] = baseline_record

  tracker = build_tracker(foundation_cfg)
  records = []
  started = time.perf_counter()
  with records_path.open("w", encoding="utf-8", buffering=1) as output_file:
    for index, path in enumerate(paths, start=1):
      try:
        frame = RecordedFramePublisher(str(path))
        tracker.reset()
        pose_result = tracker.register(
            frame.rgb,
            frame.depth,
            frame.K,
            frame.mask,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            sequence_id=path.name,
            capture_source="candidate_predictability_batch_replay",
        )
        timing = dict(getattr(tracker.estimator, "last_register_timing", {}))
        predictability = timing.get("candidate_predictability")
        if timing.get("candidate_predictability_status") != "completed" or not isinstance(predictability, dict):
          raise RuntimeError(f"candidate predictability status={timing.get('candidate_predictability_status')!r}")

        quality = evaluate_register_quality(
            tracker=tracker,
            pose=pose_result.pose,
            depth=frame.depth,
            K=frame.K,
            mask=frame.mask,
            runtime_cfg=runtime_cfg,
        )
        for candidate in predictability["candidates"]:
          centered_pose = np.asarray(candidate["centered_pose"], dtype=np.float32).reshape(4, 4)
          rendered_mask = tracker.render_pose_mask(frame.K, frame.depth.shape[:2], pose=centered_pose)
          candidate["render_iou"] = mask_iou(rendered_mask, frame.mask)

        record = {
            "schema_version": 1,
            "input": str(path),
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "phase2_accepted": bool(quality.accepted),
            "phase2_reject_reason": quality.reject_reason,
            "phase2_render_iou": quality.render_iou,
            "translation_consensus_status": timing.get("translation_consensus_status"),
            "translation_consensus_passed": timing.get("translation_consensus_passed"),
            "single_candidate_status": timing.get("single_candidate_status"),
            "coarse_scorer_status": timing.get("coarse_scorer_status"),
            "refiner_coarse_backend": timing.get("refiner_coarse_detail", {}).get("network_backend"),
            "refiner_coarse_seconds": timing.get("refiner_coarse"),
            "register_seconds": timing.get("register"),
            "candidate_predictability": predictability,
        }
        baseline = baseline_by_name.get(path.name)
        if baseline is not None and "candidate_predictability" in baseline:
          reference_translation = np.asarray(
              predictability["reference_translation_m"], dtype=np.float64
          )
          baseline_translation = np.asarray(
              baseline["candidate_predictability"]["reference_translation_m"], dtype=np.float64
          )
          record["baseline_translation_error_m"] = float(
              np.linalg.norm(reference_translation - baseline_translation)
          )
          translation_delta = reference_translation - baseline_translation
          record["baseline_xy_error_m"] = float(np.linalg.norm(translation_delta[:2]))
          record["baseline_z_error_m"] = float(abs(translation_delta[2]))
          record["baseline_phase2_accepted"] = bool(baseline.get("phase2_accepted", False))
          record["baseline_phase2_render_iou"] = baseline.get("phase2_render_iou")
          record["baseline_refiner_coarse_backend"] = baseline.get("refiner_coarse_backend")
          record["baseline_refiner_coarse_seconds"] = baseline.get("refiner_coarse_seconds")
          record["baseline_register_seconds"] = baseline.get("register_seconds")
      except Exception as error:
        record = {
            "schema_version": 1,
            "input": str(path),
            "error": f"{type(error).__name__}: {error}",
            "phase2_accepted": False,
        }
      records.append(record)
      output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
      if index % max(1, args.progress_interval) == 0 or index == len(paths):
        failures = sum("error" in item for item in records)
        print(
            f"[Phase3-0] processed={index}/{len(paths)} failures={failures} "
            f"elapsed={time.perf_counter() - started:.1f}s"
        )

  summary = summarize(
      [record for record in records if "error" not in record],
      render_iou_threshold,
      accepted_only=args.canonical_source_index is None,
  )
  summary["run_mode"] = run_mode
  summary["baseline_comparison"] = summarize_baseline_comparison(
      [record for record in records if "error" not in record]
  )
  summary["input_dir"] = str(input_dir)
  summary["config"] = str(config_path)
  summary["records_path"] = str(records_path)
  summary["processing_error_count"] = sum("error" in record for record in records)
  summary["processing_errors"] = [record for record in records if "error" in record]
  summary["elapsed_seconds"] = time.perf_counter() - started
  summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  print_summary(summary)
  print(f"[Phase3-0] records={records_path}")
  print(f"[Phase3-0] summary={summary_path}")


if __name__ == "__main__":
  main()
