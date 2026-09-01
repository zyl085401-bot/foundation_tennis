#!/usr/bin/env python3 Jetson 性能数据分析。
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
import re
import statistics


EVENT_PREFIX = "[EVENT][foundationpose_init] "
TIMING_HEADER_RE = re.compile(r"^\[TIMER\]\[SUMMARY\]\[foundationpose_init\] init=(\d+) unit=ms$")
TIMING_VALUE_RE = re.compile(
    r"^\s*(yolo_total|yolo_candidate_count|yolo_inference|yolo_postprocess_internal|"
    r"yolo_queue_wait|detection_consume_delay|foundation_total|"
    r"yolo_foundation_compute_total|init_total)\s+([0-9.]+)$"
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Correlate FoundationPose initialization events with Jetson telemetry.")
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("--max-nearest-ms", type=float, default=150.0)
  return parser.parse_args()


def parse_key_values(text: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for token in text.split():
    if "=" in token:
      key, value = token.split("=", 1)
      values[key] = value
  return values


def parse_runtime(path: Path) -> list[dict[str, object]]:
  events: dict[int, dict[str, object]] = {}
  current_init: int | None = None
  for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if raw_line.startswith(EVENT_PREFIX):
      values = parse_key_values(raw_line[len(EVENT_PREFIX):])
      init_index = int(values["init"])
      event: dict[str, object] = {
          "schema": int(values["schema"]),
          "init": init_index,
          "frame": int(values["frame"]),
          "camera_timestamp": float(values["camera_timestamp"]),
      }
      for key in (
          "yolo_start_unix_ns",
          "yolo_end_unix_ns",
          "detection_consumed_unix_ns",
          "register_start_unix_ns",
          "register_end_unix_ns",
          "init_success_unix_ns",
      ):
        event[key] = int(values[key])
      events[init_index] = event
      continue

    header_match = TIMING_HEADER_RE.match(raw_line)
    if header_match:
      current_init = int(header_match.group(1))
      continue
    if current_init is None or current_init not in events:
      continue
    value_match = TIMING_VALUE_RE.match(raw_line)
    if value_match:
      key, value = value_match.groups()
      parsed_value = int(value) if key == "yolo_candidate_count" else float(value)
      if key == "yolo_queue_wait":
        events[current_init]["detection_consume_delay"] = parsed_value
      else:
        events[current_init][key] = parsed_value

  ordered_events = [events[index] for index in sorted(events)]
  for event in ordered_events:
    if "yolo_total" in event and "foundation_total" in event:
      event["yolo_foundation_compute_total"] = (
          float(event["yolo_total"]) + float(event["foundation_total"])
      )
  return ordered_events


def read_telemetry(path: Path) -> tuple[list[int], list[dict[str, str]]]:
  if not path.is_file():
    return [], []
  with path.open("r", newline="", encoding="utf-8") as input_file:
    rows = list(csv.DictReader(input_file))
  rows = [row for row in rows if row.get("timestamp_unix_ns")]
  rows.sort(key=lambda row: int(row["timestamp_unix_ns"]))
  return [int(row["timestamp_unix_ns"]) for row in rows], rows


def nearest_row(
    timestamps: list[int],
    rows: list[dict[str, str]],
    target_ns: int,
) -> tuple[dict[str, str] | None, float | None]:
  if not timestamps:
    return None, None
  position = bisect.bisect_left(timestamps, target_ns)
  candidates = [index for index in (position - 1, position) if 0 <= index < len(timestamps)]
  nearest_index = min(candidates, key=lambda index: abs(timestamps[index] - target_ns))
  delta_ms = abs(timestamps[nearest_index] - target_ns) / 1_000_000.0
  return rows[nearest_index], delta_ms


def prefixed_values(prefix: str, row: dict[str, str] | None) -> dict[str, object]:
  if row is None:
    return {}
  return {
      f"{prefix}_{key}": value
      for key, value in row.items()
      if key not in {"timestamp_unix_ns", "timestamp_iso"}
  }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  if not rows:
    path.write_text("", encoding="utf-8")
    return
  fieldnames: list[str] = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def percentile(values: list[float], percentage: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  position = (len(ordered) - 1) * percentage / 100.0
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
  return {
      "count": len(values),
      "mean": statistics.fmean(values) if values else None,
      "min": min(values) if values else None,
      "p50": percentile(values, 50.0),
      "p95": percentile(values, 95.0),
      "p99": percentile(values, 99.0),
      "max": max(values) if values else None,
  }


def pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
  if len(pairs) < 2:
    return None
  x_values = [pair[0] for pair in pairs]
  y_values = [pair[1] for pair in pairs]
  x_mean = statistics.fmean(x_values)
  y_mean = statistics.fmean(y_values)
  numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
  x_variance = sum((x - x_mean) ** 2 for x in x_values)
  y_variance = sum((y - y_mean) ** 2 for y in y_values)
  denominator = (x_variance * y_variance) ** 0.5
  return None if denominator == 0.0 else numerator / denominator


def telemetry_correlations(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
  fields = (
      ("fast_gpu_load_pct", "fast_telemetry_covered"),
      ("fast_gpu_freq_hz", "fast_telemetry_covered"),
      ("full_gpu_load_pct", "full_telemetry_covered"),
      ("full_gpu_freq_khz", "full_telemetry_covered"),
      ("full_emc_freq_khz", "full_telemetry_covered"),
      ("full_emc_load_pct", "full_telemetry_covered"),
      ("full_vdd_in_mw", "full_telemetry_covered"),
      ("full_temp_gpu_c", "full_telemetry_covered"),
  )
  result: dict[str, dict[str, float | int | None]] = {}
  for field, coverage_field in fields:
    pairs = []
    for row in rows:
      value = row.get(field)
      yolo_total = row.get("yolo_total")
      if not row.get(coverage_field) or value in (None, "") or yolo_total is None:
        continue
      pairs.append((float(yolo_total), float(value)))
    result[field] = {
        "pair_count": len(pairs),
        "pearson_r_with_yolo_total": pearson_correlation(pairs),
    }
  return result


def main() -> int:
  args = parse_args()
  run_dir = args.run_dir.expanduser().resolve()
  events = parse_runtime(run_dir / "runtime.log")
  full_timestamps, full_rows = read_telemetry(run_dir / "jetson_telemetry.csv")
  fast_timestamps, fast_rows = read_telemetry(run_dir / "gpu_fast.csv")

  correlated: list[dict[str, object]] = []
  for event in events:
    midpoint_ns = (int(event["yolo_start_unix_ns"]) + int(event["yolo_end_unix_ns"])) // 2
    full_row, full_delta_ms = nearest_row(full_timestamps, full_rows, midpoint_ns)
    fast_row, fast_delta_ms = nearest_row(fast_timestamps, fast_rows, midpoint_ns)
    result = dict(event)
    result["yolo_midpoint_unix_ns"] = midpoint_ns
    result["full_nearest_delta_ms"] = full_delta_ms
    result["full_telemetry_covered"] = full_delta_ms is not None and full_delta_ms <= args.max_nearest_ms
    result["fast_nearest_delta_ms"] = fast_delta_ms
    result["fast_telemetry_covered"] = fast_delta_ms is not None and fast_delta_ms <= args.max_nearest_ms
    result.update(prefixed_values("full", full_row))
    result.update(prefixed_values("fast", fast_row))
    correlated.append(result)

  write_csv(run_dir / "events.csv", events)
  write_csv(run_dir / "correlated_events.csv", correlated)

  yolo_totals = [float(row["yolo_total"]) for row in events if "yolo_total" in row]
  foundation_totals = [float(row["foundation_total"]) for row in events if "foundation_total" in row]
  compute_totals = [
      float(row["yolo_foundation_compute_total"])
      for row in events
      if "yolo_foundation_compute_total" in row
  ]
  consume_delays = [
      float(row["detection_consume_delay"])
      for row in events
      if "detection_consume_delay" in row
  ]
  init_totals = [float(row["init_total"]) for row in events if "init_total" in row]
  full_deltas = [float(row["full_nearest_delta_ms"]) for row in correlated if row["full_nearest_delta_ms"] is not None]
  fast_deltas = [float(row["fast_nearest_delta_ms"]) for row in correlated if row["fast_nearest_delta_ms"] is not None]
  yolo_p95 = percentile(yolo_totals, 95.0)
  slow_events = [
      row for row in events
      if yolo_p95 is not None and float(row.get("yolo_total", -1.0)) >= yolo_p95
  ]
  candidate_groups = {}
  for candidate_count in sorted({int(row["yolo_candidate_count"]) for row in events if "yolo_candidate_count" in row}):
    candidate_groups[str(candidate_count)] = numeric_stats([
        float(row["yolo_total"])
        for row in events
        if row.get("yolo_candidate_count") == candidate_count and "yolo_total" in row
    ])
  summary = {
      "event_count": len(events),
      "full_telemetry_sample_count": len(full_rows),
      "fast_telemetry_sample_count": len(fast_rows),
      "full_covered_event_count": sum(bool(row["full_telemetry_covered"]) for row in correlated),
      "fast_covered_event_count": sum(bool(row["fast_telemetry_covered"]) for row in correlated),
      "max_nearest_ms": args.max_nearest_ms,
        "yolo_total_ms": numeric_stats(yolo_totals),
        "foundation_total_ms": numeric_stats(foundation_totals),
        "yolo_foundation_compute_total_ms": numeric_stats(compute_totals),
        "detection_consume_delay_ms": numeric_stats(consume_delays),
        "init_total_ms": numeric_stats(init_totals),
        "candidate_count_yolo_total_ms": candidate_groups,
        "p95_slow_events": {
          "threshold_ms": yolo_p95,
          "count": len(slow_events),
          "postprocess_dominated_count": sum(
            float(row.get("yolo_postprocess_internal", 0.0))
            >= float(row.get("yolo_inference", 0.0))
            for row in slow_events
          ),
          "inference_dominated_count": sum(
            float(row.get("yolo_inference", 0.0))
            > float(row.get("yolo_postprocess_internal", 0.0))
            for row in slow_events
          ),
      },
        "telemetry_correlations": telemetry_correlations(correlated),
      "full_nearest_delta_ms_max": max(full_deltas) if full_deltas else None,
      "fast_nearest_delta_ms_max": max(fast_deltas) if fast_deltas else None,
  }
  (run_dir / "correlation_summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
