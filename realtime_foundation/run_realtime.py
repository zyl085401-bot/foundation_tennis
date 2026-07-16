from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import threading
import time

import cv2
import numpy as np
import yaml

from camera.realsense_reader import RealSenseReader
from detection.yolo_segmenter import YoloSegmenter
from tracking.foundationpose_tracker import FoundationPoseRealtimeTracker


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class LatestFrameCamera:
  def __init__(self, camera: RealSenseReader):
    self.camera = camera
    self.lock = threading.Lock()
    self.stop_event = threading.Event()
    self.thread = threading.Thread(target=self._capture_loop, name="realsense-capture", daemon=True)
    self.latest_frame = None
    self.latest_frame_id = 0

  def start(self) -> None:
    self.camera.start()
    self.thread.start()

  def stop(self) -> None:
    self.stop_event.set()
    self.camera.stop()
    if self.thread.is_alive():
      self.thread.join(timeout=2.0)

  def get_latest(self):
    with self.lock:
      return self.latest_frame_id, self.latest_frame

  def _capture_loop(self) -> None:
    while not self.stop_event.is_set():
      frame = self.camera.get_frame()
      if frame is None:
        continue
      with self.lock:
        self.latest_frame = frame
        self.latest_frame_id += 1


def load_config(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as file:
    return yaml.safe_load(file)


def resolve_path(path: str | None) -> str | None:
  if path is None or os.path.isabs(path):
    return path
  return os.path.abspath(os.path.join(REPO_ROOT, path))


def should_check_detection(frame_index: int, interval: int) -> bool:
  return interval > 0 and frame_index % interval == 0


def save_pose(path: str, pose: np.ndarray) -> None:
  output_dir = os.path.dirname(path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  np.savetxt(path, pose.reshape(4, 4))


def reset_workspace_dir(path: str) -> None:
  if os.path.commonpath([REPO_ROOT, path]) != REPO_ROOT:
    raise ValueError(f"output directory must be inside workspace: {path}")
  if os.path.isdir(path):
    shutil.rmtree(path)
  os.makedirs(path, exist_ok=True)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
  ys, xs = np.where(mask > 0)
  if len(xs) == 0:
    return None
  return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def estimate_mask_translation(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> list[float] | None:
  ys, xs = np.where(mask > 0)
  if len(xs) == 0:
    return None
  valid = (mask > 0) & (depth >= 0.001)
  if not valid.any():
    return None
  uc = (xs.min() + xs.max()) / 2.0
  vc = (ys.min() + ys.max()) / 2.0
  zc = float(np.median(depth[valid]))
  center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1.0]).reshape(3, 1)) * zc
  return [float(value) for value in center.reshape(3)]


def depth_to_visualization(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
  valid = depth >= 0.001
  if mask is not None:
    valid = valid & (mask > 0)
  if not valid.any():
    return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
  low, high = np.percentile(depth[valid], [2, 98])
  if high <= low:
    high = low + 1e-6
  depth_u8 = np.clip((depth - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
  return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def tensor_or_array_to_list(value, limit: int | None = None) -> list[float]:
  if value is None:
    return []
  if hasattr(value, "detach"):
    array = value.detach().cpu().numpy()
  else:
    array = np.asarray(value)
  array = np.asarray(array).reshape(-1)
  if limit is not None:
    array = array[:limit]
  return [float(item) for item in array]


def save_init_diagnostics(
    output_dir: str,
    init_count: int,
    frame_index: int,
    color: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    detection,
    tracker: FoundationPoseRealtimeTracker,
    pose: np.ndarray,
    timing: dict,
    extra: dict | None = None,
) -> None:
  prefix = f"init_{init_count:06d}_frame_{frame_index:06d}"
  mask = detection.mask.astype(np.uint8)
  rendered_mask = tracker.render_pose_mask(K, color.shape[:2]).astype(np.uint8)
  target_mask = mask > 0
  rendered_bool = rendered_mask > 0
  intersection = int(np.logical_and(rendered_bool, target_mask).sum())
  union = int(np.logical_or(rendered_bool, target_mask).sum())
  rendered_iou = float(intersection / union) if union > 0 else 0.0
  valid_depth = depth[(mask > 0) & (depth >= 0.001)]
  scores_top5 = tensor_or_array_to_list(getattr(tracker.estimator, "scores", None), limit=5)
  top1_top2_gap = None
  if len(scores_top5) >= 2:
    top1_top2_gap = float(scores_top5[0] - scores_top5[1])

  cv2.imwrite(os.path.join(output_dir, f"{prefix}_rgb.png"), color[..., ::-1])
  cv2.imwrite(os.path.join(output_dir, f"{prefix}_mask.png"), mask * 255)
  cv2.imwrite(os.path.join(output_dir, f"{prefix}_rendered_mask.png"), rendered_mask * 255)
  cv2.imwrite(os.path.join(output_dir, f"{prefix}_depth.png"), depth_to_visualization(depth, mask))

  diagnostics = {
      "init_count": int(init_count),
      "frame_index": int(frame_index),
      "mask_area": int(detection.area),
      "mask_bbox_xyxy": mask_bbox(mask),
      "detection_box_xyxy": [float(value) for value in detection.box_xyxy],
      "detection_confidence": float(detection.confidence),
      "valid_depth_pixels": int(len(valid_depth)),
      "mask_median_depth_m": float(np.median(valid_depth)) if len(valid_depth) > 0 else None,
      "guess_translation_from_mask_m": estimate_mask_translation(depth, mask, K),
      "pose_translation_m": [float(value) for value in pose.reshape(4, 4)[:3, 3]],
      "estimator_pose_last_translation_m": tensor_or_array_to_list(getattr(tracker.estimator, "pose_last", None), limit=16)[3:12:4],
      "top5_scores": scores_top5,
      "top1_top2_score_gap": top1_top2_gap,
      "pose_hypothesis_candidates": int(timing.get("pose_hypothesis_candidates", 0)),
      "coarse_score_filter": timing.get("coarse_score_filter"),
      "coarse_score_candidates": int(timing.get("coarse_score_candidates", 0)),
      "rendered_mask_iou_with_yolo_mask": rendered_iou,
      "rendered_mask_intersection_pixels": intersection,
      "rendered_mask_union_pixels": union,
      "timing_seconds": {key: float(value) for key, value in timing.items() if isinstance(value, (int, float))},
  }
  if extra:
    diagnostics.update(extra)
  with open(os.path.join(output_dir, f"{prefix}.json"), "w", encoding="utf-8") as file:
    json.dump(diagnostics, file, indent=2, ensure_ascii=False)


def count_valid_depth_pixels(depth: np.ndarray, mask: np.ndarray, min_depth: float = 0.001) -> int:
  return int(((mask > 0) & (depth >= min_depth)).sum())


def mask_center(mask: np.ndarray) -> tuple[float, float] | None:
  ys, xs = np.where(mask > 0)
  if len(xs) == 0:
    return None
  return float(xs.mean()), float(ys.mean())


def detection_reject_reason(detection, depth: np.ndarray, runtime_cfg: dict) -> str | None:
  min_conf = float(runtime_cfg.get("min_detection_conf", 0.0))
  if detection.confidence < min_conf:
    return f"confidence {detection.confidence:.3f} < {min_conf:.3f}"

  min_area = int(runtime_cfg.get("min_detection_area", 0))
  max_area = int(runtime_cfg.get("max_detection_area", 0))
  if detection.area < min_area:
    return f"mask area {detection.area} < {min_area}"
  if max_area > 0 and detection.area > max_area:
    return f"mask area {detection.area} > {max_area}"

  center = mask_center(detection.mask)
  if center is None:
    return "empty mask"
  center_x, center_y = center
  roi = runtime_cfg.get("detection_roi_xyxy")
  if roi is not None:
    x1, y1, x2, y2 = [float(value) for value in roi]
    if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
      return f"mask center ({center_x:.1f}, {center_y:.1f}) outside ROI {roi}"

  valid_depth = depth[(detection.mask > 0) & (depth >= 0.001)]
  if valid_depth.size == 0:
    return "no valid depth inside mask"
  median_depth = float(np.median(valid_depth))
  min_median_depth = float(runtime_cfg.get("min_mask_median_depth", 0.0))
  max_median_depth = float(runtime_cfg.get("max_mask_median_depth", 0.0))
  if min_median_depth > 0 and median_depth < min_median_depth:
    return f"median depth {median_depth:.3f} < {min_median_depth:.3f}"
  if max_median_depth > 0 and median_depth > max_median_depth:
    return f"median depth {median_depth:.3f} > {max_median_depth:.3f}"

  return None


def detection_border_reject_reason(detection, image_shape: tuple[int, int], margin: int) -> str | None:
  if margin <= 0:
    return None

  height, width = image_shape
  x1, y1, x2, y2 = [float(value) for value in detection.box_xyxy]
  if x1 < margin:
    return f"bbox left edge {x1:.1f} < margin {margin}"
  if y1 < margin:
    return f"bbox top edge {y1:.1f} < margin {margin}"
  if x2 > width - margin:
    return f"bbox right edge {x2:.1f} > width-margin {width - margin}"
  if y2 > height - margin:
    return f"bbox bottom edge {y2:.1f} > height-margin {height - margin}"
  return None


def register_stability_reject_reason(
    detection,
    stable_areas: list[int],
    required_count: int,
    max_area_change: float,
) -> str | None:
  required_count = max(1, required_count)
  stable_areas.append(int(detection.area))
  if len(stable_areas) > required_count:
    del stable_areas[0:len(stable_areas) - required_count]

  if len(stable_areas) < required_count:
    return f"stable detections {len(stable_areas)}/{required_count}"

  if required_count <= 1 or max_area_change <= 0:
    return None

  min_area = min(stable_areas)
  max_area = max(stable_areas)
  area_change = (max_area - min_area) / max(max_area, 1)
  if area_change > max_area_change:
    return f"mask area change {area_change:.2f} > {max_area_change:.2f} across recent detections"

  return None


def should_log_status(frame_index: int, interval: int) -> bool:
  return interval > 0 and frame_index % interval == 0


def format_seconds(value, fallback: str = "N/A") -> str:
  if value is None:
    return fallback
  return f"{float(value):.3f}s"


def seconds_to_ms(value):
  if value is None:
    return None
  return float(value) * 1000.0


def print_timing_summary(init_index: int, rows: list[tuple[str, float | None]]) -> None:
  stage_width = 44
  value_width = 12
  line_width = stage_width + value_width + 1
  print(f"[TIMER][SUMMARY][foundationpose_init] init={init_index} unit=ms")
  print(f"{'stage':<{stage_width}}{'time_ms':>{value_width}}")
  print("-" * line_width)
  for stage, value in rows:
    if stage == "":
      print()
      continue
    if set(stage) == {"-"}:
      print(stage)
      continue
    value_text = "N/A" if value is None else f"{float(value):.3f}"
    print(f"{stage:<{stage_width}}{value_text:>{value_width}}")


def log_runtime(enabled: bool, message: str) -> None:
  if enabled:
    print(message)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
  args = parser.parse_args()
  cfg = load_config(args.config)

  camera_cfg = cfg.get("camera", {})
  yolo_cfg = cfg.get("yolo", {})
  tracker_cfg = cfg.get("foundationpose", {})
  runtime_cfg = cfg.get("runtime", {})
  runtime_mode = str(runtime_cfg.get("mode", "realtime")).lower()
  init_only = runtime_mode in ("init_only", "initialize_only", "register_only")
  success_timing_only = bool(runtime_cfg.get("success_timing_only", False))
  verbose_runtime = not success_timing_only

  camera = RealSenseReader(
      width=int(camera_cfg.get("width", 640)),
      height=int(camera_cfg.get("height", 480)),
      fps=int(camera_cfg.get("fps", 30)),
      serial=camera_cfg.get("serial"),
      depth_min=float(camera_cfg.get("depth_min", 0.001)),
      depth_max=float(camera_cfg.get("depth_max", 3.0)),
      align_to_color=bool(camera_cfg.get("align_to_color", True)),
  )
  detector = YoloSegmenter(
      weights=resolve_path(yolo_cfg["weights"]),
      target_class=yolo_cfg.get("target_class"),
      target_class_id=yolo_cfg.get("target_class_id"),
      conf=float(yolo_cfg.get("conf", 0.35)),
      imgsz=int(yolo_cfg.get("imgsz", 640)),
      device=yolo_cfg.get("device"),
      half=bool(yolo_cfg.get("half", True)),
      min_mask_area=int(yolo_cfg.get("min_mask_area", 100)),
      morph_kernel=int(yolo_cfg.get("morph_kernel", 5)),
  )
  tracker = FoundationPoseRealtimeTracker(
      mesh_file=resolve_path(tracker_cfg["mesh_file"]),
      debug_dir=resolve_path(tracker_cfg.get("debug_dir", "realtime_foundation/outputs/debug")),
      debug=int(tracker_cfg.get("debug", 1)),
      use_float32_crop_window_patch=bool(tracker_cfg.get("use_float32_crop_window_patch", True)),
      init_min_n_views=int(tracker_cfg.get("init_min_n_views", 40)),
      init_inplane_step=int(tracker_cfg.get("init_inplane_step", 60)),
      est_refine_iter=int(tracker_cfg.get("est_refine_iter", 5)),
      init_strategy=tracker_cfg.get("init_strategy", "default"),
      coarse_refine_iter=int(tracker_cfg.get("coarse_refine_iter", 1)),
      coarse_score_filter=tracker_cfg.get("coarse_score_filter", "none"),
      coarse_score_top_k=int(tracker_cfg.get("coarse_score_top_k", 999999)),
      fine_refine_iter=int(tracker_cfg.get("fine_refine_iter", 2)),
      fine_top_k=int(tracker_cfg.get("fine_top_k", 16)),
      track_refine_iter=int(tracker_cfg.get("track_refine_iter", 2)),
      vis_mode=tracker_cfg.get("vis_mode", "box"),
      contour_thickness=int(tracker_cfg.get("contour_thickness", 3)),
      axis_scale=float(tracker_cfg.get("axis_scale", 0.1)),
  )
  if success_timing_only:
    logging.getLogger().setLevel(logging.WARNING)

  detection_interval = int(runtime_cfg.get("detection_interval", 15))
  relocalize_iou = float(runtime_cfg.get("relocalize_iou", 0.25))
  min_valid_depth_pixels = int(runtime_cfg.get("min_valid_depth_pixels", 500))
  max_missing_detections = int(runtime_cfg.get("max_missing_detections", 3))
  status_log_interval = int(runtime_cfg.get("status_log_interval", 30))
  register_retry_interval = int(runtime_cfg.get("register_retry_interval", 30))
  register_required_stable_detections = int(runtime_cfg.get("register_required_stable_detections", 1))
  register_max_area_change = float(runtime_cfg.get("register_max_area_change", 0.0))
  register_border_margin = int(runtime_cfg.get("register_border_margin", 0))
  register_min_render_iou = float(runtime_cfg.get("register_min_render_iou", 0.0))
  register_max_translation_drift = float(runtime_cfg.get("register_max_translation_drift", 0.0))
  show_window = bool(runtime_cfg.get("show_window", True))
  save_init_vis = bool(runtime_cfg.get("save_init_vis", False))
  save_init_vis_dir = resolve_path(runtime_cfg.get("save_init_vis_dir", "realtime_foundation/outputs/debug/init_vis"))
  if save_init_vis:
    reset_workspace_dir(save_init_vis_dir)
  save_init_diagnostics_enabled = bool(runtime_cfg.get("save_init_diagnostics", False))
  save_init_diagnostics_dir = resolve_path(runtime_cfg.get("save_init_diagnostics_dir", "realtime_foundation/outputs/debug/init_diagnostics"))
  if save_init_diagnostics_enabled:
    reset_workspace_dir(save_init_diagnostics_dir)
  pose_output = resolve_path(runtime_cfg.get("pose_output", "realtime_foundation/outputs/latest_pose.txt"))

  frame_index = 0
  last_processed_frame_id = 0
  init_count = 0
  last_success_time = None
  last_detection = None
  current_frame_detection = None
  missing_detection_count = 0
  register_stable_areas = []
  next_register_frame = 0
  frame_buffer = LatestFrameCamera(camera)
  frame_buffer.start()
  try:
    while True:
      frame_id, frame = frame_buffer.get_latest()
      if frame is None or frame_id == last_processed_frame_id:
        time.sleep(0.001)
        continue
      last_processed_frame_id = frame_id
      color, depth, K = frame
      frame_index = frame_id
      current_frame_detection = None

      if not tracker.initialized:
        if frame_index < next_register_frame:
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          continue

        init_start = time.perf_counter()
        yolo_start = time.perf_counter()
        detection = detector.predict_mask(color)
        yolo_time = time.perf_counter() - yolo_start
        yolo_timing = getattr(detector, "last_timing", {})
        current_frame_detection = detection
        if detection is None:
          register_stable_areas.clear()
          if should_log_status(frame_index, status_log_interval):
            log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: waiting for YOLO target mask")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          continue

        validation_start = time.perf_counter()
        valid_depth_pixels = count_valid_depth_pixels(depth, detection.mask)
        log_runtime(
            verbose_runtime,
            f"[Realtime] Frame {frame_index}: YOLO target class={detection.class_name} "
            f"conf={detection.confidence:.3f}, mask_area={detection.area}, valid_depth={valid_depth_pixels}",
        )
        reject_reason = detection_reject_reason(detection, depth, runtime_cfg)
        if reject_reason is not None:
          register_stable_areas.clear()
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: reject YOLO detection: {reject_reason}")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          continue
        if valid_depth_pixels < min_valid_depth_pixels:
          register_stable_areas.clear()
          log_runtime(
              verbose_runtime,
              f"[Realtime] Frame {frame_index}: valid depth inside mask is too small "
              f"({valid_depth_pixels} < {min_valid_depth_pixels}); waiting",
          )
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          continue

        border_reject_reason = detection_border_reject_reason(detection, color.shape[:2], register_border_margin)
        if border_reject_reason is not None:
          register_stable_areas.clear()
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: wait for full target before register: {border_reject_reason}")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          continue

        stability_reject_reason = register_stability_reject_reason(
            detection,
            register_stable_areas,
            register_required_stable_detections,
            register_max_area_change,
        )
        if stability_reject_reason is not None:
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: wait for stable YOLO mask before register: {stability_reject_reason}")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          continue

        last_detection = detection
        missing_detection_count = 0
        register_stable_areas.clear()
        validation_time = time.perf_counter() - validation_start
        log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: registering FoundationPose")
        try:
          register_start = time.perf_counter()
          pose_result = tracker.register(color, depth, K, detection.mask)
          register_wall_time = time.perf_counter() - register_start
        except Exception as exc:
          tracker.reset()
          next_register_frame = frame_index + register_retry_interval
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: register failed, waiting for next detection: {exc}")
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: retry register after frame {next_register_frame}")
          continue
        timing = getattr(tracker.estimator, "last_register_timing", {})
        register_render_iou = None
        register_translation_drift = None
        mask_translation = estimate_mask_translation(depth, detection.mask, K)
        if register_max_translation_drift > 0 and mask_translation is not None:
          pose_translation = pose_result.pose.reshape(4, 4)[:3, 3]
          register_translation_drift = float(np.linalg.norm(pose_translation - np.asarray(mask_translation, dtype=np.float32)))
          if register_translation_drift > register_max_translation_drift:
            if save_init_diagnostics_enabled:
              save_init_diagnostics(
                  save_init_diagnostics_dir,
                  init_count + 1,
                  frame_index,
                  color,
                  depth,
                  K,
                  detection,
                  tracker,
                  pose_result.pose,
                  timing,
                  extra={
                      "accepted": False,
                      "reject_reason": "translation_drift_above_threshold",
                      "register_max_translation_drift": register_max_translation_drift,
                      "register_translation_drift": register_translation_drift,
                  },
              )
            tracker.reset()
            log_runtime(
                verbose_runtime,
                f"[Realtime] Frame {frame_index}: reject initialization: translation_drift "
                f"{register_translation_drift:.3f} > {register_max_translation_drift:.3f}",
            )
            continue
        if register_min_render_iou > 0:
          register_render_iou = tracker.mask_iou(K, color.shape[:2], detection.mask)
          if register_render_iou < register_min_render_iou:
            if save_init_diagnostics_enabled:
              save_init_diagnostics(
                  save_init_diagnostics_dir,
                  init_count + 1,
                  frame_index,
                  color,
                  depth,
                  K,
                  detection,
                  tracker,
                  pose_result.pose,
                  timing,
                  extra={
                      "accepted": False,
                      "reject_reason": "rendered_mask_iou_below_threshold",
                      "register_render_iou_threshold": register_min_render_iou,
                      "register_max_translation_drift": register_max_translation_drift,
                      "register_translation_drift": register_translation_drift,
                  },
              )
            tracker.reset()
            log_runtime(
                verbose_runtime,
                f"[Realtime] Frame {frame_index}: reject initialization: rendered_mask_iou "
                f"{register_render_iou:.3f} < {register_min_render_iou:.3f}",
            )
            continue
        init_count += 1
        init_total_time = time.perf_counter() - init_start
        success_time = time.perf_counter()
        success_period = None if last_success_time is None else success_time - last_success_time
        last_success_time = success_time
        register_time = float(timing.get("register", register_wall_time))
        refiner_time = timing.get("refiner")
        refiner_detail = timing.get("refiner_detail", {})
        refiner_coarse_time = timing.get("refiner_coarse")
        refiner_fine_time = timing.get("refiner_fine")
        scorer_time = timing.get("scorer")
        scorer_coarse_time = timing.get("scorer_coarse")
        scorer_fine_time = timing.get("scorer_fine")
        depth_preprocess_time = timing.get("depth_preprocess")
        pose_hypothesis_time = timing.get("pose_hypothesis")
        coarse_score_select_time = timing.get("coarse_score_select")
        topk_select_time = timing.get("topk_select")
        sort_select_time = timing.get("sort_select")
        foundation_other_time = timing.get("other")
        success_hz = None if success_period is None else 1.0 / max(success_period, 1e-6)
        log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: FoundationPose initialized")
        print_timing_summary(
            init_count,
            [
                ("yolo_total", seconds_to_ms(yolo_timing.get("total", yolo_time))),
                ("  yolo_model_predict", seconds_to_ms(yolo_timing.get("model_predict"))),
                ("  yolo_tensor_to_cpu", seconds_to_ms(yolo_timing.get("tensor_to_cpu"))),
                ("  yolo_mask_resize_clean", seconds_to_ms(yolo_timing.get("mask_resize_clean"))),
                ("  yolo_select_best_mask", seconds_to_ms(yolo_timing.get("select_best_mask"))),
                ("", None),
                ("foundation_total", seconds_to_ms(register_time)),
                ("  foundation_depth_preprocess", seconds_to_ms(depth_preprocess_time)),
                ("  foundation_pose_hypothesis", seconds_to_ms(pose_hypothesis_time)),
                ("  foundation_refiner", seconds_to_ms(refiner_time)),
                ("    refiner_coarse", seconds_to_ms(refiner_coarse_time)),
                ("    refiner_fine", seconds_to_ms(refiner_fine_time)),
                ("    refiner_crop_window", seconds_to_ms(refiner_detail.get("crop_window"))),
                ("    refiner_render", seconds_to_ms(refiner_detail.get("render"))),
                ("    refiner_render_postprocess", seconds_to_ms(refiner_detail.get("render_postprocess"))),
                ("    refiner_warp", seconds_to_ms(refiner_detail.get("warp"))),
                ("    refiner_transform", seconds_to_ms(refiner_detail.get("transform"))),
                ("    refiner_input_pack", seconds_to_ms(refiner_detail.get("input_pack"))),
                ("    refiner_network_forward", seconds_to_ms(refiner_detail.get("network_forward"))),
                ("    refiner_pose_update", seconds_to_ms(refiner_detail.get("pose_update"))),
                ("    refiner_other", seconds_to_ms(refiner_detail.get("other"))),
                ("  foundation_scorer", seconds_to_ms(scorer_time)),
                ("    coarse_score_select", seconds_to_ms(coarse_score_select_time)),
                ("    scorer_coarse", seconds_to_ms(scorer_coarse_time)),
                ("    scorer_fine", seconds_to_ms(scorer_fine_time)),
                ("  foundation_topk_select", seconds_to_ms(topk_select_time)),
                ("  foundation_sort_select", seconds_to_ms(sort_select_time)),
                ("  foundation_other", seconds_to_ms(foundation_other_time)),
                ("", None),
                ("init_total", seconds_to_ms(init_total_time)),
                ("success_period", seconds_to_ms(success_period)),
                ("-" * 57, None),
                ("success_rate_hz", success_hz),
            ],
        )
      else:
        try:
          pose_result = tracker.track(color, depth, K)
        except Exception as exc:
          tracker.reset()
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: tracking failed, waiting for detection: {exc}")
          continue
        if should_check_detection(frame_index, detection_interval):
          detection = detector.predict_mask(color)
          current_frame_detection = detection
          if detection is None:
            missing_detection_count += 1
            log_runtime(
                verbose_runtime,
                f"[Realtime] Frame {frame_index}: YOLO target missing during tracking "
                f"({missing_detection_count}/{max_missing_detections})",
            )
            if missing_detection_count >= max_missing_detections:
              tracker.reset()
              last_detection = None
              log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: target lost, stop FoundationPose tracking and wait for YOLO detection")
              continue
          else:
            missing_detection_count = 0
            last_detection = detection
            log_runtime(
                verbose_runtime,
                f"[Realtime] Frame {frame_index}: YOLO target present during tracking "
                f"class={detection.class_name} conf={detection.confidence:.3f}, keep FoundationPose tracking",
            )

      save_pose(pose_output, pose_result.pose)

      if save_init_diagnostics_enabled and pose_result.mode == "register" and current_frame_detection is not None:
        save_init_diagnostics(
            save_init_diagnostics_dir,
            init_count,
            frame_index,
            color,
            depth,
            K,
            current_frame_detection,
            tracker,
            pose_result.pose,
            getattr(tracker.estimator, "last_register_timing", {}),
            extra={
              "accepted": True,
              "register_render_iou_threshold": register_min_render_iou,
              "register_render_iou_checked": register_min_render_iou > 0,
              "register_max_translation_drift": register_max_translation_drift,
              "register_translation_drift": register_translation_drift,
            },
        )

      if show_window or (save_init_vis and pose_result.mode == "register"):
        vis = tracker.draw_visualization(color, K, pose_result.pose)
        if current_frame_detection is not None:
          mask_overlay = current_frame_detection.mask.astype(bool)
          vis[mask_overlay] = (0.65 * vis[mask_overlay] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        if save_init_vis and pose_result.mode == "register":
          image_path = os.path.join(save_init_vis_dir, f"init_{init_count:06d}_frame_{frame_index:06d}.png")
          cv2.imwrite(image_path, vis[..., ::-1])
        if show_window:
          cv2.imshow("realtime_foundation", vis[..., ::-1])
          if cv2.waitKey(1) in (27, ord("q")):
            break

      if init_only and pose_result.mode == "register":
        tracker.reset()
        last_detection = None
        missing_detection_count = 0
        register_stable_areas.clear()
        continue

      time.sleep(float(runtime_cfg.get("loop_sleep", 0.0)))
  finally:
    frame_buffer.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()