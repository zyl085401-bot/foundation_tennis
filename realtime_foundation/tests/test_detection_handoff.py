from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest

import numpy as np


def load_runtime_module():
  stubs = {
      "cv2": types.ModuleType("cv2"),
      "numpy": types.ModuleType("numpy"),
      "yaml": types.ModuleType("yaml"),
      "camera": types.ModuleType("camera"),
      "camera.realsense_reader": types.ModuleType("camera.realsense_reader"),
      "detection": types.ModuleType("detection"),
      "detection.yolo_segmenter": types.ModuleType("detection.yolo_segmenter"),
      "mros_pose_publisher": types.ModuleType("mros_pose_publisher"),
      "tracking": types.ModuleType("tracking"),
      "tracking.foundationpose_tracker": types.ModuleType("tracking.foundationpose_tracker"),
  }
  stubs["camera.realsense_reader"].RealSenseReader = object
  stubs["detection.yolo_segmenter"].YoloSegmenter = object
  stubs["mros_pose_publisher"].MrosPosePublisher = object
  stubs["tracking.foundationpose_tracker"].FoundationPoseRealtimeTracker = object
  previous = {name: sys.modules.get(name) for name in stubs}
  sys.modules.update(stubs)
  try:
    module_path = Path(__file__).resolve().parents[1] / "run_realtime.py"
    spec = importlib.util.spec_from_file_location("run_realtime_sync_test", module_path)
    if spec is None or spec.loader is None:
      raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
  finally:
    for name, old_module in previous.items():
      if old_module is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = old_module


runtime = load_runtime_module()


def load_analyzer_module():
  module_path = Path(__file__).resolve().parents[1] / "tools" / "analyze_jetson_telemetry.py"
  spec = importlib.util.spec_from_file_location("analyze_jetson_telemetry_test", module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {module_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


analyzer = load_analyzer_module()


class FakeDetector:
  def __init__(self, results):
    self.results = list(results)
    self.calls = 0
    self.call_condition = threading.Condition()
    self.last_timing = {"total": 0.001}
    self.last_candidate_count = 1

  def predict_mask(self, _color):
    with self.call_condition:
      index = self.calls
      self.calls += 1
      self.call_condition.notify_all()
    result = self.results[index] if index < len(self.results) else None
    if isinstance(result, BaseException):
      raise result
    return result

  def wait_for_calls(self, count: int, timeout: float = 1.0) -> bool:
    with self.call_condition:
      return self.call_condition.wait_for(lambda: self.calls >= count, timeout=timeout)


def frame(frame_id: int):
  return runtime.FrameMessage(
      frame_id=frame_id,
      timestamp=float(frame_id),
      color=object(),
      depth=object(),
      K=object(),
  )


class LatestTopicTests(unittest.TestCase):
  def test_sequence_wait_uses_state_not_notification_history(self):
    topic = runtime.LatestTopic("test")
    first_sequence = topic.publish("first")
    self.assertEqual((first_sequence, "first"), topic.wait_for_newer(0, timeout=0.0))
    self.assertIsNone(topic.wait_for_newer(first_sequence, timeout=0.01))
    second_sequence = topic.publish("second")
    self.assertEqual((second_sequence, "second"), topic.wait_for_newer(first_sequence, timeout=0.0))


class PrecisionFallbackConfigTests(unittest.TestCase):
  def test_refiner_candidate_engine_paths_preserve_order(self):
    backends = runtime.resolve_tensorrt_backends({
        "tensorrt": {
            "refiner": {
                "candidate_engine_paths": {
                    "refiner_coarse": {
                        5: ["n5_int8.engine", "n5_fp16.engine"],
                    },
                },
            },
        },
    })
    self.assertEqual(
        [
            runtime.resolve_path("n5_int8.engine"),
            runtime.resolve_path("n5_fp16.engine"),
        ],
        backends["refiner"]["candidate_engine_paths"]["refiner_coarse"][5],
    )

  def test_yolo_int8_io_prefers_int8_io_then_existing_int8_then_fp16(self):
    candidates = runtime.resolve_yolo_weight_candidates({
        "precision": "int8",
        "io_precision": "int8",
        "int8_io_weights": "int8_io.engine",
        "int8_weights": "int8_fp32_io.engine",
        "weights": "fp16.engine",
        "fallback_to_fp32_io": True,
        "fallback_to_fp16": True,
    })
    self.assertEqual(
        [
            runtime.resolve_path("int8_io.engine"),
            runtime.resolve_path("int8_fp32_io.engine"),
            runtime.resolve_path("fp16.engine"),
        ],
        candidates,
    )

  def test_yolo_int8_io_can_disable_fp32_io_fallback(self):
    candidates = runtime.resolve_yolo_weight_candidates({
        "precision": "int8",
        "io_precision": "int8",
        "int8_io_weights": "int8_io.engine",
        "int8_weights": "int8_fp32_io.engine",
        "weights": "fp16.engine",
        "fallback_to_fp32_io": False,
        "fallback_to_fp16": True,
    })
    self.assertEqual(
        [runtime.resolve_path("int8_io.engine"), runtime.resolve_path("fp16.engine")],
        candidates,
    )

  def test_yolo_fp32_io_mode_skips_int8_io_engine(self):
    candidates = runtime.resolve_yolo_weight_candidates({
        "precision": "int8",
        "io_precision": "fp32",
        "int8_io_weights": "int8_io.engine",
        "int8_weights": "int8_fp32_io.engine",
        "weights": "fp16.engine",
        "fallback_to_fp16": False,
    })
    self.assertEqual([runtime.resolve_path("int8_fp32_io.engine")], candidates)

  def test_yolo_rejects_unknown_io_precision(self):
    with self.assertRaisesRegex(ValueError, "I/O precision"):
      runtime.resolve_yolo_weight_candidates({
          "precision": "int8",
          "io_precision": "uint8",
      })

  def test_yolo_int8_prefers_int8_then_fp16(self):
    candidates = runtime.resolve_yolo_weight_candidates({
        "precision": "int8",
        "int8_weights": "int8.engine",
        "weights": "fp16.engine",
        "fallback_to_fp16": True,
    })
    self.assertEqual(
        [runtime.resolve_path("int8.engine"), runtime.resolve_path("fp16.engine")],
        candidates,
    )

  def test_yolo_fp16_switch_skips_int8(self):
    candidates = runtime.resolve_yolo_weight_candidates({
        "precision": "fp16",
        "int8_weights": "int8.engine",
        "weights": "fp16.engine",
        "fallback_to_fp16": True,
    })
    self.assertEqual([runtime.resolve_path("fp16.engine")], candidates)

  def test_yolo_int8_can_disable_fp16_fallback(self):
    candidates = runtime.resolve_yolo_weight_candidates({
        "precision": "int8",
        "int8_weights": "int8.engine",
        "weights": "fp16.engine",
        "fallback_to_fp16": False,
    })
    self.assertEqual([runtime.resolve_path("int8.engine")], candidates)


class FoundationPoseConfigTests(unittest.TestCase):
  def test_build_tracker_propagates_disabled_fine_stage(self):
    class RecordingTracker:
      def __init__(self, **kwargs):
        self.kwargs = kwargs

    original_tracker = runtime.FoundationPoseRealtimeTracker
    runtime.FoundationPoseRealtimeTracker = RecordingTracker
    try:
      tracker = runtime.build_tracker({
          "mesh_file": "mesh.obj",
          "fine_stage_enabled": False,
          "fine_refine_iter": 1,
      })
    finally:
      runtime.FoundationPoseRealtimeTracker = original_tracker

    self.assertFalse(tracker.kwargs["fine_stage_enabled"])
    self.assertEqual(1, tracker.kwargs["fine_refine_iter"])

  def test_build_tracker_propagates_initial_candidate_selection(self):
    class RecordingTracker:
      def __init__(self, **kwargs):
        self.kwargs = kwargs

    original_tracker = runtime.FoundationPoseRealtimeTracker
    runtime.FoundationPoseRealtimeTracker = RecordingTracker
    try:
      tracker = runtime.build_tracker({
          "mesh_file": "mesh.obj",
          "initial_candidate_selection": {
              "enabled": True,
              "mode": "so3_farthest",
              "count": 5,
          },
      })
    finally:
      runtime.FoundationPoseRealtimeTracker = original_tracker

    self.assertEqual(
        {"enabled": True, "mode": "so3_farthest", "count": 5},
        tracker.kwargs["initial_candidate_selection"],
    )

  def test_build_tracker_propagates_translation_consensus(self):
    class RecordingTracker:
      def __init__(self, **kwargs):
        self.kwargs = kwargs

    config = {
        "enabled": True,
        "shadow_only": True,
        "median_distance_threshold_m": 0.004,
        "inlier_distance_threshold_m": 0.006,
        "min_inlier_count": 4,
        "fallback": "coarse_scorer",
    }
    original_tracker = runtime.FoundationPoseRealtimeTracker
    runtime.FoundationPoseRealtimeTracker = RecordingTracker
    try:
      tracker = runtime.build_tracker({
          "mesh_file": "mesh.obj",
          "translation_consensus": config,
      })
    finally:
      runtime.FoundationPoseRealtimeTracker = original_tracker

    self.assertEqual(config, tracker.kwargs["translation_consensus"])

  def test_build_tracker_propagates_candidate_predictability_shadow(self):
    class RecordingTracker:
      def __init__(self, **kwargs):
        self.kwargs = kwargs

    config = {
        "enabled": True,
        "expected_candidate_count": 5,
        "output_path": "outputs/phase3_0.jsonl",
        "summary_interval": 25,
    }
    original_tracker = runtime.FoundationPoseRealtimeTracker
    runtime.FoundationPoseRealtimeTracker = RecordingTracker
    try:
      tracker = runtime.build_tracker({
          "mesh_file": "mesh.obj",
          "candidate_predictability_shadow": config,
      })
    finally:
      runtime.FoundationPoseRealtimeTracker = original_tracker

    self.assertEqual(config, tracker.kwargs["candidate_predictability_shadow"])

  def test_build_tracker_propagates_single_candidate_mode(self):
    class RecordingTracker:
      def __init__(self, **kwargs):
        self.kwargs = kwargs

    config = {"enabled": True}
    original_tracker = runtime.FoundationPoseRealtimeTracker
    runtime.FoundationPoseRealtimeTracker = RecordingTracker
    try:
      tracker = runtime.build_tracker({
          "mesh_file": "mesh.obj",
          "single_candidate_mode": config,
      })
    finally:
      runtime.FoundationPoseRealtimeTracker = original_tracker

    self.assertEqual(config, tracker.kwargs["single_candidate_mode"])


class RegisterRecoveryTests(unittest.TestCase):
  @staticmethod
  def _quality(accepted: bool, reason: str | None, render_iou: float):
    return runtime.RegisterQualityResult(
        accepted=accepted,
        reject_reason=reason,
        translation_drift=None,
        render_iou=render_iou,
        top1_top2_score_gap=None,
    )

  def test_recovery_switch_disabled_does_not_retry(self):
    quality = self._quality(False, "rendered_mask_iou_below_threshold", 0.88)
    self.assertFalse(runtime.should_run_register_recovery(quality, {"enabled": False}))

  def test_enabled_recovery_retries_same_frame_and_uses_profile(self):
    class FakeTracker:
      def __init__(self):
        self.estimator = types.SimpleNamespace(last_register_timing={})
        self.profiles = []
        self.reset_count = 0

      def register(self, *args, registration_profile=None, **kwargs):
        self.profiles.append(registration_profile)
        self.estimator.last_register_timing = {
            "register": 0.01 if registration_profile is None else 0.02,
        }
        return types.SimpleNamespace(pose=np.eye(4, dtype=np.float32))

      def reset(self):
        self.reset_count += 1

    qualities = iter([
        self._quality(False, "rendered_mask_iou_below_threshold", 0.88),
        self._quality(True, None, 0.93),
    ])
    original_evaluate = runtime.evaluate_register_quality
    runtime.evaluate_register_quality = lambda **kwargs: next(qualities)
    tracker = FakeTracker()
    profile = {"initial_candidate_selection": {"enabled": True, "count": 5}}
    try:
      result = runtime.register_with_optional_recovery(
          tracker=tracker,
          color=np.zeros((2, 2, 3), dtype=np.uint8),
          depth=np.ones((2, 2), dtype=np.float32),
          K=np.eye(3, dtype=np.float32),
          mask=np.ones((2, 2), dtype=np.uint8),
          runtime_cfg={
              "register_recovery": {
                  "enabled": True,
                  "trigger_reasons": ["rendered_mask_iou_below_threshold"],
                  "profile": profile,
              },
          },
      )
    finally:
      runtime.evaluate_register_quality = original_evaluate

    self.assertEqual([None, profile], tracker.profiles)
    self.assertEqual(1, tracker.reset_count)
    self.assertTrue(result.recovery_attempted)
    self.assertTrue(result.recovery_succeeded)
    self.assertAlmostEqual(0.03, result.timing["register"])
    self.assertEqual("recovery", result.timing["register_final_attempt"])


class CandidatePredictabilityRecorderTests(unittest.TestCase):
  def test_recorder_persists_and_aggregates_completed_samples(self):
    original_numpy = runtime.np
    runtime.np = np
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "phase3_0.jsonl"
        recorder = runtime.CandidatePredictabilityRecorder(
            str(output_path),
            reset_on_start=True,
            summary_interval=2,
        )
        timing = {
            "candidate_predictability_status": "completed",
            "candidate_predictability": {
                "candidates": [
                    {
                        "candidate_id": 0,
                        "source_candidate_id": 7,
                        "selected_by_phase2": True,
                        "error_xy_m": 0.003,
                        "error_z_m": 0.004,
                        "error_translation_m": 0.005,
                    },
                ],
            },
            "translation_consensus_status": "fast_path",
            "translation_consensus_passed": True,
        }
        self.assertFalse(recorder.record(
            init_index=1,
            frame_id=10,
            timestamp=1.0,
            timing=timing,
            render_iou=0.95,
        ))
        self.assertTrue(recorder.record(
            init_index=2,
            frame_id=11,
            timestamp=2.0,
            timing=timing,
            render_iou=0.96,
        ))
        summary = recorder.summary()
        recorder.close()

        self.assertEqual(2, summary["accepted_sample_count"])
        self.assertEqual(2, summary["candidates"][0]["selected_count"])
        self.assertAlmostEqual(0.005, summary["candidates"][0]["error_translation_p95_m"])
        self.assertEqual(2, len(output_path.read_text(encoding="utf-8").splitlines()))
    finally:
      runtime.np = original_numpy


class TelemetryAnalyzerTests(unittest.TestCase):
  def test_compute_total_excludes_legacy_queue_wait(self):
    runtime_log = "\n".join((
        "[EVENT][foundationpose_init] schema=1 init=1 frame=10 camera_timestamp=1.0 "
        "yolo_start_unix_ns=100 yolo_end_unix_ns=200 detection_consumed_unix_ns=300 "
        "register_start_unix_ns=400 register_end_unix_ns=500 init_success_unix_ns=600",
        "[TIMER][SUMMARY][foundationpose_init] init=1 unit=ms",
        "yolo_total 12.000",
        "yolo_queue_wait 50.000",
        "foundation_total 148.000",
        "yolo_foundation_compute_total 999.000",
        "init_total 220.000",
    ))
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "runtime.log"
      path.write_text(runtime_log, encoding="utf-8")
      events = analyzer.parse_runtime(path)

    self.assertEqual(1, len(events))
    self.assertEqual(50.0, events[0]["detection_consume_delay"])
    self.assertEqual(160.0, events[0]["yolo_foundation_compute_total"])


class YoloDetectionWorkerTests(unittest.TestCase):
  def make_worker(self, detector):
    image_topic = runtime.LatestTopic("images")
    worker = runtime.YoloDetectionWorker(detector, image_topic, frame_stride=1)
    worker.start()
    self.addCleanup(worker.stop)
    return image_topic, worker

  def test_positive_candidate_blocks_next_inference_until_resolved(self):
    detector = FakeDetector([object(), None])
    image_topic, worker = self.make_worker(detector)

    image_topic.publish(frame(1))
    first = worker.wait_for_detection(0, timeout=1.0)
    self.assertIsNotNone(first)
    first_sequence, first_message = first
    self.assertIsNotNone(first_message.detection)

    image_topic.publish(frame(2))
    self.assertFalse(detector.wait_for_calls(2, timeout=0.05))

    worker.resume_candidate_search(first_sequence)
    self.assertTrue(detector.wait_for_calls(2, timeout=1.0))
    second = worker.wait_for_detection(first_sequence, timeout=1.0)
    self.assertIsNotNone(second)
    self.assertIsNone(second[1].detection)

  def test_early_resolution_is_not_lost(self):
    detector = FakeDetector([object(), None])
    image_topic, worker = self.make_worker(detector)

    image_topic.publish(frame(1))
    first_sequence, _message = worker.wait_for_detection(0, timeout=1.0)
    worker.resume_candidate_search(first_sequence)
    image_topic.publish(frame(2))

    self.assertTrue(detector.wait_for_calls(2, timeout=1.0))

  def test_none_detection_does_not_require_acknowledgment(self):
    detector = FakeDetector([None, None])
    image_topic, worker = self.make_worker(detector)

    image_topic.publish(frame(1))
    first_sequence, first_message = worker.wait_for_detection(0, timeout=1.0)
    self.assertIsNone(first_message.detection)
    image_topic.publish(frame(2))

    second = worker.wait_for_detection(first_sequence, timeout=1.0)
    self.assertIsNotNone(second)
    self.assertIsNone(second[1].detection)

  def test_finish_candidate_search_restores_free_running_tracking_mode(self):
    detector = FakeDetector([object(), object()])
    image_topic, worker = self.make_worker(detector)

    image_topic.publish(frame(1))
    first_sequence, _message = worker.wait_for_detection(0, timeout=1.0)
    worker.finish_candidate_search(first_sequence)
    image_topic.publish(frame(2))

    second = worker.wait_for_detection(first_sequence, timeout=1.0)
    self.assertIsNotNone(second)
    self.assertIsNotNone(second[1].detection)

  def test_pending_candidate_does_not_hold_inference_lock(self):
    detector = FakeDetector([object()])
    image_topic, worker = self.make_worker(detector)

    image_topic.publish(frame(1))
    first_sequence, _message = worker.wait_for_detection(0, timeout=1.0)
    started = time.perf_counter()
    worker.pause()
    elapsed = time.perf_counter() - started
    self.assertLess(elapsed, 0.25)
    worker.resume_candidate_search(first_sequence)
    worker.resume()

  def test_worker_failure_wakes_and_fails_waiter(self):
    detector = FakeDetector([ValueError("synthetic detector failure")])
    image_topic, worker = self.make_worker(detector)

    image_topic.publish(frame(1))
    with self.assertRaisesRegex(RuntimeError, "YOLO detection thread failed"):
      worker.wait_for_detection(0, timeout=1.0)

  def test_stop_releases_pending_candidate(self):
    detector = FakeDetector([object()])
    image_topic, worker = self.make_worker(detector)

    image_topic.publish(frame(1))
    worker.wait_for_detection(0, timeout=1.0)
    worker.stop()

    self.assertFalse(worker.thread.is_alive())


if __name__ == "__main__":
  unittest.main()
