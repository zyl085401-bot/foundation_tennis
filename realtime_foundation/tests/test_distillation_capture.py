import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from realtime_foundation.tracking.distillation_capture import DistillationCaptureWriter


class DistillationCaptureWriterTest(unittest.TestCase):
  @staticmethod
  def _group(candidate_count: int = 5, iteration_count: int = 2) -> dict:
    candidate_ids = np.asarray([21, 8, 13, 5, 34], dtype=np.int64)[:candidate_count]
    logits = np.asarray([0.2, 0.9, -0.1, 0.5, 0.4], dtype=np.float32)[:candidate_count]
    order = logits.argsort()[::-1].astype(np.int64)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], candidate_count, axis=0)
    iterations = []
    poses_before = poses.copy()
    for iteration_index in range(iteration_count):
      poses_after = poses_before.copy()
      poses_after[:, 0, 3] += np.arange(candidate_count, dtype=np.float32) * 0.001
      iterations.append({
          "iteration": iteration_index + 1,
          "poses_before": poses_before,
          "poses_after": poses_after,
          "raw_trans": np.zeros((candidate_count, 3), dtype=np.float32),
          "raw_rot": np.zeros((candidate_count, 3), dtype=np.float32),
          "trans_applied": np.zeros((candidate_count, 3), dtype=np.float32),
          "rot_applied": np.repeat(np.eye(3, dtype=np.float32)[None], candidate_count, axis=0),
          "shared_feature": np.full((candidate_count, 512), iteration_index, dtype=np.float32),
          "network_backend": "pytorch",
          "input_size": np.asarray([160, 160], dtype=np.int32),
      })
      poses_before = poses_after
    return {
        "schema_version": 1,
        "feature_version": "refinenet_ab_mean_v1",
        "candidate_ids": candidate_ids,
        "candidate_count": candidate_count,
        "fine_iterations": iterations,
        "teacher_raw_logits": logits,
        "teacher_public_scores": logits + 100.0,
        "teacher_order_positions": order,
        "teacher_order_candidate_ids": candidate_ids[order],
        "teacher_top1_candidate_id": int(candidate_ids[order[0]]),
        "teacher_margin": float(logits[order[0]] - logits[order[1]]),
        "teacher_backend": "tensorrt",
        "final_poses": poses_before,
    }

  def test_writes_complete_candidate_group_and_metadata(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      writer = DistillationCaptureWriter(
          {
              "enabled": True,
              "output_dir": temporary_directory,
              "run_id": "unit_test",
              "expected_candidate_count": 5,
              "max_groups": 2,
          },
          {"refiner": {"checkpoint": {"sha256": "abc"}}},
      )
      record = writer.write_group(
          self._group(),
          frame_id=42,
          timestamp=123.5,
          object_id=7,
          sequence_id="sequence_a",
          source="unit_test",
      )

      self.assertIsNotNone(record)
      self.assertEqual(writer.saved_groups, 1)
      with np.load(record["npz"]) as arrays:
        self.assertEqual(arrays["candidate_ids"].tolist(), [21, 8, 13, 5, 34])
        self.assertEqual(arrays["fine_iter_02__shared_feature"].shape, (5, 512))
        self.assertEqual(int(arrays["teacher_order_candidate_ids"][0]), 8)
      with open(record["json"], encoding="utf-8") as input_file:
        group_metadata = json.load(input_file)
      self.assertEqual(group_metadata["frame_id"], 42)
      self.assertEqual(group_metadata["sequence_id"], "sequence_a")
      self.assertEqual(group_metadata["fine_iteration_count"], 2)
      with (writer.run_dir / "dataset_metadata.json").open(encoding="utf-8") as input_file:
        dataset_metadata = json.load(input_file)
      self.assertEqual(
          dataset_metadata["version_metadata"]["refiner"]["checkpoint"]["sha256"],
          "abc",
      )
      self.assertEqual(len((writer.run_dir / "manifest.jsonl").read_text().splitlines()), 1)

  def test_non_strict_writer_rejects_incomplete_teacher_logits(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      writer = DistillationCaptureWriter(
          {
              "enabled": True,
              "output_dir": temporary_directory,
              "run_id": "invalid_test",
              "expected_candidate_count": 5,
              "strict": False,
          },
          {},
      )
      group = self._group()
      group["teacher_raw_logits"][2] = np.nan
      record = writer.write_group(
          group,
          frame_id=1,
          timestamp=None,
          object_id=None,
      )
      self.assertIsNone(record)
      self.assertEqual(writer.saved_groups, 0)
      self.assertIn("finite numeric data", writer.last_error)


if __name__ == "__main__":
  unittest.main()
