from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from run_realtime import build_tracker, load_config


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Replay recorded register frames through one tracker instance to capture real A/B INT8 inputs.'
  )
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--frame-dir', type=Path, required=True)
  parser.add_argument('--max-frames', type=int, default=200)
  parser.add_argument('--stride', type=int, default=1)
  args = parser.parse_args()
  if args.max_frames < 1:
    parser.error('--max-frames must be positive')
  if args.stride < 1:
    parser.error('--stride must be positive')
  return args


def load_recorded_frame(path: Path) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    required = ('frame_id', 'timestamp', 'rgb', 'depth', 'K', 'yolo_mask')
    missing = [name for name in required if name not in data]
    if missing:
      raise KeyError(f'{path} is missing arrays: {missing}')
    frame_id = int(data['frame_id'].item())
    timestamp = float(data['timestamp'].item())
    rgb = np.ascontiguousarray(data['rgb'], dtype=np.uint8)
    depth = np.ascontiguousarray(data['depth'], dtype=np.float32)
    K = np.ascontiguousarray(data['K'], dtype=np.float32)
    mask = np.ascontiguousarray(data['yolo_mask'], dtype=np.uint8)
  if rgb.ndim != 3 or rgb.shape[2] != 3:
    raise ValueError(f'{path} has invalid RGB shape {rgb.shape}')
  if depth.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2] or K.shape != (3, 3):
    raise ValueError(
        f'{path} shape mismatch: rgb={rgb.shape}, depth={depth.shape}, mask={mask.shape}, K={K.shape}'
    )
  return frame_id, timestamp, rgb, depth, K, mask


def main() -> int:
  args = parse_args()
  config_path = args.config.expanduser().resolve()
  frame_dir = args.frame_dir.expanduser().resolve()
  config = load_config(str(config_path))
  foundation_config = dict(config.get('foundationpose', {}) or {})
  capture_config = dict(foundation_config.get('network_input_capture', {}) or {})
  if not bool(capture_config.get('enabled', False)):
    raise RuntimeError('foundationpose.network_input_capture.enabled must be true in the capture config')

  frame_paths = sorted(frame_dir.glob('*_register.npz'))[::args.stride][:args.max_frames]
  if not frame_paths:
    raise RuntimeError(f'No *_register.npz files found in {frame_dir}')

  tracker = build_tracker(foundation_config, candidate_pipeline_debug_enabled=False)
  success_count = 0
  failures: list[str] = []
  started = time.perf_counter()
  for index, frame_path in enumerate(frame_paths, start=1):
    try:
      frame_id, timestamp, rgb, depth, K, mask = load_recorded_frame(frame_path)
      tracker.reset()
      tracker.register(
          rgb,
          depth,
          K,
          mask,
          frame_id=frame_id,
          timestamp=timestamp,
          sequence_id=frame_path.name,
          capture_source='int8_calibration_replay',
      )
      success_count += 1
    except Exception as error:
      failures.append(f'{frame_path.name}: {type(error).__name__}: {error}')
    if index % 10 == 0 or index == len(frame_paths):
      print(f'[INT8 CAPTURE] progress={index}/{len(frame_paths)} success={success_count} failed={len(failures)}')

  elapsed = time.perf_counter() - started
  print(
      f'[INT8 CAPTURE] complete frames={len(frame_paths)} success={success_count} '
      f'failed={len(failures)} elapsed={elapsed:.3f}s'
  )
  for failure in failures[:20]:
    print(f'[INT8 CAPTURE][FAILED] {failure}')
  if success_count == 0:
    raise RuntimeError('All recorded-frame registrations failed; no calibration inputs were captured')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
