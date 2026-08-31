#!/usr/bin/env python3
"""Compare two FoundationPose TensorRT engines on captured real network inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from FoundationPose.learning.training.tensorrt_runner import TensorRTEngineRunner


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--capture-dir', type=Path, required=True)
  parser.add_argument('--network', choices=('refiner', 'scorer'), required=True)
  parser.add_argument('--stage', required=True, help='Exact captured stage name.')
  parser.add_argument('--candidate-count', type=int, required=True)
  parser.add_argument('--reference-engine', type=Path, required=True)
  parser.add_argument('--candidate-engine', type=Path, required=True)
  parser.add_argument('--max-samples', type=int)
  parser.add_argument('--output', type=Path)
  return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
  return ordered[index]


def load_captures(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
  captures = []
  for path in sorted(args.capture_dir.glob('*.pt')):
    sample = torch.load(path, map_location='cpu', weights_only=False)
    if sample.get('network') != args.network or sample.get('stage') != args.stage:
      continue
    if int(sample['A'].shape[0]) != args.candidate_count:
      continue
    captures.append((path, sample))
    if args.max_samples is not None and len(captures) >= args.max_samples:
      break
  if not captures:
    raise RuntimeError(
        f'No captures matched network={args.network!r}, stage={args.stage!r}, '
        f'candidate_count={args.candidate_count} in {args.capture_dir}'
    )
  return captures


def run_engine(
    runner: TensorRTEngineRunner,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], float]:
  outputs = runner(inputs)
  torch.cuda.synchronize()
  elapsed = runner.collect_last_cuda_timing()
  if elapsed is None:
    raise RuntimeError('TensorRT runner did not return CUDA timing events')
  return {name: tensor.detach().cpu().clone() for name, tensor in outputs.items()}, elapsed * 1000.0


def main() -> None:
  args = parse_args()
  captures = load_captures(args)
  reference = TensorRTEngineRunner(str(args.reference_engine), expected_network=args.network)
  candidate = TensorRTEngineRunner(str(args.candidate_engine), expected_network=args.network)

  per_output: dict[str, dict[str, list[float]]] = {}
  reference_ms: list[float] = []
  candidate_ms: list[float] = []
  scorer_argmax_equal = 0
  scorer_order_equal = 0

  for _, sample in captures:
    inputs = {
        'A': sample['A'].to(device='cuda', dtype=torch.float32).contiguous(),
        'B': sample['B'].to(device='cuda', dtype=torch.float32).contiguous(),
    }
    reference_outputs, reference_elapsed = run_engine(reference, inputs)
    candidate_outputs, candidate_elapsed = run_engine(candidate, inputs)
    reference_ms.append(reference_elapsed)
    candidate_ms.append(candidate_elapsed)
    if reference_outputs.keys() != candidate_outputs.keys():
      raise RuntimeError(
          f'Output mismatch: reference={sorted(reference_outputs)}, candidate={sorted(candidate_outputs)}'
      )

    for name in reference_outputs:
      lhs = reference_outputs[name].float()
      rhs = candidate_outputs[name].float()
      if lhs.shape != rhs.shape:
        raise RuntimeError(f'{name} shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}')
      absolute = (lhs - rhs).abs()
      relative = absolute / lhs.abs().clamp_min(1e-6)
      metrics = per_output.setdefault(name, {'max_abs': [], 'mean_abs': [], 'p99_abs': [], 'max_rel': []})
      metrics['max_abs'].append(float(absolute.max()))
      metrics['mean_abs'].append(float(absolute.mean()))
      metrics['p99_abs'].append(float(torch.quantile(absolute.flatten(), 0.99)))
      metrics['max_rel'].append(float(relative.max()))

    if args.network == 'scorer':
      lhs = reference_outputs['score_logit'].flatten()
      rhs = candidate_outputs['score_logit'].flatten()
      scorer_argmax_equal += int(int(lhs.argmax()) == int(rhs.argmax()))
      scorer_order_equal += int(torch.equal(torch.argsort(lhs, descending=True), torch.argsort(rhs, descending=True)))

  output_summary = {}
  for name, metrics in per_output.items():
    output_summary[name] = {
        key: {
            'max': max(values),
            'mean': statistics.fmean(values),
            'median': statistics.median(values),
            'p95': percentile(values, 0.95),
        }
        for key, values in metrics.items()
    }

  summary: dict[str, Any] = {
      'network': args.network,
      'stage': args.stage,
      'candidate_count': args.candidate_count,
      'sample_count': len(captures),
      'reference_engine': str(args.reference_engine),
      'reference_sha256': reference.metadata.get('engine_sha256'),
      'candidate_engine': str(args.candidate_engine),
      'candidate_sha256': candidate.metadata.get('engine_sha256'),
      'outputs': output_summary,
      'latency_ms': {
          'reference': {
              'median': statistics.median(reference_ms),
              'mean': statistics.fmean(reference_ms),
              'p95': percentile(reference_ms, 0.95),
          },
          'candidate': {
              'median': statistics.median(candidate_ms),
              'mean': statistics.fmean(candidate_ms),
              'p95': percentile(candidate_ms, 0.95),
          },
      },
  }
  if args.network == 'scorer':
    summary['ranking'] = {
        'argmax_equal_count': scorer_argmax_equal,
        'argmax_equal_fraction': scorer_argmax_equal / len(captures),
        'full_order_equal_count': scorer_order_equal,
        'full_order_equal_fraction': scorer_order_equal / len(captures),
    }

  serialized = json.dumps(summary, indent=2, sort_keys=True) + '\n'
  print(serialized, end='')
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding='utf-8')


if __name__ == '__main__':
  main()
