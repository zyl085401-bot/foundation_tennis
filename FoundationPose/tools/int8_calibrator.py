from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

try:
  import tensorrt as trt
except ImportError:  # Allows dataset auditing on hosts without TensorRT.
  trt = None


_CALIBRATOR_BASE = trt.IInt8EntropyCalibrator2 if trt is not None else object


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


class NetworkInputCaptureDataset:
  """Validated real A/B tensors captured immediately before network inference."""

  def __init__(
      self,
      capture_dir: Path,
      input_shapes: dict[str, tuple[int, ...]],
      expected_network: str | None = None,
      max_samples: int | None = None,
  ):
    self.capture_dir = capture_dir.expanduser().resolve()
    if not self.capture_dir.is_dir():
      raise FileNotFoundError(self.capture_dir)
    self.input_shapes = {str(name): tuple(int(value) for value in shape) for name, shape in input_shapes.items()}
    self.expected_network = expected_network
    self.max_samples = None if max_samples is None else int(max_samples)
    if self.max_samples is not None and self.max_samples < 1:
      raise ValueError('max_samples must be positive')

    self.records: list[dict[str, Any]] = []
    self.rejected: list[dict[str, str]] = []
    for path in sorted(self.capture_dir.rglob('*.pt')):
      try:
        sample = self._load(path)
        self._validate_sample(path, sample)
      except Exception as error:
        self.rejected.append({'path': str(path), 'reason': f'{type(error).__name__}: {error}'})
        continue
      self.records.append({
          'path': path,
          'sha256': file_sha256(path),
          'network': sample.get('network'),
          'stage': sample.get('stage'),
      })
      if self.max_samples is not None and len(self.records) >= self.max_samples:
        break
    if not self.records:
      rejected_summary = '; '.join(f"{item['path']}: {item['reason']}" for item in self.rejected[:3])
      raise RuntimeError(
          f'No calibration captures in {self.capture_dir} match input shapes {self.input_shapes}'
          + (f'; examples: {rejected_summary}' if rejected_summary else '')
      )

  @staticmethod
  def _load(path: Path) -> dict[str, Any]:
    sample = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(sample, dict):
      raise TypeError('capture must contain a dictionary')
    return sample

  def _validate_sample(self, path: Path, sample: dict[str, Any]) -> None:
    if self.expected_network is not None and sample.get('network') != self.expected_network:
      raise ValueError(f"network={sample.get('network')!r}, expected {self.expected_network!r}")
    for name, expected_shape in self.input_shapes.items():
      tensor = sample.get(name)
      if not torch.is_tensor(tensor):
        raise TypeError(f'{name} is not a tensor')
      if tuple(tensor.shape) != expected_shape:
        raise ValueError(f'{name} shape={tuple(tensor.shape)}, expected {expected_shape}')
      if not tensor.is_floating_point():
        raise TypeError(f'{name} dtype={tensor.dtype} is not floating point')
      if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} contains non-finite values')

  def load_tensors(self, index: int) -> dict[str, torch.Tensor]:
    sample = self._load(self.records[index]['path'])
    return {
        name: sample[name].to(device='cpu', dtype=torch.float32).contiguous()
        for name in self.input_shapes
    }

  def metadata(self) -> dict[str, Any]:
    return {
        'capture_dir': str(self.capture_dir),
        'expected_network': self.expected_network,
        'input_shapes': {name: list(shape) for name, shape in self.input_shapes.items()},
        'sample_count': len(self.records),
        'samples': [
            {
                'path': str(record['path'].relative_to(self.capture_dir)),
                'sha256': record['sha256'],
                'network': record['network'],
                'stage': record['stage'],
            }
            for record in self.records
        ],
        'rejected_count': len(self.rejected),
        'rejected_examples': self.rejected[:10],
    }


class NetworkInputEntropyCalibrator(_CALIBRATOR_BASE):
  """TensorRT entropy calibrator using captured float32 network inputs."""

  def __init__(
      self,
      input_shapes: dict[str, tuple[int, ...]],
      cache_path: Path,
      capture_dir: Path | None = None,
      expected_network: str | None = None,
      max_samples: int | None = None,
  ):
    if trt is None:
      raise ImportError('TensorRT is required to create an INT8 calibrator')
    super().__init__()
    self.input_shapes = {str(name): tuple(int(value) for value in shape) for name, shape in input_shapes.items()}
    batch_sizes = {shape[0] for shape in self.input_shapes.values()}
    if len(batch_sizes) != 1:
      raise ValueError(f'All calibration inputs must share one leading batch dimension, got {self.input_shapes}')
    self.batch_size = next(iter(batch_sizes))
    self.cache_path = cache_path.expanduser().resolve()
    self.dataset = (
        NetworkInputCaptureDataset(
            capture_dir=capture_dir,
            input_shapes=self.input_shapes,
            expected_network=expected_network,
            max_samples=max_samples,
        )
        if capture_dir is not None else None
    )
    if self.dataset is None and not self.cache_path.is_file():
      raise FileNotFoundError('INT8 calibration requires capture data or an existing calibration cache')
    self._sample_index = 0
    self._device_buffers = {
        name: torch.empty(shape, device='cuda', dtype=torch.float32)
        for name, shape in self.input_shapes.items()
    }

  def get_batch_size(self) -> int:
    return self.batch_size

  def get_batch(self, names: list[str]) -> list[int] | None:
    if self.dataset is None or self._sample_index >= len(self.dataset.records):
      return None
    tensors = self.dataset.load_tensors(self._sample_index)
    self._sample_index += 1
    pointers = []
    for name in names:
      if name not in self._device_buffers:
        raise KeyError(f'Unexpected TensorRT calibration input {name!r}')
      self._device_buffers[name].copy_(tensors[name], non_blocking=False)
      pointers.append(int(self._device_buffers[name].data_ptr()))
    return pointers

  def read_calibration_cache(self) -> bytes | None:
    return self.cache_path.read_bytes() if self.cache_path.is_file() else None

  def write_calibration_cache(self, cache: bytes) -> None:
    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
    self.cache_path.write_bytes(cache)

  def metadata(self) -> dict[str, Any]:
    result = {
        'algorithm': 'IInt8EntropyCalibrator2',
        'cache_path': str(self.cache_path),
        'cache_sha256': file_sha256(self.cache_path) if self.cache_path.is_file() else None,
        'batch_size': self.batch_size,
    }
    result['dataset'] = self.dataset.metadata() if self.dataset is not None else None
    return result


def write_calibration_manifest(path: Path, metadata: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
