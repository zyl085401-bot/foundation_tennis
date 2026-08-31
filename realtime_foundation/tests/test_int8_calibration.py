from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / 'FoundationPose' / 'tools' / 'int8_calibrator.py'
SPEC = importlib.util.spec_from_file_location('int8_calibrator_test_module', MODULE_PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError(f'Unable to load {MODULE_PATH}')
calibration_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration_module)
NetworkInputCaptureDataset = calibration_module.NetworkInputCaptureDataset


class NetworkInputCaptureDatasetTests(unittest.TestCase):
  def write_capture(self, directory: Path, name: str, network: str, shape: tuple[int, ...]) -> None:
    torch.save({
        'network': network,
        'stage': 'scorer_fine',
        'A': torch.ones(shape, dtype=torch.float32),
        'B': torch.full(shape, 2.0, dtype=torch.float32),
        'output': {},
    }, directory / name)

  def test_filters_network_and_shape_and_loads_float32(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      directory = Path(temporary_dir)
      self.write_capture(directory, 'valid.pt', 'scorer', (5, 6, 160, 160))
      self.write_capture(directory, 'wrong_network.pt', 'refiner', (5, 6, 160, 160))
      self.write_capture(directory, 'wrong_shape.pt', 'scorer', (6, 6, 160, 160))

      dataset = NetworkInputCaptureDataset(
          capture_dir=directory,
          input_shapes={'A': (5, 6, 160, 160), 'B': (5, 6, 160, 160)},
          expected_network='scorer',
      )

      self.assertEqual(1, len(dataset.records))
      self.assertEqual(2, len(dataset.rejected))
      tensors = dataset.load_tensors(0)
      self.assertEqual(torch.float32, tensors['A'].dtype)
      self.assertTrue(tensors['A'].is_contiguous())

  def test_rejects_empty_matching_dataset(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      directory = Path(temporary_dir)
      self.write_capture(directory, 'wrong.pt', 'refiner', (5, 6, 160, 160))
      with self.assertRaisesRegex(RuntimeError, 'No calibration captures'):
        NetworkInputCaptureDataset(
            capture_dir=directory,
            input_shapes={'A': (5, 6, 160, 160), 'B': (5, 6, 160, 160)},
            expected_network='scorer',
        )


if __name__ == '__main__':
  unittest.main()
