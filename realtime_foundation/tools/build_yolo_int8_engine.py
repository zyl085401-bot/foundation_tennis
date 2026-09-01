# YOLO INT8 引擎构建
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_ultralytics_engine_metadata(path: Path) -> dict:
  with path.open('rb') as engine_file:
    metadata_length = int.from_bytes(engine_file.read(4), byteorder='little', signed=True)
    if metadata_length <= 0 or metadata_length > 16 * 1024 * 1024:
      raise RuntimeError(f'Invalid Ultralytics engine metadata length in {path}: {metadata_length}')
    metadata = json.loads(engine_file.read(metadata_length).decode('utf-8'))
  if not isinstance(metadata, dict):
    raise RuntimeError(f'Ultralytics engine metadata must be a dictionary: {path}')
  return metadata


def build_from_onnx_and_cache(
    onnx_path: Path,
    cache_path: Path,
    output_path: Path,
    metadata_template_path: Path,
    workspace_gib: float,
    imgsz: tuple[int, int],
    batch: int,
    data_path: Path,
    use_fp16_fallback: bool,
    fp16_layer_prefixes: tuple[str, ...],
    use_int8_input: bool = False,
    int8_input_dynamic_range: float = 1.0,
) -> Path:
  import tensorrt as trt

  class CacheCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, calibration_cache: Path):
      super().__init__()
      self.calibration_cache = calibration_cache

    def get_batch_size(self) -> int:
      return batch

    def get_batch(self, _names):
      return None

    def read_calibration_cache(self) -> bytes:
      return self.calibration_cache.read_bytes()

    def write_calibration_cache(self, cache: bytes) -> None:
      self.calibration_cache.write_bytes(cache)

  logger = trt.Logger(trt.Logger.INFO)
  builder = trt.Builder(logger)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, logger)
  if not parser.parse_from_file(str(onnx_path)):
    errors = '\n'.join(str(parser.get_error(index)) for index in range(parser.num_errors))
    raise RuntimeError(f'TensorRT failed to parse {onnx_path}:\n{errors}')

  input_tensor = network.get_input(0)
  if input_tensor is None:
    raise RuntimeError('Parsed YOLO network has no input tensor')
  if use_int8_input:
    if int8_input_dynamic_range <= 0:
      raise ValueError('INT8 input dynamic range must be positive')
    input_tensor.dtype = trt.int8
    input_tensor.dynamic_range = (-float(int8_input_dynamic_range), float(int8_input_dynamic_range))

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024**3))
  config.builder_optimization_level = 5
  config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
  config.set_flag(trt.BuilderFlag.INT8)
  if use_fp16_fallback:
    if not builder.platform_has_fast_fp16:
      raise RuntimeError('TensorRT reports no fast FP16 support for INT8 fallback layers')
    config.set_flag(trt.BuilderFlag.FP16)
  constrained_layers = []
  if fp16_layer_prefixes:
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    for layer_index in range(network.num_layers):
      layer = network.get_layer(layer_index)
      if not any(layer.name.startswith(prefix) for prefix in fp16_layer_prefixes):
        continue
      output_types = [
          layer.get_output(output_index).dtype
          for output_index in range(layer.num_outputs)
          if layer.get_output(output_index) is not None
      ]
      if not output_types or any(dtype not in (trt.float16, trt.float32) for dtype in output_types):
        continue
      layer_dtype = (
          trt.float32
          if layer.type in (trt.LayerType.ACTIVATION, trt.LayerType.ELEMENTWISE)
          else trt.float16
      )
      layer.precision = layer_dtype
      for output_index in range(layer.num_outputs):
        layer.set_output_type(output_index, layer_dtype)
      constrained_layers.append({'name': layer.name, 'precision': str(layer_dtype)})
    if not constrained_layers:
      raise RuntimeError(f'No TensorRT layers matched FP16 prefixes: {fp16_layer_prefixes}')
    print(json.dumps({'fp16_constrained_layers': constrained_layers}, indent=2))
  config.int8_calibrator = CacheCalibrator(cache_path)
  plan = builder.build_serialized_network(network, config)
  if plan is None:
    raise RuntimeError('TensorRT failed to build the cached-calibration INT8/FP16 engine')

  metadata = read_ultralytics_engine_metadata(metadata_template_path)
  metadata.update({
      'batch': batch,
      'imgsz': list(imgsz),
      'half': False,
      'int8': True,
      'dynamic': False,
      'data': str(data_path.resolve()),
      'int8_io': bool(use_int8_input),
      'input_dtype': 'int8' if use_int8_input else 'float32',
      'input_dynamic_range': float(int8_input_dynamic_range) if use_int8_input else None,
      'input_quantization_scale': (
          float(int8_input_dynamic_range) / 127.0 if use_int8_input else None
      ),
  })
  export_args = dict(metadata.get('args', {}) or {})
  export_args.update({
      'batch': batch,
      'imgsz': list(imgsz),
      'half': False,
      'int8': True,
      'dynamic': False,
      'data': str(data_path.resolve()),
  })
  metadata['args'] = export_args
  metadata['int8_fp16_constrained_layers'] = constrained_layers
  metadata['int8_io_schema_version'] = 1
  encoded_metadata = json.dumps(metadata).encode('utf-8')
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('wb') as output_file:
    output_file.write(len(encoded_metadata).to_bytes(4, byteorder='little', signed=True))
    output_file.write(encoded_metadata)
    output_file.write(plan)
  return output_path.resolve()


def prepare_calibration_dataset(
    frame_dir: Path,
    output_dir: Path,
    max_frames: int | None,
    class_name: str,
) -> Path:
  frame_paths = sorted(frame_dir.glob('*_register.npz'))
  if max_frames is not None:
    frame_paths = frame_paths[:max_frames]
  if not frame_paths:
    raise RuntimeError(f'No *_register.npz files found in {frame_dir}')

  image_dir = output_dir / 'images' / 'val'
  image_dir.mkdir(parents=True, exist_ok=True)
  records = []
  for frame_path in frame_paths:
    with np.load(frame_path, allow_pickle=False) as frame:
      if 'rgb' not in frame:
        raise KeyError(f'{frame_path} has no rgb array')
      rgb = np.asarray(frame['rgb'])
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
      raise ValueError(f'{frame_path} rgb must be HWC uint8, got shape={rgb.shape} dtype={rgb.dtype}')
    image_path = image_dir / f'{frame_path.stem}.jpg'
    if not cv2.imwrite(str(image_path), rgb[..., ::-1]):
      raise RuntimeError(f'Failed to write {image_path}')
    records.append({'source': str(frame_path.resolve()), 'image': str(image_path.resolve())})

  dataset_yaml = output_dir / 'dataset.yaml'
  dataset_yaml.write_text(
      yaml.safe_dump({
          'path': str(output_dir.resolve()),
          'train': 'images/val',
          'val': 'images/val',
          'names': {0: class_name},
      }, sort_keys=False),
      encoding='utf-8',
  )
  (output_dir / 'manifest.json').write_text(
      json.dumps({'frame_count': len(records), 'records': records}, indent=2) + '\n',
      encoding='utf-8',
  )
  return dataset_yaml


def main() -> None:
  parser = argparse.ArgumentParser(description='Build an Ultralytics YOLO INT8 TensorRT engine from recorded RGB frames.')
  parser.add_argument('--weights', type=Path, required=True)
  parser.add_argument('--frame-dir', type=Path, required=True)
  parser.add_argument('--calibration-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--imgsz', type=int, nargs=2, default=(480, 640), metavar=('HEIGHT', 'WIDTH'))
  parser.add_argument('--batch', type=int, default=1)
  parser.add_argument('--max-frames', type=int, default=500)
  parser.add_argument('--workspace-gib', type=float, default=4.0)
  parser.add_argument('--class-name', default='cup')
  parser.add_argument('--fp16-template', type=Path)
  parser.add_argument('--cache-only', action='store_true')
  parser.add_argument(
      '--int8-input',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='Build an INT8 input binding. Cache-only mode is required; outputs remain floating point.',
  )
  parser.add_argument(
      '--int8-input-dynamic-range',
      type=float,
      default=1.0,
      help='Symmetric real-value range for normalized RGB input; 1.0 maps q=127 to x=1.0.',
  )
  parser.add_argument('--fp16-fallback', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      '--fp16-layer-prefix',
      action='append',
      default=['/model.23/proto/cv3'],
      help='TensorRT layer-name prefix to keep in FP16; may be supplied multiple times.',
  )
  parser.add_argument(
      '--nms',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='Request an added NMS export layer; YOLO26 is natively end-to-end and defaults to false.',
  )
  args = parser.parse_args()

  if not args.weights.is_file():
    raise FileNotFoundError(args.weights)
  if args.max_frames < 1:
    raise ValueError('--max-frames must be positive')
  if args.int8_input and not args.cache_only:
    raise ValueError('--int8-input currently requires --cache-only')
  if args.int8_input_dynamic_range <= 0:
    raise ValueError('--int8-input-dynamic-range must be positive')
  dataset_yaml = prepare_calibration_dataset(
      frame_dir=args.frame_dir,
      output_dir=args.calibration_dir,
      max_frames=args.max_frames,
      class_name=args.class_name,
  )

  from ultralytics import YOLO

  model = YOLO(str(args.weights), task='segment')
  if args.cache_only:
    onnx_path = args.weights.with_suffix('.onnx')
    cache_path = args.weights.with_suffix('.cache')
    template_path = args.fp16_template
    if not onnx_path.is_file() or not cache_path.is_file():
      raise FileNotFoundError(f'--cache-only requires {onnx_path} and {cache_path}')
    if template_path is None or not template_path.is_file():
      raise FileNotFoundError('--cache-only requires an existing --fp16-template engine')
    exported_path = build_from_onnx_and_cache(
        onnx_path=onnx_path,
        cache_path=cache_path,
        output_path=args.output,
        metadata_template_path=template_path,
        workspace_gib=args.workspace_gib,
        imgsz=tuple(args.imgsz),
        batch=args.batch,
        data_path=dataset_yaml,
        use_fp16_fallback=args.fp16_fallback,
        fp16_layer_prefixes=tuple(args.fp16_layer_prefix),
        use_int8_input=args.int8_input,
        int8_input_dynamic_range=args.int8_input_dynamic_range,
    )
  else:
    exported_path = Path(model.export(
        format='engine',
        int8=True,
        half=False,
        data=str(dataset_yaml),
        imgsz=list(args.imgsz),
        batch=args.batch,
        dynamic=False,
        simplify=False,
        nms=args.nms,
        workspace=args.workspace_gib,
    )).resolve()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  if exported_path != args.output.resolve():
    shutil.copy2(exported_path, args.output)
  print(json.dumps({
      'engine': str(args.output.resolve()),
      'calibration_dataset': str(dataset_yaml.resolve()),
      'calibration_frames': len(list((args.calibration_dir / 'images' / 'val').glob('*.jpg'))),
      'export_nms': bool(args.nms),
      'int8_input': bool(args.int8_input),
      'input_dynamic_range': float(args.int8_input_dynamic_range) if args.int8_input else None,
  }, indent=2))


if __name__ == '__main__':
  main()
