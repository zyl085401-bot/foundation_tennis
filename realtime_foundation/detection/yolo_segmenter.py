from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_segment_config.yml")


def read_ultralytics_engine_metadata(path: str) -> dict[str, Any]:
  engine_path = Path(path).expanduser()
  with engine_path.open("rb") as engine_file:
    metadata_length = int.from_bytes(engine_file.read(4), byteorder="little", signed=True)
    if metadata_length <= 0 or metadata_length > 16 * 1024 * 1024:
      raise RuntimeError(f"invalid Ultralytics engine metadata length in {engine_path}: {metadata_length}")
    metadata = json.loads(engine_file.read(metadata_length).decode("utf-8"))
  if not isinstance(metadata, dict):
    raise RuntimeError(f"Ultralytics engine metadata must be a dictionary: {engine_path}")
  return metadata


class _DirectTensorRTEngineBackend:
  """Minimal fixed-shape TensorRT backend for explicit INT8 input bindings."""

  def __init__(self, engine_path: str):
    import tensorrt as trt
    import torch

    path = Path(engine_path).expanduser().resolve()
    payload = path.read_bytes()
    if len(payload) < 5:
      raise RuntimeError(f"truncated Ultralytics TensorRT engine: {path}")
    metadata_length = int.from_bytes(payload[:4], byteorder="little", signed=True)
    plan_offset = 4 + metadata_length
    if metadata_length <= 0 or plan_offset >= len(payload):
      raise RuntimeError(f"invalid Ultralytics TensorRT engine wrapper: {path}")
    self.metadata = json.loads(payload[4:plan_offset].decode("utf-8"))
    self._logger = trt.Logger(trt.Logger.WARNING)
    self._runtime = trt.Runtime(self._logger)
    self._engine = self._runtime.deserialize_cuda_engine(payload[plan_offset:])
    if self._engine is None:
      raise RuntimeError(f"failed to deserialize TensorRT engine: {path}")
    self._context = self._engine.create_execution_context()
    if self._context is None:
      raise RuntimeError(f"failed to create TensorRT execution context: {path}")

    self.format = "engine"
    self.dynamic = False
    self.end2end = bool(self.metadata.get("end2end", True))
    self.fp16 = False
    self.device = torch.device("cuda:0")
    self.names = self.metadata.get("names", {})
    self.bindings = {}
    self._input_names = []
    self._output_names = []
    self._output_buffers = {}
    self._torch_dtypes = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    for index in range(self._engine.num_io_tensors):
      name = self._engine.get_tensor_name(index)
      trt_dtype = self._engine.get_tensor_dtype(name)
      if trt_dtype not in self._torch_dtypes:
        raise TypeError(f"unsupported TensorRT binding dtype for {name}: {trt_dtype}")
      shape = tuple(int(value) for value in self._engine.get_tensor_shape(name))
      if any(value < 0 for value in shape):
        raise RuntimeError(f"dynamic TensorRT binding is unsupported for {name}: {shape}")
      self.bindings[name] = SimpleNamespace(shape=shape, dtype=np.dtype(trt.nptype(trt_dtype)))
      mode = self._engine.get_tensor_mode(name)
      if mode == trt.TensorIOMode.INPUT:
        self._input_names.append(name)
      elif mode == trt.TensorIOMode.OUTPUT:
        self._output_names.append(name)
    if self._input_names != ["images"]:
      raise RuntimeError(f"expected one YOLO input named 'images', got {self._input_names}")

  def __call__(self, images):
    import torch

    binding = self.bindings["images"]
    expected_dtype = torch.int8 if np.dtype(binding.dtype) == np.dtype(np.int8) else torch.float32
    if tuple(images.shape) != tuple(binding.shape) or images.dtype != expected_dtype:
      raise RuntimeError(
          f"YOLO TensorRT input expects {tuple(binding.shape)}/{expected_dtype}, "
          f"got {tuple(images.shape)}/{images.dtype}"
      )
    if not images.is_cuda or not images.is_contiguous():
      raise RuntimeError("YOLO TensorRT input must be a contiguous CUDA tensor")
    self._context.set_tensor_address("images", images.data_ptr())
    outputs = {}
    for name in self._output_names:
      shape = tuple(self.bindings[name].shape)
      trt_dtype = self._engine.get_tensor_dtype(name)
      tensor = self._output_buffers.get(name)
      if tensor is None or tuple(tensor.shape) != shape:
        tensor = torch.empty(shape, dtype=self._torch_dtypes[trt_dtype], device=images.device)
        self._output_buffers[name] = tensor
      self._context.set_tensor_address(name, tensor.data_ptr())
      outputs[name] = tensor
    stream = torch.cuda.current_stream(device=images.device)
    if not self._context.execute_async_v3(stream_handle=stream.cuda_stream):
      raise RuntimeError("YOLO TensorRT execution failed")
    missing = {"output0", "output1"} - set(outputs)
    if missing:
      raise RuntimeError(f"YOLO TensorRT outputs are missing: {sorted(missing)}")
    return [outputs["output0"], outputs["output1"]]


@dataclass
class YoloMaskResult:
  mask: np.ndarray
  class_id: int
  class_name: str
  confidence: float
  box_xyxy: np.ndarray
  area: int


class YoloSegmenter:
  def __init__(
      self,
      weights: str,
      target_class: str | None = None,
      target_class_id: int | None = None,
      conf: float = 0.35,
      imgsz: int | tuple[int, int] = 640,
      device: str | None = None,
      half: bool = True,
      min_mask_area: int = 100,
      morph_kernel: int = 5,
      gpu_best_mask_topk: int = 1,
      input_is_rgb: bool = True,
      execution_path: str = "legacy",
      profile_stages: bool = False,
      fallback_to_legacy: bool = True,
      postprocess_backend: str = "gpu",
      fallback_weights: list[str] | tuple[str, ...] | None = None,
        int8_input_dynamic_range: float = 1.0,
  ):
    try:
      from ultralytics import YOLO
    except ImportError as exc:
      raise ImportError(
          "ultralytics is required for YOLO segmentation. Install it in your environment with: pip install ultralytics"
      ) from exc

    self._yolo_class = YOLO
    weight_candidates = list(dict.fromkeys([str(weights), *(str(path) for path in (fallback_weights or ()) if path)]))
    self.selected_weights = weight_candidates[0]
    self._remaining_weights = weight_candidates[1:]
    self.weight_fallback_failures: list[str] = []
    self.postprocess_backend = str(postprocess_backend).lower()
    self.int8_input_dynamic_range = float(int8_input_dynamic_range)
    if self.int8_input_dynamic_range != 1.0:
      raise ValueError("the integer-only YOLO preprocessing path currently requires int8_input_dynamic_range=1.0")
    if self.postprocess_backend not in ("gpu", "cpu"):
      raise ValueError("postprocess_backend must be 'gpu' or 'cpu'")
    self.requested_execution_path = str(execution_path).lower()
    if self.requested_execution_path not in ("legacy", "fast"):
      raise ValueError("execution_path must be 'legacy' or 'fast'")
    if self.postprocess_backend == "cpu" and self.requested_execution_path != "fast":
      raise ValueError("postprocess_backend='cpu' requires execution_path='fast'")
    try:
      self.model = self._create_model(self.selected_weights)
    except Exception as error:
      self.weight_fallback_failures.append(
          f"{self.selected_weights}: {type(error).__name__}: {error}"
      )
      self.model = self._create_next_available_model()
    self.target_class = target_class
    self.target_class_id = target_class_id
    self.conf = conf
    self.imgsz = normalize_imgsz(imgsz)
    self.device = device
    self.half = half
    self.min_mask_area = min_mask_area
    self.morph_kernel = morph_kernel
    self.gpu_best_mask_topk = max(1, int(gpu_best_mask_topk))
    self.input_is_rgb = input_is_rgb
    self.active_execution_path = "legacy"
    self.profile_stages = bool(profile_stages)
    self.fallback_to_legacy = bool(fallback_to_legacy)
    self.fast_path_fallback_reason: str | None = None
    self.fast_path_contract: dict[str, Any] = {}
    self.last_execution_path = "legacy"
    self.last_candidate_count = 0
    self.mask_cleanup_fast_count = 0
    self.mask_cleanup_legacy_count = 0
    self.names = self._normalise_names(getattr(self.model, "names", {}))
    self.last_timing = {}

    if self.target_class_id is None and self.target_class is not None:
      self.target_class_id = self._class_name_to_id(self.target_class)

    self.morphology_kernel = (
      np.ones((self.morph_kernel, self.morph_kernel), dtype=np.uint8)
      if self.morph_kernel > 1
      else None
    )

    self._fast_predictor = None
    self._fast_backend = None
    self._fast_host_input = None
    self._fast_host_array = None
    self._fast_device_input = None
    self._fast_device_uint8 = None
    self._fast_device_int16 = None
    self._fast_input_shape: tuple[int, int, int, int] | None = None
    self._fast_max_det = 300
    if self.requested_execution_path == "fast":
      try:
        self._initialize_fast_path()
      except Exception as exc:
        if self._switch_to_next_weights(
            f"fast-path initialization failed: {type(exc).__name__}: {exc}"
        ):
          return
        if not self.fallback_to_legacy or self.postprocess_backend == "cpu":
          raise
        self._fallback_to_legacy(f"initialization failed: {type(exc).__name__}: {exc}")

  def _create_model(self, weights: str):
    return self._yolo_class(weights, task="segment")

  def _create_next_available_model(self):
    while self._remaining_weights:
      candidate = self._remaining_weights.pop(0)
      try:
        model = self._create_model(candidate)
      except Exception as error:
        self.weight_fallback_failures.append(
            f"{candidate}: {type(error).__name__}: {error}"
        )
        continue
      self.selected_weights = candidate
      return model
    details = "; ".join(self.weight_fallback_failures)
    raise RuntimeError(f"No usable YOLO weights remain; {details}")

  def _switch_to_next_weights(self, reason: str) -> bool:
    self.weight_fallback_failures.append(f"{self.selected_weights}: {reason}")
    while self._remaining_weights:
      try:
        self.model = self._create_next_available_model()
        self.names = self._normalise_names(getattr(self.model, "names", {}))
        if self.target_class_id is None and self.target_class is not None:
          self.target_class_id = self._class_name_to_id(self.target_class)
        self._fast_predictor = None
        self._fast_backend = None
        self._fast_host_input = None
        self._fast_host_array = None
        self._fast_device_input = None
        self._fast_device_uint8 = None
        self._fast_device_int16 = None
        self.active_execution_path = "legacy"
        if self.requested_execution_path == "fast":
          self._initialize_fast_path()
        return True
      except Exception as error:
        self.weight_fallback_failures.append(
            f"{self.selected_weights}: {type(error).__name__}: {error}"
        )
    return False
  def predict_mask(self, image: np.ndarray) -> YoloMaskResult | None:
    if self.active_execution_path == "fast":
      import torch

      fast_started = time.perf_counter()
      try:
        with torch.inference_mode():
          return self._predict_mask_fast(image)
      except Exception as exc:
        if self._switch_to_next_weights(
            f"fast-path runtime failed: {type(exc).__name__}: {exc}"
        ):
          return self.predict_mask(image)
        if not self.fallback_to_legacy or self.postprocess_backend == "cpu":
          raise
        self._fallback_to_legacy(f"runtime validation failed: {type(exc).__name__}: {exc}")
        result = self._predict_mask_legacy(image)
        self.last_timing["total"] = time.perf_counter() - fast_started
        return result
    return self._predict_mask_legacy(image)

  @staticmethod
  def _new_timing(profile_stages: bool = True) -> dict[str, float | None]:
    stage_default = 0.0 if profile_stages else None
    return {
        "model_predict": 0.0,
        "model_preprocess": stage_default,
        "model_inference": stage_default,
        "model_postprocess": stage_default,
        "model_framework_overhead": stage_default,
        "tensor_to_cpu": stage_default,
        "mask_resize_clean": stage_default,
        "select_best_mask": stage_default,
        "total": 0.0,
    }

  def _predict_mask_legacy(self, image: np.ndarray) -> YoloMaskResult | None:
    timing = self._new_timing(profile_stages=True)
    total_start = time.perf_counter()
    self.last_execution_path = "legacy"
    self.last_candidate_count = 0
    if image is None or image.size == 0:
      self.last_timing = timing
      return None

    height, width = image.shape[:2]
    source = image[..., ::-1] if self.input_is_rgb else image
    model_start = time.perf_counter()
    prediction = self.model.predict(
        source=source,
        imgsz=self.imgsz,
        conf=self.conf,
        device=self.device,
        half=self.half,
        verbose=False,
    )[0]
    timing["model_predict"] = time.perf_counter() - model_start
    prediction_speed = getattr(prediction, "speed", {}) or {}
    model_stage_times = []
    for stage in ("preprocess", "inference", "postprocess"):
      stage_ms = prediction_speed.get(stage)
      if stage_ms is not None:
        stage_seconds = float(stage_ms) / 1000.0
        timing[f"model_{stage}"] = stage_seconds
        model_stage_times.append(stage_seconds)
    if len(model_stage_times) == 3:
      timing["model_framework_overhead"] = max(
          0.0,
          timing["model_predict"] - sum(model_stage_times),
      )

    if prediction.masks is None or prediction.boxes is None:
      timing["total"] = time.perf_counter() - total_start
      self.last_timing = timing
      return None

    tensor_start = time.perf_counter()
    masks = prediction.masks.data.detach().cpu().numpy()
    class_ids = prediction.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = prediction.boxes.conf.detach().cpu().numpy()
    boxes = prediction.boxes.xyxy.detach().cpu().numpy()
    timing["tensor_to_cpu"] = time.perf_counter() - tensor_start
    self.last_candidate_count = int(len(masks))

    best_result = self._select_best_result(
        masks,
        class_ids,
        confidences,
        boxes,
        width,
        height,
        timing,
    )
    timing["total"] = time.perf_counter() - total_start
    self.last_timing = timing
    return best_result

  def _initialize_fast_path(self) -> None:
    import torch

    height, width = self._normalized_input_size()
    engine_metadata = read_ultralytics_engine_metadata(self.selected_weights)
    if bool(engine_metadata.get("int8_io", False)):
      backend = _DirectTensorRTEngineBackend(self.selected_weights)
      predictor = SimpleNamespace(args=SimpleNamespace(max_det=300))
    else:
      dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
      dummy_source = dummy_rgb[..., ::-1] if self.input_is_rgb else dummy_rgb
      self.model.predict(
          source=dummy_source,
          imgsz=self.imgsz,
          conf=self.conf,
          device=self.device,
          half=self.half,
          verbose=False,
      )
      predictor = getattr(self.model, "predictor", None)
      backend = getattr(predictor, "model", None)
    if predictor is None or backend is None:
      raise RuntimeError("Ultralytics predictor backend was not initialized")
    if str(getattr(backend, "format", "")) != "engine":
      raise RuntimeError(f"expected TensorRT engine, got format={getattr(backend, 'format', None)!r}")
    if bool(getattr(backend, "dynamic", False)):
      raise RuntimeError("dynamic TensorRT engines are not supported by the fixed-shape path")
    if not bool(getattr(backend, "end2end", False)):
      raise RuntimeError("expected an end-to-end TensorRT segmentation engine")
    device = getattr(backend, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
      raise RuntimeError(f"expected CUDA TensorRT backend, got device={device!r}")

    bindings = dict(getattr(backend, "bindings", {}))
    expected_shapes = {
        "images": (1, 3, height, width),
        "output0": (1, 300, 38),
        "output1": (1, 32, height // 4, width // 4),
    }
    binding_dtypes = {}
    for name, expected_shape in expected_shapes.items():
      binding = bindings.get(name)
      if binding is None:
        raise RuntimeError(f"missing TensorRT binding {name!r}")
      actual_shape = tuple(int(value) for value in binding.shape)
      if actual_shape != expected_shape:
        raise RuntimeError(f"binding {name!r} shape {actual_shape} != {expected_shape}")
      actual_dtype = np.dtype(binding.dtype)
      allowed_dtypes = (
          (np.dtype(np.int8), np.dtype(np.float32))
          if name == "images"
          else (np.dtype(np.float16), np.dtype(np.float32))
      )
      if actual_dtype not in allowed_dtypes:
        raise RuntimeError(f"binding {name!r} dtype {binding.dtype} is unsupported")
      binding_dtypes[name] = actual_dtype

    input_dtype = binding_dtypes["images"]
    backend_fp16 = bool(getattr(backend, "fp16", False))
    if input_dtype not in (np.dtype(np.int8), np.dtype(np.float32)) or backend_fp16:
      raise RuntimeError(
        f"expected current GPU engine to use INT8 or FP32 input, got dtype={input_dtype} "
          f"backend.fp16={backend_fp16}"
      )

    backend_names = self._normalise_names(getattr(backend, "names", {}))
    if backend_names:
      self.names = backend_names
    if self.target_class_id is None and self.target_class is not None:
      self.target_class_id = self._class_name_to_id(self.target_class)

    input_shape = expected_shapes["images"]
    input_torch_dtype = torch.int8 if input_dtype == np.dtype(np.int8) else torch.float32
    host_input = torch.empty(input_shape, dtype=torch.uint8, pin_memory=True)
    self._fast_predictor = predictor
    self._fast_backend = backend
    self._fast_host_input = host_input
    self._fast_host_array = host_input.numpy()
    self._fast_device_input = torch.empty(input_shape, dtype=input_torch_dtype, device=device)
    self._fast_device_uint8 = (
      torch.empty(input_shape, dtype=torch.uint8, device=device)
      if input_dtype == np.dtype(np.int8) else None
    )
    self._fast_device_int16 = (
      torch.empty(input_shape, dtype=torch.int16, device=device)
      if input_dtype == np.dtype(np.int8) else None
    )
    self._fast_input_shape = input_shape
    self._fast_max_det = int(getattr(predictor.args, "max_det", 300))
    self.fast_path_contract = {
        "format": "engine",
        "device": str(device),
        "dynamic": False,
        "end2end": True,
        "input_shape": list(input_shape),
        "input_dtype": str(input_dtype),
        "input_dynamic_range": self.int8_input_dynamic_range if input_dtype == np.dtype(np.int8) else None,
        "integer_only_preprocess": input_dtype == np.dtype(np.int8),
        "host_input_dtype": str(host_input.numpy().dtype),
        "detection_output_shape": list(expected_shapes["output0"]),
        "detection_output_dtype": str(binding_dtypes["output0"]),
        "prototype_output_shape": list(expected_shapes["output1"]),
        "prototype_output_dtype": str(binding_dtypes["output1"]),
        "postprocess_backend": self.postprocess_backend,
        "max_det": self._fast_max_det,
    }
    self.active_execution_path = "fast"
    print(
        f"[YOLO][FAST] enabled: weights={self.selected_weights} "
        f"postprocess={self.postprocess_backend} input={input_shape}/{input_dtype} "
        f"outputs={expected_shapes['output0']},{expected_shapes['output1']}"
    )

  def _predict_mask_fast(self, image: np.ndarray) -> YoloMaskResult | None:
    import torch
    from ultralytics.utils import ops

    timing = self._new_timing(profile_stages=self.profile_stages)
    total_start = time.perf_counter()
    self.last_execution_path = "fast"
    self.last_candidate_count = 0
    if image is None or image.size == 0:
      timing["total"] = time.perf_counter() - total_start
      self.last_timing = timing
      return None
    height, width = self._validate_fast_input(image)
    if self._fast_backend is None or self._fast_device_input is None or self._fast_host_array is None:
      raise RuntimeError("fast executor buffers are unavailable")

    model_start = time.perf_counter()
    preprocess_start = self._start_profiled_stage(torch)
    rgb = image if self.input_is_rgb else image[..., ::-1]
    chw_rgb = np.moveaxis(rgb, 2, 0)
    np.copyto(self._fast_host_array[0], chw_rgb, casting="no")
    if self._fast_device_input.dtype == torch.int8:
      if self._fast_device_uint8 is None or self._fast_device_int16 is None:
        raise RuntimeError("INT8 preprocessing buffers are unavailable")
      self._fast_device_uint8.copy_(self._fast_host_input, non_blocking=True)
      self._fast_device_int16.copy_(self._fast_device_uint8)
      self._fast_device_int16.mul_(127).add_(127).floor_divide_(255)
      self._fast_device_input.copy_(self._fast_device_int16)
    else:
      self._fast_device_input.copy_(self._fast_host_input, non_blocking=True)
      self._fast_device_input.div_(255.0)
    timing["model_preprocess"] = self._finish_profiled_stage(torch, preprocess_start)

    inference_start = self._start_profiled_stage(torch)
    predictions = self._fast_backend(self._fast_device_input)
    timing["model_inference"] = self._finish_profiled_stage(torch, inference_start)

    postprocess_start = self._start_profiled_stage(torch)
    if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
      raise RuntimeError(f"expected two TensorRT outputs, got {type(predictions).__name__}")
    detections, prototypes = predictions
    if tuple(detections.shape) != (1, 300, 38):
      raise RuntimeError(f"unexpected detection output shape {tuple(detections.shape)}")
    if tuple(prototypes.shape) != (1, 32, height // 4, width // 4):
      raise RuntimeError(f"unexpected prototype output shape {tuple(prototypes.shape)}")

    if self.postprocess_backend == "cpu":
      timing["model_postprocess"] = self._finish_profiled_stage(torch, postprocess_start)
      tensor_start = time.perf_counter()
      detections_cpu = detections.detach().cpu()
      prototypes_cpu = prototypes.detach().cpu()
      if timing["tensor_to_cpu"] is not None:
        timing["tensor_to_cpu"] = time.perf_counter() - tensor_start

      cpu_postprocess_start = time.perf_counter()
      masks_cpu, prediction_cpu = self._postprocess_masks_cpu(
          detections_cpu,
          prototypes_cpu,
          height,
          width,
          ops,
      )
      if timing["model_postprocess"] is not None:
        timing["model_postprocess"] = (
            float(timing["model_postprocess"])
            + time.perf_counter()
            - cpu_postprocess_start
        )
    else:
      prediction = detections[0]
      prediction = prediction[prediction[:, 4] > self.conf][:self._fast_max_det]
      if prediction.shape[0] == 0:
        masks = None
      else:
        masks = ops.process_mask(
            prototypes[0],
            prediction[:, 6:].float(),
            prediction[:, :4].float(),
            (height, width),
            upsample=True,
        )
        prediction[:, :4] = ops.scale_boxes(
            (height, width),
            prediction[:, :4],
            image.shape,
        )
        keep = masks.amax((-2, -1)) > 0
        prediction = prediction[keep]
        masks = masks[keep]
      timing["model_postprocess"] = self._finish_profiled_stage(torch, postprocess_start)

    if self.profile_stages:
      timing["model_predict"] = time.perf_counter() - model_start
      separately_timed_transfer = (
          float(timing["tensor_to_cpu"])
          if self.postprocess_backend == "cpu" and timing["tensor_to_cpu"] is not None
          else 0.0
      )
      timing["model_framework_overhead"] = max(
          0.0,
          float(timing["model_predict"])
          - sum(
              float(timing[name])
              for name in ("model_preprocess", "model_inference", "model_postprocess")
          )
          - separately_timed_transfer,
      )
    else:
      timing["model_predict"] = None

    no_masks = (
        masks_cpu is None or prediction_cpu.shape[0] == 0
        if self.postprocess_backend == "cpu"
        else masks is None or prediction.shape[0] == 0
    )
    if no_masks:
      if not self.profile_stages:
        torch.cuda.synchronize(self._fast_backend.device)
      timing["total"] = time.perf_counter() - total_start
      self.last_timing = timing
      return None

    if self.postprocess_backend == "gpu":
      self.last_candidate_count = int(prediction.shape[0])
      selected_indices = self._select_best_candidate_indices_gpu(masks, prediction, timing)
      if selected_indices is None:
        timing["total"] = time.perf_counter() - total_start
        self.last_timing = timing
        return None

      tensor_start = time.perf_counter()
      masks_cpu = masks[selected_indices].detach().cpu().numpy()
      prediction_cpu = prediction[selected_indices, :6].detach().cpu().numpy()
      if self.profile_stages:
        timing["tensor_to_cpu"] = time.perf_counter() - tensor_start
    else:
      self.last_candidate_count = int(len(masks_cpu))

    best_result = self._select_best_result(
        masks_cpu,
        prediction_cpu[:, 5].astype(int),
        prediction_cpu[:, 4],
        prediction_cpu[:, :4],
        width,
        height,
        timing,
    )
    timing["total"] = time.perf_counter() - total_start
    self.last_timing = timing
    return best_result

  def _postprocess_masks_cpu(
      self,
      detections_cpu,
      prototypes_cpu,
      height: int,
      width: int,
      ops_module,
  ) -> tuple[np.ndarray | None, np.ndarray]:
    prediction = detections_cpu[0]
    prediction = prediction[prediction[:, 4] > self.conf][:self._fast_max_det]
    if prediction.shape[0] == 0:
      return None, np.empty((0, 6), dtype=np.float32)

    masks = self._process_masks_cpu_with_gpu_crop(
        prototypes_cpu[0],
        prediction[:, 6:].float(),
        prediction[:, :4].float(),
        (height, width),
    )
    prediction[:, :4] = ops_module.scale_boxes(
        (height, width),
        prediction[:, :4],
        (height, width),
    )
    keep = masks.amax((-2, -1)) > 0
    prediction = prediction[keep]
    masks = masks[keep]
    return (
        np.ascontiguousarray(masks.numpy(), dtype=np.uint8),
        np.ascontiguousarray(prediction[:, :6].float().numpy(), dtype=np.float32),
    )

  @staticmethod
  def _process_masks_cpu_with_gpu_crop(prototypes, mask_coefficients, boxes, shape):
    import torch
    import torch.nn.functional as functional

    channels, mask_height, mask_width = prototypes.shape
    masks = (
        mask_coefficients @ prototypes.float().view(channels, -1)
    ).view(-1, mask_height, mask_width)

    scaled_boxes = boxes.clone()
    scaled_boxes[:, (0, 2)] *= mask_width / float(shape[1])
    scaled_boxes[:, (1, 3)] *= mask_height / float(shape[0])
    x1, y1, x2, y2 = torch.chunk(scaled_boxes[:, :, None], 4, dim=1)
    columns = torch.arange(mask_width, dtype=x1.dtype)[None, None, :]
    rows = torch.arange(mask_height, dtype=x1.dtype)[None, :, None]
    masks *= (columns >= x1) * (columns < x2) * (rows >= y1) * (rows < y2)
    masks = functional.interpolate(
        masks[None],
        shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    return masks.gt_(0.0).byte()

  def _select_best_result(
      self,
      masks: np.ndarray,
      class_ids: np.ndarray,
      confidences: np.ndarray,
      boxes: np.ndarray,
      width: int,
      height: int,
      timing: dict[str, float | None],
  ) -> YoloMaskResult | None:

    best_result = None
    best_score = -1.0
    for index, raw_mask in enumerate(masks):
      select_start = time.perf_counter()
      class_id = int(class_ids[index])
      if self.target_class_id is not None and class_id != self.target_class_id:
        self._accumulate_timing(timing, "select_best_mask", time.perf_counter() - select_start)
        continue
      self._accumulate_timing(timing, "select_best_mask", time.perf_counter() - select_start)

      resize_start = time.perf_counter()
      mask = self._resize_and_clean_mask(raw_mask, width, height)
      self._accumulate_timing(timing, "mask_resize_clean", time.perf_counter() - resize_start)

      select_start = time.perf_counter()
      area = int(mask.sum())
      if area < self.min_mask_area:
        self._accumulate_timing(timing, "select_best_mask", time.perf_counter() - select_start)
        continue

      confidence = float(confidences[index])
      score = confidence * np.sqrt(area)
      if score <= best_score:
        self._accumulate_timing(timing, "select_best_mask", time.perf_counter() - select_start)
        continue

      best_score = score
      best_result = YoloMaskResult(
          mask=mask,
          class_id=class_id,
          class_name=self.names.get(class_id, str(class_id)),
          confidence=confidence,
          box_xyxy=boxes[index].astype(np.float32),
          area=area,
      )
      self._accumulate_timing(timing, "select_best_mask", time.perf_counter() - select_start)
    return best_result

  def _select_best_candidate_indices_gpu(
      self,
      masks,
      prediction,
      timing: dict[str, float | None],
  ) -> Any:
    import torch

    select_start = self._start_profiled_stage(torch)
    class_ids = prediction[:, 5].to(torch.int64)
    confidences = prediction[:, 4].float()

    keep = torch.ones_like(confidences, dtype=torch.bool)
    if self.target_class_id is not None:
      keep = class_ids == int(self.target_class_id)
    if not bool(keep.any().item()):
      elapsed = self._finish_profiled_stage(torch, select_start)
      if elapsed is not None:
        self._accumulate_timing(timing, "select_best_mask", elapsed)
      return None

    filtered_masks = masks[keep]
    filtered_confidences = confidences[keep]
    filtered_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)

    areas = filtered_masks.reshape(filtered_masks.shape[0], -1).sum(dim=1).float()
    scores = filtered_confidences * torch.sqrt(areas.clamp_min(1.0))
    topk = min(int(self.gpu_best_mask_topk), int(scores.shape[0]))
    best_local_indices = torch.topk(scores, k=topk, largest=True, sorted=True).indices
    best_global_indices = filtered_indices[best_local_indices]

    elapsed = self._finish_profiled_stage(torch, select_start)
    if elapsed is not None:
      self._accumulate_timing(timing, "select_best_mask", elapsed)
    return best_global_indices

  def _normalized_input_size(self) -> tuple[int, int]:
    if isinstance(self.imgsz, tuple):
      return int(self.imgsz[0]), int(self.imgsz[1])
    size = int(self.imgsz)
    return size, size

  def _validate_fast_input(self, image: np.ndarray) -> tuple[int, int]:
    if not isinstance(image, np.ndarray):
      raise TypeError(f"expected numpy.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
      raise ValueError(f"expected uint8 input, got {image.dtype}")
    height, width = self._normalized_input_size()
    expected_shape = (height, width, 3)
    if image.shape != expected_shape:
      raise ValueError(f"expected input shape {expected_shape}, got {image.shape}")
    if not image.flags.c_contiguous:
      raise ValueError("expected C-contiguous input")
    return height, width

  def _start_profiled_stage(self, torch_module) -> float | None:
    if not self.profile_stages:
      return None
    torch_module.cuda.synchronize(self._fast_backend.device)
    return time.perf_counter()

  def _finish_profiled_stage(self, torch_module, started: float | None) -> float | None:
    if started is None:
      return None
    torch_module.cuda.synchronize(self._fast_backend.device)
    return time.perf_counter() - started

  @staticmethod
  def _accumulate_timing(timing: dict[str, float | None], name: str, elapsed: float) -> None:
    if timing[name] is not None:
      timing[name] = float(timing[name]) + elapsed

  def _fallback_to_legacy(self, reason: str) -> None:
    if self.fast_path_fallback_reason is None:
      self.fast_path_fallback_reason = reason
      print(f"[YOLO][FAST] falling back to legacy path: {reason}", file=sys.stderr)
    self.active_execution_path = "legacy"

  def execution_metadata(self) -> dict[str, Any]:
    return {
        "requested_path": self.requested_execution_path,
        "active_path": self.active_execution_path,
        "last_path": self.last_execution_path,
        "postprocess_backend": self.postprocess_backend,
        "selected_weights": self.selected_weights,
        "profile_stages": self.profile_stages,
        "fallback_to_legacy": self.fallback_to_legacy,
        "fallback_reason": self.fast_path_fallback_reason,
        "contract": self.fast_path_contract,
        "mask_cleanup": {
            "fast_count": self.mask_cleanup_fast_count,
            "legacy_count": self.mask_cleanup_legacy_count,
        },
    }

  def _resize_and_clean_mask(self, raw_mask: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    raw_mask_array = np.asarray(raw_mask)
    if (
        raw_mask_array.shape == (height, width)
        and raw_mask_array.dtype == np.uint8
        and raw_mask_array.flags.c_contiguous
        and raw_mask_array.size > 0
        and int(raw_mask_array.max()) <= 1
    ):
      self.mask_cleanup_fast_count += 1
      mask = raw_mask_array
      if self.morphology_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morphology_kernel)
        return mask
      return mask.copy()

    self.mask_cleanup_legacy_count += 1
    mask = np.asarray(raw_mask, dtype=np.float32)
    mask_height, mask_width = mask.shape[-2:]

    gain = min(mask_width / float(width), mask_height / float(height))
    pad_width = max(mask_width - width * gain, 0.0) / 2.0
    pad_height = max(mask_height - height * gain, 0.0) / 2.0
    left = max(0, int(round(pad_width - 0.1)))
    right = min(mask_width, int(round(mask_width - pad_width + 0.1)))
    top = max(0, int(round(pad_height - 0.1)))
    bottom = min(mask_height, int(round(mask_height - pad_height + 0.1)))

    if right > left and bottom > top:
      mask = mask[top:bottom, left:right]

    if mask.shape != (height, width):
      mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 0.5).astype(np.uint8)

    if self.morphology_kernel is not None:
      mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel)
      mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morphology_kernel)

    return mask

  def _class_name_to_id(self, class_name: str) -> int:
    for class_id, name in self.names.items():
      if name == class_name:
        return class_id
    raise ValueError(f"target_class '{class_name}' was not found in YOLO class names: {self.names}")

  @staticmethod
  def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
      return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def normalize_imgsz(value: Any) -> int | tuple[int, int]:
  if isinstance(value, (list, tuple)):
    if len(value) != 2:
      raise ValueError("imgsz must be an integer or [height, width]")
    height, width = (int(item) for item in value)
    if height <= 0 or width <= 0:
      raise ValueError("imgsz dimensions must be positive")
    return height, width

  size = int(value)
  if size <= 0:
    raise ValueError("imgsz must be positive")
  return size


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Test YOLO segmentation mask inference on a single image.")
  parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to YOLO segmentation YAML config.")
  parser.add_argument("--image", type=str, default=None, help="Override test image path from YAML.")
  parser.add_argument("--weights", type=str, default=None, help="Override YOLO segmentation weights path from YAML.")
  parser.add_argument("--target-class", type=str, default=None, help="Override target class name from YAML.")
  parser.add_argument("--target-class-id", type=int, default=None, help="Override target class id from YAML.")
  parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold from YAML.")
  parser.add_argument(
      "--imgsz",
      type=int,
      nargs="+",
      default=None,
      help="Override image size with one value or height width, for example --imgsz 480 640.",
  )
  parser.add_argument("--device", type=str, default=None, help="Override device from YAML, for example cuda:0 or cpu.")
  parser.add_argument("--half", action="store_true", default=None, help="Override YAML and use FP16 inference.")
  parser.add_argument("--no-half", action="store_false", dest="half", help="Override YAML and disable FP16 inference.")
  parser.add_argument("--min-mask-area", type=int, default=None, help="Override minimum mask area from YAML.")
  parser.add_argument("--morph-kernel", type=int, default=None, help="Override morphology kernel size from YAML.")
  parser.add_argument(
      "--gpu-best-mask-topk",
      type=int,
      default=None,
      help="Override GPU preselection top-k candidates before CPU mask cleanup.",
  )
  parser.add_argument("--execution-path", choices=("legacy", "fast"), default=None)
  parser.add_argument("--profile-stages", action="store_true", default=None)
  parser.add_argument("--no-profile-stages", action="store_false", dest="profile_stages")
  parser.add_argument("--fallback-to-legacy", action="store_true", default=None)
  parser.add_argument("--no-fallback-to-legacy", action="store_false", dest="fallback_to_legacy")
  parser.add_argument("--postprocess-backend", choices=("gpu", "cpu"), default=None)
  parser.add_argument("--output-dir", type=str, default=None, help="Override output directory from YAML.")
  parser.add_argument("--show", action="store_true", help="Show overlay preview window.")
  return parser.parse_args()


def load_yolo_config(path: str) -> dict:
  try:
    import yaml
  except ImportError as exc:
    raise ImportError("PyYAML is required to read YOLO config. Install it with: pip install pyyaml") from exc

  if not os.path.exists(path):
    raise RuntimeError(f"YOLO config file does not exist: {path}")
  with open(path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file) or {}
  if not isinstance(data, dict):
    raise RuntimeError(f"YOLO config must be a YAML mapping: {path}")
  return data


def resolve_config_path(path: str | None, config_dir: str) -> str | None:
  if path is None or os.path.isabs(path):
    return path
  return os.path.abspath(os.path.join(config_dir, path))


def build_runtime_config(args: argparse.Namespace) -> dict:
  config_path = os.path.abspath(args.config)
  config_dir = os.path.dirname(config_path)
  config = load_yolo_config(config_path)
  runtime_config = {
      "image": config.get("image"),
      "weights": config.get("weights", "../yolo_weights/best.pt"),
      "target_class": config.get("target_class"),
      "target_class_id": config.get("target_class_id"),
      "conf": float(config.get("conf", 0.35)),
      "imgsz": normalize_imgsz(config.get("imgsz", 640)),
      "device": config.get("device"),
      "half": bool(config.get("half", True)),
      "min_mask_area": int(config.get("min_mask_area", 100)),
      "morph_kernel": int(config.get("morph_kernel", 5)),
      "gpu_best_mask_topk": int(config.get("gpu_best_mask_topk", 1)),
      "execution_path": str(config.get("execution_path", "legacy")),
      "profile_stages": bool(config.get("profile_stages", False)),
      "fallback_to_legacy": bool(config.get("fallback_to_legacy", True)),
      "postprocess_backend": str(config.get("postprocess_backend", "gpu")),
      "output_dir": config.get("output_dir", "../outputs/yolo_segment_test"),
  }

  for key in (
      "image",
      "weights",
      "target_class",
      "target_class_id",
      "conf",
      "imgsz",
      "device",
      "half",
      "min_mask_area",
      "morph_kernel",
      "gpu_best_mask_topk",
      "execution_path",
      "profile_stages",
      "fallback_to_legacy",
      "postprocess_backend",
      "output_dir",
  ):
    value = getattr(args, key)
    if value is not None:
      runtime_config[key] = value

  runtime_config["imgsz"] = normalize_imgsz(runtime_config["imgsz"])

  runtime_config["image"] = resolve_config_path(runtime_config["image"], config_dir)
  runtime_config["weights"] = resolve_config_path(runtime_config["weights"], config_dir)
  runtime_config["output_dir"] = resolve_config_path(runtime_config["output_dir"], config_dir)
  runtime_config["config_path"] = config_path
  return runtime_config


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
  import cv2

  overlay = image_bgr.copy()
  mask_bool = mask.astype(bool)
  overlay[mask_bool] = (0, 0, 255)
  return cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)


def run_cli(args: argparse.Namespace) -> None:
  import cv2

  config = build_runtime_config(args)
  print(f"[YOLO] Loaded config: {config['config_path']}")
  print("[YOLO] Requested inference config:")
  print(f"  image: {config['image']}")
  print(f"  weights: {config['weights']}")
  print(f"  target_class: {config['target_class']}")
  print(f"  target_class_id: {config['target_class_id']}")
  print(f"  conf: {config['conf']}")
  print(f"  imgsz: {config['imgsz']}")
  print(f"  device: {config['device']}")
  print(f"  half: {config['half']}")
  print(f"  execution_path: {config['execution_path']}")
  print(f"  gpu_best_mask_topk: {config['gpu_best_mask_topk']}")
  print(f"  profile_stages: {config['profile_stages']}")
  print(f"  fallback_to_legacy: {config['fallback_to_legacy']}")
  print(f"  postprocess_backend: {config['postprocess_backend']}")

  if config["image"] is None:
    raise RuntimeError("No test image configured. Set image in yolo_segment_config.yml or pass --image.")
  if not os.path.exists(config["image"]):
    raise RuntimeError(f"Test image does not exist: {config['image']}")
  if not os.path.exists(config["weights"]):
    raise RuntimeError(f"YOLO weights do not exist: {config['weights']}")

  image_bgr = cv2.imread(config["image"])
  if image_bgr is None:
    raise RuntimeError(f"Failed to read image: {config['image']}")

  detector = YoloSegmenter(
      weights=config["weights"],
      target_class=config["target_class"],
      target_class_id=config["target_class_id"],
      conf=config["conf"],
      imgsz=config["imgsz"],
      device=config["device"],
      half=config["half"],
      min_mask_area=config["min_mask_area"],
      morph_kernel=config["morph_kernel"],
      gpu_best_mask_topk=config["gpu_best_mask_topk"],
      input_is_rgb=True,
      execution_path=config["execution_path"],
      profile_stages=config["profile_stages"],
      fallback_to_legacy=config["fallback_to_legacy"],
      postprocess_backend=config["postprocess_backend"],
  )

  image_rgb = image_bgr[..., ::-1]
  result = detector.predict_mask(image_rgb)
  if result is None:
    print("[YOLO] No target mask detected")
    return

  print("[YOLO] Detection result:")
  print(f"  class_id: {result.class_id}")
  print(f"  class_name: {result.class_name}")
  print(f"  confidence: {result.confidence:.4f}")
  print(f"  area: {result.area}")
  print(f"  box_xyxy: {result.box_xyxy.tolist()}")

  os.makedirs(config["output_dir"], exist_ok=True)
  mask_path = os.path.join(config["output_dir"], "mask.png")
  overlay_path = os.path.join(config["output_dir"], "overlay.png")
  cv2.imwrite(mask_path, (result.mask * 255).astype(np.uint8))
  overlay = make_overlay(image_bgr, result.mask)
  cv2.imwrite(overlay_path, overlay)
  print(f"[YOLO] Saved mask: {mask_path}")
  print(f"[YOLO] Saved overlay: {overlay_path}")

  if args.show:
    cv2.imshow("YOLO segmentation overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
  args = parse_args()
  try:
    run_cli(args)
  except (ImportError, RuntimeError, ValueError) as exc:
    print(f"[YOLO] {exc}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()