from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsense_config.yml")


class RealSenseReader:
  def __init__(
      self,
      width: int = 640,
      height: int = 480,
      fps: int = 30,
      serial: str | None = None,
      depth_min: float = 0.001,
      depth_max: float = 3.0,
      align_to_color: bool = True,
      reset_before_start: bool = True,
      verbose: bool = False,
  ):
    try:
      import pyrealsense2 as rs
    except ImportError as exc:
      raise ImportError(
          "pyrealsense2 is required for RealSense input. Install it in your environment or replace this reader."
      ) from exc

    self.rs = rs
    self.width = width
    self.height = height
    self.fps = fps
    self.serial = serial
    self.depth_min = depth_min
    self.depth_max = depth_max
    self.align_to_color = align_to_color
    self.reset_before_start = reset_before_start
    self.verbose = verbose
    self.pipeline = None
    self.config = None
    self.align = None
    self.profile = None
    self.depth_scale = None
    self.K = None
    self.last_frame_error_log_time = 0.0
    self._create_pipeline_objects()

  def _create_pipeline_objects(self) -> None:
    self.pipeline = self.rs.pipeline()
    self.config = self.rs.config()
    if self.serial:
      self.config.enable_device(self.serial)
    self.config.enable_stream(self.rs.stream.depth, self.width, self.height, self.rs.format.z16, self.fps)
    self.config.enable_stream(self.rs.stream.color, self.width, self.height, self.rs.format.bgr8, self.fps)
    self.align = self.rs.align(self.rs.stream.color) if self.align_to_color else None

  def start(self) -> None:
    if self.reset_before_start:
      self.stop()
      self._create_pipeline_objects()

    devices = self.query_devices()
    self._log_requested_config(devices)
    if not devices:
      raise RuntimeError("No RealSense device found. Check USB connection, permissions, and librealsense installation.")
    if self.serial and all(device["serial"] != self.serial for device in devices):
      serials = ", ".join(device["serial"] for device in devices)
      raise RuntimeError(f"Requested RealSense serial '{self.serial}' was not found. Available serials: {serials}")

    try:
      self.profile = self.pipeline.start(self.config)
    except RuntimeError as exc:
      raise RuntimeError(f"Failed to start RealSense pipeline: {exc}") from exc

    depth_sensor = self.profile.get_device().first_depth_sensor()
    self.depth_scale = float(depth_sensor.get_depth_scale())
    color_stream = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    self.K = np.array(
        [[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    self._log_started_config(color_stream)

  def stop(self) -> None:
    if self.profile is not None and self.pipeline is not None:
      try:
        self.pipeline.stop()
      except RuntimeError as exc:
        print(f"[RealSense] Failed to stop pipeline cleanly: {exc}")
      finally:
        self.profile = None
        self.depth_scale = None
        self.K = None

  def get_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
      frames = self.pipeline.wait_for_frames(timeout_ms=5000)
      if self.align is not None:
        frames = self.align.process(frames)
    except RuntimeError as exc:
      self._log_frame_error(exc)
      return None
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
      return None

    color_bgr = np.asanyarray(color_frame.get_data())
    color_rgb = color_bgr[..., ::-1].copy()
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
    depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0
    self._validate_frame_shapes(color_rgb, depth)
    return color_rgb, depth, self.K.copy()

  def _log_frame_error(self, exc: RuntimeError) -> None:
    now = time.time()
    if now - self.last_frame_error_log_time >= 1.0:
      print(f"[RealSense] Failed to get frame, waiting for next frame: {exc}")
      self.last_frame_error_log_time = now

  def __enter__(self) -> "RealSenseReader":
    self.start()
    return self

  def __exit__(self, exc_type, exc, traceback) -> None:
    self.stop()

  def query_devices(self) -> list[dict[str, str]]:
    context = self.rs.context()
    devices = []
    for device in context.query_devices():
      devices.append({
          "name": self._camera_info(device, self.rs.camera_info.name),
          "serial": self._camera_info(device, self.rs.camera_info.serial_number),
          "firmware": self._camera_info(device, self.rs.camera_info.firmware_version),
          "usb_type": self._camera_info(device, self.rs.camera_info.usb_type_descriptor),
      })
    return devices

  def _log_requested_config(self, devices: list[dict[str, str]]) -> None:
    if not self.verbose:
      return
    print("[RealSense] Requested stream config:")
    print(f"  color: {self.width}x{self.height}@{self.fps} bgr8")
    print(f"  depth: {self.width}x{self.height}@{self.fps} z16")
    print(f"  depth range: {self.depth_min:.3f} m - {self.depth_max:.3f} m")
    print(f"  align depth to color: {self.align_to_color}")
    print(f"  reset before start: {self.reset_before_start}")
    print(f"  serial filter: {self.serial if self.serial else 'none'}")
    print(f"[RealSense] Devices found: {len(devices)}")
    for index, device in enumerate(devices):
      print(
          f"  #{index}: name={device['name']}, serial={device['serial']}, "
          f"firmware={device['firmware']}, usb={device['usb_type']}"
      )

  def _log_started_config(self, color_stream) -> None:
    if not self.verbose:
      return
    depth_stream = self.profile.get_stream(self.rs.stream.depth).as_video_stream_profile()
    color_intrinsics = color_stream.get_intrinsics()
    depth_intrinsics = depth_stream.get_intrinsics()
    print("[RealSense] Pipeline started")
    print("  frame alignment: depth aligned to color stream" if self.align_to_color else "  frame alignment: disabled")
    print(f"  depth scale: {self.depth_scale:.8f} m/unit")
    print(
        f"  actual color stream: {color_intrinsics.width}x{color_intrinsics.height}@{color_stream.fps()} "
        f"format={color_stream.format()}"
    )
    print(
        f"  actual depth stream: {depth_intrinsics.width}x{depth_intrinsics.height}@{depth_stream.fps()} "
        f"format={depth_stream.format()}"
    )
    print("  K:")
    print(self.K)

  def _validate_frame_shapes(self, color: np.ndarray, depth: np.ndarray) -> None:
    color_height, color_width = color.shape[:2]
    depth_height, depth_width = depth.shape[:2]
    if self.align_to_color and (color_height != depth_height or color_width != depth_width):
      raise RuntimeError(
          "Aligned RealSense frame shape mismatch: "
          f"color={color_width}x{color_height}, depth={depth_width}x{depth_height}. "
          "Depth must be aligned to color before FoundationPose tracking."
      )

  def _camera_info(self, device, info) -> str:
    try:
      if device.supports(info):
        return str(device.get_info(info))
    except RuntimeError:
      pass
    return "unknown"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Preview RealSense RGB-D frames and camera intrinsics.")
  parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to RealSense camera YAML config.")
  parser.add_argument("--show", action="store_true", help="Show RGB and depth preview windows.")
  parser.add_argument("--width", type=int, default=None, help="Override color/depth width from YAML.")
  parser.add_argument("--height", type=int, default=None, help="Override color/depth height from YAML.")
  parser.add_argument("--fps", type=int, default=None, help="Override stream FPS from YAML.")
  parser.add_argument("--serial", type=str, default=None, help="Override RealSense serial from YAML.")
  parser.add_argument("--depth-min", type=float, default=None, help="Override minimum valid depth in meters from YAML.")
  parser.add_argument("--depth-max", type=float, default=None, help="Override maximum valid depth in meters from YAML.")
  parser.add_argument("--no-align", action="store_true", help="Disable depth-to-color alignment for debugging only.")
  parser.add_argument("--no-reset-before-start", action="store_true", help="Do not stop and recreate the RealSense pipeline before starting.")
  parser.add_argument("--log-interval", type=float, default=None, help="Override seconds between frame debug prints from YAML.")
  return parser.parse_args()


def load_camera_config(path: str) -> dict:
  try:
    import yaml
  except ImportError as exc:
    raise ImportError("PyYAML is required to read RealSense camera config. Install it with: pip install pyyaml") from exc

  if not os.path.exists(path):
    raise RuntimeError(f"RealSense config file does not exist: {path}")
  with open(path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file) or {}
  if not isinstance(data, dict):
    raise RuntimeError(f"RealSense config must be a YAML mapping: {path}")
  return data


def build_runtime_config(args: argparse.Namespace) -> dict:
  config = load_camera_config(args.config)
  runtime_config = {
      "width": int(config.get("width", 640)),
      "height": int(config.get("height", 480)),
      "fps": int(config.get("fps", 30)),
      "serial": config.get("serial"),
      "depth_min": float(config.get("depth_min", 0.001)),
      "depth_max": float(config.get("depth_max", 3.0)),
      "align_to_color": bool(config.get("align_to_color", True)),
      "reset_before_start": bool(config.get("reset_before_start", True)),
      "log_interval": float(config.get("log_interval", 1.0)),
  }

  for key in ("width", "height", "fps", "serial", "depth_min", "depth_max", "log_interval"):
    value = getattr(args, key)
    if value is not None:
      runtime_config[key] = value
  if args.no_align:
    runtime_config["align_to_color"] = False
  if args.no_reset_before_start:
    runtime_config["reset_before_start"] = False
  runtime_config["config_path"] = os.path.abspath(args.config)
  return runtime_config


def depth_to_colormap(depth: np.ndarray, depth_max: float) -> np.ndarray:
  import cv2

  depth_vis = np.clip(depth / depth_max, 0.0, 1.0)
  depth_vis = (depth_vis * 255).astype(np.uint8)
  return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


def run_preview(args: argparse.Namespace) -> None:
  import cv2

  config = build_runtime_config(args)
  print(f"[RealSense] Loaded config: {config['config_path']}")

  with RealSenseReader(
      width=config["width"],
      height=config["height"],
      fps=config["fps"],
      serial=config["serial"],
      depth_min=config["depth_min"],
      depth_max=config["depth_max"],
      align_to_color=config["align_to_color"],
      reset_before_start=config["reset_before_start"],
      verbose=True,
  ) as reader:
    print("Press q or ESC to exit")

    frame_count = 0
    fps_count = 0
    last_log_time = time.perf_counter()
    while True:
      frame = reader.get_frame()
      if frame is None:
        continue

      color_rgb, depth, _ = frame
      frame_count += 1
      fps_count += 1
      now = time.perf_counter()
      if now - last_log_time >= config["log_interval"]:
        valid_depth = depth[depth > 0]
        median_depth = float(np.median(valid_depth)) if valid_depth.size else 0.0
        measured_fps = fps_count / (now - last_log_time)
        print(
            f"[Frame {frame_count}] color={color_rgb.shape}, depth={depth.shape}, "
            f"valid_depth={valid_depth.size}, median_depth={median_depth:.3f} m, fps={measured_fps:.2f}"
        )
        fps_count = 0
        last_log_time = now

      if args.show:
        color_bgr = color_rgb[..., ::-1]
        depth_bgr = depth_to_colormap(depth, config["depth_max"])
        preview = np.hstack([color_bgr, depth_bgr])
        cv2.imshow("RealSense RGB | Depth", preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
          break
      else:
        valid_depth = depth[depth > 0]
        median_depth = float(np.median(valid_depth)) if valid_depth.size else 0.0
        print(f"color={color_rgb.shape}, depth={depth.shape}, median_depth={median_depth:.3f} m")
        break

  cv2.destroyAllWindows()


def main() -> None:
  args = parse_args()
  try:
    run_preview(args)
  except (ImportError, RuntimeError) as exc:
    print(f"[RealSense] {exc}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()