from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import trimesh


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOUNDATIONPOSE_DIR = os.path.join(REPO_ROOT, "FoundationPose")
if FOUNDATIONPOSE_DIR not in sys.path:
  sys.path.insert(0, FOUNDATIONPOSE_DIR)

from estimater import (  # noqa: E402
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    dr,
    draw_posed_3d_box,
    draw_xyz_axis,
    nvdiffrast_render,
    set_logging_format,
    set_seed,
)
import learning.training.predict_pose_refine as pose_refine_module  # noqa: E402
import learning.training.predict_score as score_module  # noqa: E402


_ORIGINAL_COMPUTE_CROP_WINDOW_TF_BATCH = pose_refine_module.compute_crop_window_tf_batch


def _compute_crop_window_tf_batch_float32(*, pts, H, W, poses, K, crop_ratio, out_size, method, mesh_diameter=None):
  if torch.is_tensor(pts):
    pts = pts.to(device="cuda", dtype=torch.float32)
  else:
    pts = torch.as_tensor(np.asarray(pts, dtype=np.float32), device="cuda", dtype=torch.float32)

  if torch.is_tensor(poses):
    poses = poses.to(device="cuda", dtype=torch.float32)
  else:
    poses = torch.as_tensor(np.asarray(poses, dtype=np.float32), device="cuda", dtype=torch.float32)

  if torch.is_tensor(K):
    K = K.to(device="cuda", dtype=torch.float32)
  else:
    K = torch.as_tensor(np.asarray(K, dtype=np.float32), device="cuda", dtype=torch.float32)

  if method != "box_3d":
    return _ORIGINAL_COMPUTE_CROP_WINDOW_TF_BATCH(
        pts=pts,
        H=H,
        W=W,
        poses=poses,
        K=K,
        crop_ratio=float(crop_ratio),
        out_size=out_size,
        method=method,
        mesh_diameter=float(mesh_diameter) if mesh_diameter is not None else None,
    )

  batch_size = len(poses)
  radius = float(mesh_diameter) * float(crop_ratio) / 2.0
  offsets = torch.tensor(
      [
          [0.0, 0.0, 0.0],
          [radius, 0.0, 0.0],
          [-radius, 0.0, 0.0],
          [0.0, radius, 0.0],
          [0.0, -radius, 0.0],
      ],
      device="cuda",
      dtype=torch.float32,
  )
  crop_pts = poses[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
  projected = (K.reshape(3, 3) @ crop_pts.reshape(-1, 3).T).T
  uvs = projected[:, :2] / projected[:, 2:3]
  uvs = uvs.reshape(batch_size, -1, 2)
  center = uvs[:, 0]
  radius_px = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(batch_size, -1).max(axis=-1)[0].reshape(-1)
  left = center[:, 0] - radius_px
  right = center[:, 0] + radius_px
  top = center[:, 1] - radius_px
  bottom = center[:, 1] + radius_px
  return _compute_crop_tf_batch_float32(left, right, top, bottom, out_size)


def _compute_crop_tf_batch_float32(left: torch.Tensor, right: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor, out_size) -> torch.Tensor:
  batch_size = len(left)
  left = left.round()
  right = right.round()
  top = top.round()
  bottom = bottom.round()

  tf = torch.eye(3, device="cuda", dtype=torch.float32)[None].expand(batch_size, -1, -1).contiguous()
  tf[:, 0, 2] = -left
  tf[:, 1, 2] = -top
  new_tf = torch.eye(3, device="cuda", dtype=torch.float32)[None].expand(batch_size, -1, -1).contiguous()
  new_tf[:, 0, 0] = float(out_size[0]) / (right - left)
  new_tf[:, 1, 1] = float(out_size[1]) / (bottom - top)
  return new_tf @ tf


def _set_crop_window_patch(enabled: bool) -> None:
  crop_fn = _compute_crop_window_tf_batch_float32 if enabled else _ORIGINAL_COMPUTE_CROP_WINDOW_TF_BATCH
  pose_refine_module.compute_crop_window_tf_batch = crop_fn
  score_module.compute_crop_window_tf_batch = crop_fn


@dataclass
class PoseResult:
  pose: np.ndarray
  initialized: bool
  mode: str


class FoundationPoseRealtimeTracker:
  def __init__(
      self,
      mesh_file: str,
      debug_dir: str,
      debug: int = 1,
      use_float32_crop_window_patch: bool = True,
      init_min_n_views: int = 40,
      init_inplane_step: int = 60,
      est_refine_iter: int = 5,
      init_strategy: str = "default",
      coarse_refine_iter: int = 1,
      coarse_score_filter: str = "none",
      coarse_score_top_k: int = 999999,
      fine_refine_iter: int = 2,
      fine_top_k: int = 16,
      track_refine_iter: int = 2,
      vis_mode: str = "box",
      contour_thickness: int = 3,
      axis_scale: float = 0.1,
  ):
    set_logging_format()
    set_seed(0)

    self.mesh_file = mesh_file
    self.debug_dir = debug_dir
    self.debug = debug
    self.use_float32_crop_window_patch = use_float32_crop_window_patch
    self.init_min_n_views = init_min_n_views
    self.init_inplane_step = init_inplane_step
    self.est_refine_iter = est_refine_iter
    self.init_strategy = init_strategy
    self.coarse_refine_iter = coarse_refine_iter
    self.coarse_score_filter = coarse_score_filter
    self.coarse_score_top_k = coarse_score_top_k
    self.fine_refine_iter = fine_refine_iter
    self.fine_top_k = fine_top_k
    self.track_refine_iter = track_refine_iter
    self.vis_mode = vis_mode
    self.contour_thickness = contour_thickness
    self.axis_scale = axis_scale
    self.initialized = False
    self.last_pose = None

    _set_crop_window_patch(self.use_float32_crop_window_patch)

    os.makedirs(self.debug_dir, exist_ok=True)
    self.mesh = trimesh.load(mesh_file)
    self.mesh.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
    self.model_normals = np.asarray(self.mesh.vertex_normals, dtype=np.float32)
    self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
    self.to_origin = np.asarray(self.to_origin, dtype=np.float32)
    self.extents = np.asarray(self.extents, dtype=np.float32)
    self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3).astype(np.float32)

    self.scorer = ScorePredictor()
    self.refiner = PoseRefinePredictor()
    self.glctx = dr.RasterizeCudaContext()
    self.estimator = FoundationPose(
        model_pts=self.mesh.vertices,
        model_normals=self.model_normals,
        mesh=self.mesh,
        scorer=self.scorer,
        refiner=self.refiner,
        debug_dir=self.debug_dir,
        debug=self.debug,
        glctx=self.glctx,
        init_min_n_views=self.init_min_n_views,
        init_inplane_step=self.init_inplane_step,
    )
    self._ensure_estimator_float32()

  def reset(self) -> None:
    self.initialized = False
    self.last_pose = None
    self.estimator.pose_last = None

  def register(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> PoseResult:
    color, depth, K = self._prepare_frame_inputs(color, depth, K)
    mask = self._valid_mask(mask, depth)
    pose = self.estimator.register(
        K=K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=self.est_refine_iter,
        init_strategy=self.init_strategy,
        coarse_refine_iter=self.coarse_refine_iter,
        coarse_score_filter=self.coarse_score_filter,
        coarse_score_top_k=self.coarse_score_top_k,
        fine_refine_iter=self.fine_refine_iter,
        fine_top_k=self.fine_top_k,
    )
    self.initialized = True
    self.last_pose = pose
    return PoseResult(pose=pose, initialized=True, mode="register")

  def track(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray) -> PoseResult:
    if not self.initialized:
      raise RuntimeError("FoundationPoseRealtimeTracker.track called before register")
    color, depth, K = self._prepare_frame_inputs(color, depth, K)
    pose = self.estimator.track_one(rgb=color, depth=depth, K=K, iteration=self.track_refine_iter)
    self.last_pose = pose
    return PoseResult(pose=pose, initialized=True, mode="track")

  def render_pose_mask(self, K: np.ndarray, image_shape: tuple[int, int], pose: np.ndarray | None = None) -> np.ndarray:
    pose_to_render = self.estimator.pose_last if pose is None else pose
    K = np.ascontiguousarray(K, dtype=np.float32)
    height, width = image_shape[:2]
    ob_in_cams = torch.as_tensor(pose_to_render, device="cuda", dtype=torch.float).reshape(1, 4, 4)
    _, render_depth, _ = nvdiffrast_render(
        K=K,
        H=height,
        W=width,
        ob_in_cams=ob_in_cams,
        glctx=self.glctx,
        mesh_tensors=self.estimator.mesh_tensors,
        output_size=np.asarray([height, width]),
        use_light=False,
    )
    return (render_depth[0].detach().cpu().numpy() > 0.001).astype(np.uint8)

  def mask_iou(self, K: np.ndarray, image_shape: tuple[int, int], target_mask: np.ndarray) -> float:
    if not self.initialized:
      return 0.0
    rendered_mask = self.render_pose_mask(K, image_shape)
    target = np.ascontiguousarray(target_mask > 0, dtype=np.uint8)
    intersection = np.logical_and(rendered_mask > 0, target > 0).sum()
    union = np.logical_or(rendered_mask > 0, target > 0).sum()
    return float(intersection / union) if union > 0 else 0.0

  def draw_visualization(self, color: np.ndarray, K: np.ndarray, pose: np.ndarray | None = None) -> np.ndarray:
    color = np.ascontiguousarray(color, dtype=np.uint8)
    K = np.ascontiguousarray(K, dtype=np.float32)
    if pose is None:
      pose = self.last_pose
    if pose is None:
      return color.copy()

    center_pose = pose @ np.linalg.inv(self.to_origin)
    vis = color.copy()
    if self.vis_mode in ("box", "both"):
      vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=self.bbox)
    if self.vis_mode in ("contour", "both"):
      contour_mask = self.render_pose_mask(K, color.shape[:2]) * 255
      contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if contours:
        cv2.drawContours(vis, contours, -1, color=(255, 255, 0), thickness=self.contour_thickness, lineType=cv2.LINE_AA)
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=self.axis_scale,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    return vis

  @staticmethod
  def _valid_mask(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    mask = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    mask = mask & (depth >= 0.001)
    return np.ascontiguousarray(mask, dtype=np.uint8)

  @staticmethod
  def _prepare_frame_inputs(color: np.ndarray, depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color = np.ascontiguousarray(color, dtype=np.uint8)
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    K = np.ascontiguousarray(K, dtype=np.float32)
    return color, depth, K

  def _ensure_estimator_float32(self) -> None:
    self.estimator.mesh.vertices = np.asarray(self.estimator.mesh.vertices, dtype=np.float32)
    self.estimator.pts = self.estimator.pts.float().contiguous()
    self.estimator.normals = self.estimator.normals.float().contiguous()
    self.estimator.rot_grid = self.estimator.rot_grid.float().contiguous()
    for key, value in self.estimator.mesh_tensors.items():
      if torch.is_tensor(value) and value.is_floating_point():
        self.estimator.mesh_tensors[key] = value.float().contiguous()