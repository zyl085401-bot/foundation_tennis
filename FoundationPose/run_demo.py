# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import psutil
import subprocess
import threading
import time


class ResourceMonitor:
  def __init__(self, interval=0.2):
    self.interval = interval
    self.running = False
    self.thread = None
    self.gpu_peak_mib = 0
    self.cpu_rss_peak_gb = 0
    self.cpu_percent_peak = 0
    self.process = psutil.Process(os.getpid())

  def _sample_gpu_mib(self):
    try:
      out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
      values = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
      return max(values) if values else 0
    except Exception:
      return 0

  def _loop(self):
    self.process.cpu_percent(interval=None)
    while self.running:
      self.gpu_peak_mib = max(self.gpu_peak_mib, self._sample_gpu_mib())
      self.cpu_rss_peak_gb = max(self.cpu_rss_peak_gb, self.process.memory_info().rss/(1024**3))
      self.cpu_percent_peak = max(self.cpu_percent_peak, self.process.cpu_percent(interval=None))
      time.sleep(self.interval)

  def start(self):
    self.running = True
    self.thread = threading.Thread(target=self._loop, daemon=True)
    self.thread.start()

  def stop(self):
    self.running = False
    if self.thread is not None:
      self.thread.join()

  @property
  def gpu_peak_gb(self):
    return self.gpu_peak_mib/1024.0

  @property
  def cpu_peak_cores(self):
    return self.cpu_percent_peak/100.0


def _format_time(seconds):
  if seconds is None:
    return 'N/A'
  if seconds < 1.0:
    return f'{seconds*1000:.3f} ms'
  return f'{seconds:.3f} s'


def print_runtime_summary(total_time, register_time, refiner_time, scorer_time, track_times, monitor):
  track_avg = sum(track_times)/len(track_times) if len(track_times)>0 else None
  track_fps = 1.0/track_avg if track_avg is not None and track_avg>0 else None

  print('\n========== Runtime Summary ==========')
  print('总耗时:')
  print(f'  run_demo.py: {total_time:.3f} s')

  print('\n第一帧:')
  print(f'  register: {register_time:.3f} s' if register_time is not None else '  register: N/A')
  print(f'  refiner: {refiner_time:.3f} s' if refiner_time is not None else '  refiner: N/A')
  print(f'  scorer: {scorer_time:.3f} s' if scorer_time is not None else '  scorer: N/A')

  print('\n后续帧:')
  print(f'  track avg: {track_avg*1000:.3f} ms' if track_avg is not None else '  track avg: N/A')
  print(f'  track FPS: {track_fps:.2f} FPS' if track_fps is not None else '  track FPS: N/A')

  print('\n资源:')
  print(f'  GPU peak: {monitor.gpu_peak_gb:.3f} GB')
  print(f'  CPU RAM peak: {monitor.cpu_rss_peak_gb:.3f} GB')
  print(f'  CPU usage: {monitor.cpu_peak_cores:.2f} cores')
  print('=====================================\n')


def draw_mesh_contour(K, img, ob_in_cam, mesh_tensors, glctx, line_color=(255,255,0), linewidth=3):
  H, W = img.shape[:2]
  ob_in_cams = torch.as_tensor(ob_in_cam, device='cuda', dtype=torch.float).reshape(1,4,4)
  _, render_depth, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=ob_in_cams, glctx=glctx, mesh_tensors=mesh_tensors, output_size=np.asarray([H,W]), use_light=False)
  mask = (render_depth[0].detach().cpu().numpy()>0.001).astype(np.uint8)*255
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  vis = img.copy()
  if len(contours)>0:
    cv2.drawContours(vis, contours, -1, color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
  return vis


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/cup0708/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/cup0708')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--vis_mode', type=str, default='box', choices=['box', 'contour', 'both'], help='Visualization overlay: 3D bbox, mesh contour, or both')
  parser.add_argument('--contour_thickness', type=int, default=3)
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  monitor = ResourceMonitor(interval=0.2)
  monitor.start()
  total_t0 = time.perf_counter()
  register_time = None
  refiner_time = None
  scorer_time = None
  track_times = []

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      timing = getattr(est, 'last_register_timing', {})
      register_time = timing.get('register')
      refiner_time = timing.get('refiner')
      scorer_time = timing.get('scorer')

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
      timing = getattr(est, 'last_track_timing', {})
      if 'track' in timing:
        track_times.append(timing['track'])

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = color.copy()
      if args.vis_mode in ['box', 'both']:
        vis = draw_posed_3d_box(reader.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
      if args.vis_mode in ['contour', 'both']:
        vis = draw_mesh_contour(reader.K, img=vis, ob_in_cam=est.pose_last, mesh_tensors=est.mesh_tensors, glctx=glctx, linewidth=args.contour_thickness)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

  torch.cuda.synchronize()
  total_time = time.perf_counter() - total_t0
  monitor.stop()
  print_runtime_summary(total_time, register_time, refiner_time, scorer_time, track_times, monitor)

