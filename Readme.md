
# 使用foundationpose的时候这个命令会显示分割物体轮廓形状的效果

python run_demo.py --vis_mode contour

# 仿真
python3 realtime_foundation/simulate_recorded_camera.py \
  --input realtime_foundation/outputs/error/refiner_coarse_160/frame_001172_register.npz \
  --config realtime_foundation/config.yaml

# 如需查看同一次 register 的完整候选筛选流水线，将 config.yaml 最底部的
# simulation.candidate_pipeline_debug_enabled 改为 true。结果保存在：
# outputs/simulated_results/<帧名>/candidate_pipeline/repeat_001/
# 其中 candidate_pipeline.json/npz 保存完整数据，各阶段 *_contact_sheet.jpg 用于人工检查。

export DISPLAY=:0
xhost +si:localuser:root

cd ~/workspace/yolo_foundationpose_ball_20hz

bash FoundationPose/docker/run_container_jetson.sh \
  python3 realtime_foundation/run_realtime.py \
  --config realtime_foundation/config.yaml

# 在基础容器里启动相机发布器 启动RealSense并通过mROS发布彩色、对齐深度和相机内参话题
cd /home/nvidia/workspace/yolo_foundationpose_ball_20hz

CONTAINER_NAME=foundationpose-mros-camera \
bash FoundationPose/docker/run_container_jetson.sh \
python3 realtime_foundation/camera/depth_color_publish \
  --config realtime_foundation/config.yaml


# 在联网的x86_64开发机准备Jetson Python 3.10/aarch64的mROS离线依赖。
bash FoundationPose/docker/prepare_jetson_mros_offline.sh

# 将仓库同步到NX后，离线构建包含mROS的基础镜像。
bash FoundationPose/docker/build_jetson.sh

# 进入NX容器后验证mROS和RealSense导入。
python3 -c 'import mros, pyrealsense2; print(mros.__file__)'

# Jetson 性能关联采集：同时记录运行日志、50 ms GPU 状态、200 ms
# GPU/EMC/温度/功耗和 tegrastats。Ctrl+C 后自动生成 events.csv、
# correlated_events.csv 和 correlation_summary.json。
bash FoundationPose/docker/run_jetson_telemetry.sh


# 初始化阶段的数据流与计时

# tracker 未初始化时，主线程直接等待新的 YOLO DetectionMessage。YOLO
# 发布候选目标后会唤醒主线程，并暂停提交下一帧推理；主线程立即使用消息
# 自带的同帧 RGB、depth、K 和 mask 执行 FoundationPose register。候选被
# 拒绝、register 失败或 init_only 完成并 reset 后，YOLO 恢复搜索。
#
# timing summary 中：
#   yolo_total                       YOLO 纯检测计算时间
#   foundation_total                 FoundationPose register 计算时间
#   yolo_foundation_compute_total    上述两项之和（纯模型计算总时间）
#   detection_consume_delay          YOLO 完成到主线程取得结果的通知/调度延迟
#   init_total                       YOLO 开始到初始化质量检查通过的端到端时间

# 减小mesh模型的大小
  python3 FoundationPose/tools/generate_textured_lod.py \
  FoundationPose/demo_data/cup0708/mesh/textured_simple.obj \
  FoundationPose/demo_data/cup0708/mesh/textured_simple_lod3.obj \
  --target-triangles 7000


  python3 FoundationPose/tools/generate_textured_lod.py \
  FoundationPose/demo_data/tennis/mesh/textured_simple.obj \
  FoundationPose/demo_data/tennis/mesh/textured_simple_0.03125.obj \
  --triangle-ratio 0.03125

# 在外部x86主机连接NX上的mROS Agent。
export MROS_AGENT_URI=tcp://192.168.55.2:11315
mrostopic list

# FoundationPose通过mROS发布以下结果：
# /foundationpose/object_pose
# /foundationpose/object_pose_base
# /foundationpose/status
# /foundationpose/visualization
# /wheelarm/target


source /home/nvidia/jason/bbs/install/setup.bash
export MROS_AGENT_IP=192.168.55.2
mrosagent