# 使用foundationpose的时候这个命令会显示分割物体轮廓形状的效果

python run_demo.py --vis_mode contour


python3 -m pip install pyrealsense2
python3 -m pip install -U "ultralytics>=8.3.0" --no-deps

python3 realtime_foundation/simulate_recorded_camera.py \
  --input realtime_foundation/outputs/error/refiner_coarse_160/frame_001172_register.npz \
  --config realtime_foundation/config.yaml

# 如需查看同一次 register 的完整候选筛选流水线，将 config.yaml 最底部的
# simulation.candidate_pipeline_debug_enabled 改为 true。结果保存在：
# outputs/simulated_results/<帧名>/candidate_pipeline/repeat_001/
# 其中 candidate_pipeline.json/npz 保存完整数据，各阶段 *_contact_sheet.jpg 用于人工检查。

export DISPLAY=:0
xhost +si:localuser:root

cd ~/workspace/yolo_foundationpose

bash FoundationPose/docker/run_container_jetson_ros2.sh \
  python3 realtime_foundation/run_realtime.py \
  --config realtime_foundation/config.yaml

# 启动RealSense并通过mROS发布彩色、对齐深度和相机内参话题
python3 realtime_foundation/camera/depth_color_publish \
  --config realtime_foundation/config.yaml

# 相机分辨率、帧率、序列号、mROS话题和曝光参数统一在config.yaml的camera/camera.mros中设置。

# 在联网的x86_64开发机准备Jetson Python 3.10/aarch64的mROS离线依赖。
bash FoundationPose/docker/prepare_jetson_mros_offline.sh

# 将仓库同步到NX后，离线构建包含mROS的基础镜像。
bash FoundationPose/docker/build_jetson.sh

# 进入NX容器后验证mROS和RealSense导入。
python3 -c 'import mros, pyrealsense2; print(mros.__file__)'

# Jetson 性能关联采集：同时记录运行日志、50 ms GPU 状态、200 ms
# GPU/EMC/温度/功耗和 tegrastats。Ctrl+C 后自动生成 events.csv、
# correlated_events.csv 和 correlation_summary.json。
bash FoundationPose/docker/run_jetson_ros2_telemetry.sh

# 可选采样间隔（毫秒）
FULL_INTERVAL_MS=200 FAST_INTERVAL_MS=50 \
  bash FoundationPose/docker/run_jetson_ros2_telemetry.sh

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

# 再次开一个容器进去
docker exec -it foundationpose-jetson-ros2 bash -lc '
  source /opt/ros/humble/setup.bash
  cd /workspace/yolo_foundationpose
  exec python3 realtime_foundation/run_realtime.py \
    --config realtime_foundation/config.yaml
'






终端 1：启动机器人状态发布器
cd /home/guest/eillen
source /opt/ros/humble/setup.bash

# 不要通过 -p 直接传入 URDF XML；XML 中的换行和特殊字符会导致 ROS 参数解析失败。
URDF=/home/guest/eillen/robot-description/tron2a/WFYG_TRON2A/urdf/robot.urdf
PARAMS=/tmp/robot_state_publisher_params.yaml
{
  printf '/**:\n  ros__parameters:\n    robot_description: |\n'
  sed 's/^/      /' "$URDF"
} > "$PARAMS"

ros2 run robot_state_publisher robot_state_publisher \
  --ros-args --params-file "$PARAMS"

终端 2：发布默认关节角度
# 仅在没有真实机器人关节状态时运行。若 /limx_robot_state_publisher 已发布 /joint_states，跳过本终端，避免两个节点同时发布 /joint_states。
cd /home/guest/eillen
source /opt/ros/humble/setup.bash
# 将 URDF 文件作为位置参数传入；否则节点会等待 /robot_description 话题。
ros2 run joint_state_publisher joint_state_publisher \
  /home/guest/eillen/robot-description/tron2a/WFYG_TRON2A/urdf/robot.urdf

终端 3：启动 RViz2
cd /home/guest/eillen
source /opt/ros/humble/setup.bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=99
export ROS_LOCALHOST_ONLY=0
rviz2