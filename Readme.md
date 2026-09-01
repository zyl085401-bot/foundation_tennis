
# 使用foundationpose的时候这个命令会显示分割物体轮廓形状的效果

python run_demo.py --vis_mode contour

# 仿真
python3 realtime_foundation/tools/simulate_recorded_camera.py \
  --input realtime_foundation/outputs/error/refiner_coarse_160/frame_001172_register.npz \
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


# 减小mesh模型的大小

  python3 FoundationPose/tools/generate_textured_lod.py \
  FoundationPose/demo_data/tennis/mesh/textured_simple.obj \
  FoundationPose/demo_data/tennis/mesh/textured_simple_0.03125.obj \
  --triangle-ratio 0.03125
###################################################################################################
# 在外部x86主机连接NX上的mROS Agent。
export MROS_AGENT_URI=tcp://192.168.55.2:11315
mrostopic list


source /home/nvidia/jason/bbs/install/setup.bash
export MROS_AGENT_IP=192.168.55.2
mrosagent

# 在基础容器里启动相机发布器 启动RealSense并通过mROS发布彩色、对齐深度和相机内参话题
cd /home/nvidia/workspace/yolo_foundationpose_ball_20hz

CONTAINER_NAME=foundationpose-mros-camera \
bash FoundationPose/docker/run_container_jetson.sh \
python3 realtime_foundation/camera/depth_color_publish \
  --config realtime_foundation/config.yaml

# 开启foundation流程
export DISPLAY=:0
xhost +si:localuser:root

cd ~/workspace/yolo_foundationpose_ball_20hz

bash FoundationPose/docker/run_container_jetson.sh \
  python3 realtime_foundation/run_realtime.py \
  --config realtime_foundation/config.yaml

