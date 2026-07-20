# 使用foundationpose的时候这个命令会显示分割物体轮廓形状的效果

python run_demo.py --vis_mode contour


python3 -m pip install pyrealsense2
python3 -m pip install -U "ultralytics>=8.3.0" --no-deps

python3 simulate_recorded_camera.py   --input outputs/error/frame_000858_register.npz   --config config.yaml 