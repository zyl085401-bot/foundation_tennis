# Realtime FoundationPose tools

This directory contains offline calibration, benchmarking, simulation, audit, and telemetry utilities. None of these scripts are imported by the production realtime loop.

Run commands from the repository root.

| Tool | Purpose |
| --- | --- |
| `simulate_recorded_camera.py` | Replay recorded RGB-D/mask frames through FoundationPose. |
| `collect_candidate_predictability.py` | Evaluate candidate-pose predictability over recorded frames. |
| `capture_int8_calibration.py` | Capture real network inputs for INT8 calibration. |
| `compare_trt_engine_outputs.py` | Compare FoundationPose TensorRT engine outputs. |
| `build_yolo_int8_engine.py` | Build a YOLO INT8 TensorRT engine from ONNX and calibration cache. |
| `yolo_replay_benchmark.py` | Benchmark YOLO against recorded RGB frames. |
| `jetson_telemetry.py` | Record Jetson GPU, EMC, thermal, and power telemetry. |
| `analyze_jetson_telemetry.py` | Correlate telemetry with FoundationPose runtime events. |
| `audit_distillation_dataset.py` | Validate captured distillation datasets. |

`realsense_config.yml` is the standalone RealSense preview configuration used by `camera/realsense_reader.py`.

`yolo_segment_config.yml` and `rgb/006.png` are standalone YOLO validation resources used by `detection/yolo_segmenter.py`.

Interactive/manual tests are kept under `realtime_foundation/tests/`; automated unit tests use the `test_*.py` naming convention.
