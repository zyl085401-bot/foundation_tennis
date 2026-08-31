# YOLO 回放基准用法

脚本使用录制的 NPZ RGB 帧测试 TensorRT YOLO 分割耗时，并为后续优化生成可比较的参考结果。

## 输入

默认配置：

```text
realtime_foundation/config.yaml
```

默认录制帧目录：

```text
realtime_foundation/outputs/frame_records/frame_data
```

每个 NPZ 文件必须包含 `rgb`，格式为 `uint8`、`H x W x 3`、RGB 通道顺序。配置中的 YOLO 权重必须是 TensorRT `.engine` 文件。

## 建立基线

先停止实时 FoundationPose 程序，避免 GPU 竞争，然后在 Jetson 容器中运行：

```bash
cd /workspace/yolo_foundationpose

python3 realtime_foundation/detection/yolo_replay_benchmark.py \
  --mode baseline \
  --execution-path legacy \
  --run-name legacy_baseline
```

默认执行：

- 1 次冷启动推理，单独记录。
- 10 次预热，不计入统计。
- 100 次正式推理。

也可以明确指定参数：

```bash
python3 realtime_foundation/detection/yolo_replay_benchmark.py \
  --mode baseline \
  --config realtime_foundation/config.yaml \
  --input-dir realtime_foundation/outputs/frame_records/frame_data \
  --iterations 100 \
  --warmup 10 \
  --execution-path legacy \
  --run-name legacy_baseline
```

## 优化后对比

保持相同 TensorRT 引擎、YOLO 配置和录制帧，运行：

```bash
python3 realtime_foundation/detection/yolo_replay_benchmark.py \
  --mode compare \
  --execution-path fast \
  --profile-stages \
  --reference realtime_foundation/outputs/yolo_benchmark/legacy_baseline \
  --run-name fast_profiled_compare
```

`--reference` 可以指向基线运行目录，也可以直接指向其中的 `manifest.json`。

快速路径对比默认禁止自动回退；如果固定输入或 TensorRT binding 不符合要求，脚本会直接报错，避免把 legacy 结果误记为 fast 结果。

逐帧输出一致性由人工确认后，关闭阶段同步测量生产路径端到端耗时：

```bash
python3 realtime_foundation/detection/yolo_replay_benchmark.py \
  --mode compare \
  --execution-path fast \
  --no-profile-stages \
  --reference realtime_foundation/outputs/yolo_benchmark/legacy_baseline \
  --run-name fast_wall_compare
```

生产程序当前仍默认使用 `execution_path: legacy`。人工确认对比结果后，把 `realtime_foundation/config.yaml` 中的 `execution_path` 改为 `fast`，并保持 `profile_stages: false`。

## 后处理开关

固定形状 fast path 支持 GPU 或 CPU mask 后处理：

```yaml
yolo:
  execution_path: fast
  weights: realtime_foundation/yolo_weights/best_480x640_trt104.engine
  postprocess_backend: gpu     # gpu|cpu
```

- `gpu`：在 GPU 上生成和缩放 mask。
- `cpu`：将 TensorRT 输出复制到 CPU 后生成 mask。

建立 GPU 基线：

```bash
python3 realtime_foundation/detection/yolo_replay_benchmark.py \
  --mode baseline \
  --execution-path fast \
  --postprocess-backend gpu \
  --run-name gpu_postprocess_baseline
```

CPU 后处理要求 `execution_path: fast`，并且不会静默退回 legacy GPU 后处理。

## INT8 输入实验引擎

当前 INT8 输入方案把相机 `uint8` RGB 在 GPU 上通过整数运算直接量化为对称 `int8`，避免生成 FP32 输入张量。TensorRT 输出仍保持 FP16/FP32，现有 FP32-I/O INT8 Engine 和 FP16 Engine 均保留。

在 Jetson 容器中使用已有 ONNX、Calibration Cache 和 FP16 Engine 包装元数据构建：

```bash
cd /workspace/yolo_foundationpose

python3 realtime_foundation/detection/build_yolo_int8_engine.py \
  --weights realtime_foundation/yolo_weights/best.pt \
  --frame-dir realtime_foundation/outputs/frame_records/frame_data \
  --calibration-dir realtime_foundation/outputs/int8_calibration/yolo \
  --output realtime_foundation/yolo_weights/best_480x640_int8_io_trt104.engine \
  --fp16-template realtime_foundation/yolo_weights/best_480x640_trt104.engine \
  --cache-only \
  --int8-input \
  --int8-input-dynamic-range 1.0 \
  --fp16-layer-prefix /model.23/proto \
  --max-frames 160 \
  --workspace-gib 8
```

要求同目录中已经存在 `best.onnx` 和 `best.cache`。配置为 `precision: int8`、`io_precision: int8` 时，默认候选顺序为：

1. INT8-I/O INT8 Engine。
2. 原 FP32-I/O INT8 Engine。
3. FP16 Engine（仅当 `fallback_to_fp16: true`）。

日志中的 `[YOLO][FAST]` 应显示 `input=.../int8`，并且 `fast_path_contract` 中的 `integer_only_preprocess` 应为 `true`。INT8 输入路径当前只支持 `int8_input_dynamic_range: 1.0`。

## 输出

默认输出目录：

```text
realtime_foundation/outputs/yolo_benchmark/<run-name>
```

主要文件：

- `summary.json`：冷启动及正式推理的 median、P90、P99 等统计结果。
- `samples.csv`：每次推理的阶段耗时、候选数量、mask、bbox 和置信度信息。
- `metadata.json`：引擎 SHA256、实际 TensorRT 后端、软件版本、功耗模式和温度。
- `manifest.json`：固定输入清单及后续对比所需信息。
- `references/`：基线模式生成的参考 mask、bbox、置信度和类别。
- `visualizations/`：每个录制帧对应的 YOLO 分割可视化 JPG；图片导出不计入推理耗时。

基线模式下，图片包含 RGB 原图、半透明分割 mask、轮廓、bbox、类别、置信度和 mask 面积。对比模式下，每张图片从左到右依次为原图、基线结果、当前结果，并显示 mask IoU。

## 常用参数

```text
--mode baseline|compare   建立基线或与基线对比
--config PATH             YAML 配置文件
--input-dir PATH          录制 NPZ 帧目录
--output-root PATH        基准输出根目录
--run-name NAME           本次运行目录名
--iterations N            正式推理次数
--warmup N                预热次数
--max-frames N            最多使用多少个录制帧，0 表示全部
--reference PATH          compare 模式使用的基线目录或 manifest.json
--progress-every N        每隔多少次打印进度，0 表示不打印
--execution-path MODE     legacy 或固定形状 fast 路径
--profile-stages          fast 路径开启分阶段 CUDA 同步计时
--no-profile-stages       fast 路径只测端到端 wall time
--allow-fast-fallback     允许 fast 失败后回退 legacy；一致性对比时不要使用
--postprocess-backend MODE  gpu 或 cpu mask 后处理
--allow-engine-difference  compare 模式允许当前引擎 SHA256 与参考不同
```