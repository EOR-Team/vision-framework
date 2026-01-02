# Vision Framework

Made by _**@n1ghts4kura**_ - 26.1.3

## 配置说明

### 硬件配置

- **CPU**: AMD Ryzen 5 5600
- **内存**: 16GB
- **显卡**: NVIDIA GeForce RTX 2060
- **操作系统**: Ubuntu 24.04.3 LTS

### 软件配置

- **Python版本**: 3.10.0
- **Python依赖**: `requirements/n1ghts4kura_26.1.3.txt`
- **CUDA版本**: 12.9
- **cuDNN版本**: 9.17.1

---

## 使用指南

### 环境准备
- 创建/激活虚拟环境后安装依赖：`pip install -r requirements/n1ghts4kura_26.1.3.txt`
- 确保 `edgetpu_compiler` 已安装且在 PATH。

### 1) 生成校准数据集（可复现）
使用训练/验证集中抽样图片：
```bash
./venv/bin/python -m src.build_calib_subset \
  --data-root data/26_1_3/adjust \               # adjust/ 为训练模型的数据集根目录 
  --output data/26_1_3/adjust/calib \            # calib/ 为输出校准集目录 用于后续量化
  --count 200 \                                  # 抽样图片数量 用于校准
  --seed 42 \                                    # 随机种子
  --overwrite                                    # 覆盖已存在目录
```
输出：`calib/images` 与 `calib/labels`，供后续量化。

### 2) 导出 + 全 INT8 + Edge TPU 编译（自动化）
```bash
./venv/bin/python -m src.export_to_coral \
  --model yolov8n.26.1.2.pt \                    # 正常训练完成的 YOLOv8 模型权重文件
  --data data/26_1_3/adjust/data.yaml \          # 校准数据集配置文件
  --imgsz 320 \                                  # 导出尺寸 **重要** 不要修改 推理时也使用该尺寸
  --overwrite                                    # 覆盖已存在输出目录
```
说明：
- `--calib-dir` 可选，默认使用 `data.yaml` 同级的 `calib/images`。
- 脚本流程：Ultralytics 导出 INT8 TFLite → 使用校准集强制生成全 INT8（in/out uint8）TFLite → `edgetpu_compiler -o model/output` 编译。
输出：
- `model/output/<name>_fullint8.tflite`
- `model/output/<name>_fullint8_edgetpu.tflite`（及编译日志 `.log`）

### 3) 树莓派 + Coral 使用提示
- 将 `*_edgetpu.tflite` 拷贝到树莓派，确保已安装 `libedgetpu` 与 `tflite-runtime/pycoral`。
- 推理时按 uint8 输入、匹配导出尺寸（默认 320）。
