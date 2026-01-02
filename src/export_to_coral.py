"""
YOLO 模型一键导出到 Edge TPU（先转 INT8 TFLite，再用 edgetpu_compiler 编译）。

流程：
1) 从 model/trained 选择训练好的权重；
2) 调用 ultralytics 导出 INT8 TFLite（需提供 data.yaml 做校准）；
3) 复制到 model/output 并用 edgetpu_compiler 生成 *_edgetpu.tflite。

用法：
    python -m src.export_to_coral \
        --model yolov8n.26.1.2.pt \
        --data data/26_1_3/adjust/data.yaml  \
        --imgsz 320 \
        --overwrite

主要参数：
- --model       model/trained 下的文件名或相对路径（必填）
- --data        数据集 YAML（INT8 校准必填）
- --imgsz       导出输入尺寸，默认 320（方形）
- --calib-dir   校准图片目录，默认取 data.yaml 同级的 calib/images
- --overwrite   允许覆盖 output 目录已有文件

依赖：已安装 ultralytics，且 PATH 中可找到 edgetpu_compiler。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "ultralytics is required. Install with: pip install ultralytics"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
TRAINED_DIR = ROOT / "model" / "trained"
ORIGIN_DIR = ROOT / "model" / "origin"
OUTPUT_DIR = ROOT / "model" / "output"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO to Coral Edge TPU")
    parser.add_argument(
        "--model",
        required=True,
        help="Model filename under model/trained or a relative path within that directory",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset YAML path for INT8 calibration (e.g., data.yaml)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=320, help="Image size for export (square)",
    )
    parser.add_argument(
        "--calib-dir",
        default=None,
        help="Directory containing calibration images (defaults to <data.yaml dir>/calib/images)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they exist",
    )
    return parser.parse_args(argv)


def validate_path(path: Path, desc: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def resolve_model_path(name: str) -> Path:
    candidate = (TRAINED_DIR / name).resolve()
    return validate_path(candidate, "Model")


def check_compiler() -> str:
    compiler = shutil.which("edgetpu_compiler")
    if not compiler:
        raise RuntimeError(
            "edgetpu_compiler not found in PATH. Install with 'sudo apt-get install edgetpu-compiler'"
        )
    return compiler


def locate_saved_model(tflite_path: Path) -> Path:
    # Ultralytics places tflite inside the saved_model directory; fall back to sibling search.
    parent = tflite_path.parent
    if (parent / "saved_model.pb").exists():
        return parent
    alt = parent.parent / f"{parent.name}_saved_model"
    if (alt / "saved_model.pb").exists():
        return alt
    raise FileNotFoundError(
        f"SavedModel not found near {tflite_path}. Checked {parent} and {alt}"
    )


def export_to_tflite(model_path: Path, data_yaml: str, imgsz: int) -> Path:
    validate_path(Path(data_yaml).resolve(), "data.yaml")
    print(f"[INFO] Loading model: {model_path}")
    yolo_model = YOLO(str(model_path))
    print(f"[INFO] Exporting to INT8 TFLite with imgsz={imgsz}, data={data_yaml}")
    result = yolo_model.export(
        format="tflite",
        int8=True,
        imgsz=imgsz,
        data=data_yaml,
        nms=False,  # keep raw outputs for custom postprocessing if needed
        device="cpu",
    )
    tflite_path = Path(result) if isinstance(result, str) else Path(result[0])
    if not tflite_path.exists():
        raise RuntimeError(f"Export did not create TFLite file: {tflite_path}")
    print(f"[OK] INT8 TFLite saved at: {tflite_path}")
    return tflite_path


def to_full_int8(saved_model_dir: Path, calib_dir: Path, imgsz: int, overwrite: bool) -> Path:
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
    images = sorted(calib_dir.glob("*"))
    if not images:
        raise RuntimeError(f"No images found for calibration in {calib_dir}")
    print(f"[INFO] Calibration images: {len(images)} from {calib_dir}")

    def rep_ds() -> Iterable[list[np.ndarray]]:
        for p in images:
            img = Image.open(p).convert("RGB").resize((imgsz, imgsz))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # NHWC float32
            yield [arr[None, ...]]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_bytes = converter.convert()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{saved_model_dir.name.replace('_saved_model', '')}_fullint8.tflite"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite to replace.")
    out_path.write_bytes(tflite_bytes)
    print(f"[OK] Full INT8 TFLite saved at: {out_path}")
    return out_path


def compile_edgetpu(compiler: str, tflite_path: Path, overwrite: bool) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # If file already in OUTPUT_DIR, avoid redundant copy.
    target_tflite = OUTPUT_DIR / tflite_path.name
    if tflite_path.resolve() != target_tflite.resolve():
        if target_tflite.exists() and not overwrite:
            raise FileExistsError(f"{target_tflite} exists. Use --overwrite to replace.")
        shutil.copy2(tflite_path, target_tflite)
        print(f"[INFO] Copied INT8 TFLite to: {target_tflite}")
    else:
        print(f"[INFO] Using INT8 TFLite in output dir: {target_tflite}")

    print("[INFO] Running edgetpu_compiler ...")
    cmd = [compiler, "-o", str(OUTPUT_DIR), str(target_tflite)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("[COMPILER STDOUT]\n" + proc.stdout)
    print("[COMPILER STDERR]\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"edgetpu_compiler failed with code {proc.returncode}")

    compiled_path = OUTPUT_DIR / f"{target_tflite.stem}_edgetpu.tflite"
    if not compiled_path.exists():
        raise RuntimeError(
            f"Expected compiled model not found: {compiled_path}. Check compiler output above."
        )
    print(f"[OK] Edge TPU model: {compiled_path}")
    return compiled_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    model_path = resolve_model_path(args.model)
    compiler = check_compiler()
    try:
        # Step 1: Ultralytics export (gives INT8 TFLite + SavedModel)
        tflite_path = export_to_tflite(model_path, args.data, args.imgsz)

        # Step 2: Force full INT8 (in/out uint8) via TF converter using calibration images
        saved_model_dir = locate_saved_model(tflite_path)
        calib_dir = Path(args.calib_dir) if args.calib_dir else Path(args.data).resolve().parent / "calib" / "images"
        full_int8 = to_full_int8(saved_model_dir, calib_dir, args.imgsz, args.overwrite)

        # Step 3: Edge TPU compile
        compiled = compile_edgetpu(compiler, full_int8, args.overwrite)
    except Exception as exc:  # pragma: no cover - CLI flow
        print(f"[ERROR] {exc}")
        return 1

    print("[DONE] Conversion complete.")
    print(f"Full INT8: {full_int8}")
    print(f"Edge TPU: {compiled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
