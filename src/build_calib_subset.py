"""
校准数据集生成脚本（从 train/val 抽样复制图片与标签）。

快速用法：
    python -m src.build_calib_subset \
        --data-root data/26_1_3/adjust \
        --output data/26_1_3/adjust/calib \
        --count 200 \
        --seed 42 \
        --overwrite

主要参数：
- --data-root   数据集根目录，需包含 images/train, images/val, labels/train, labels/val
- --output      输出目录（会创建 images/ 与 labels/ 子目录）
- --count       抽样图片数量，默认 200（若少于总量则取全部）
- --seed        随机种子，保证可重复
- --overwrite   允许覆盖已存在的输出目录

脚本会给输出文件名加上 train__ / val__ 前缀以避免同名冲突，缺失标签的图片会被跳过。
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create calibration subset")
    parser.add_argument(
        "--data-root",
        default="data/26_1_3/adjust",
        help="Dataset root containing images/train, images/val, labels/train, labels/val",
    )
    parser.add_argument(
        "--output",
        default="data/26_1_3/adjust/calib",
        help="Output directory for calibration subset",
    )
    parser.add_argument(
        "--count", type=int, default=200, help="Number of images to sample",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deleting existing output directory before writing",
    )
    return parser.parse_args(argv)


def collect_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for split in ("train", "val"):
        split_dir = root / "images" / split
        if not split_dir.exists():
            continue
        for ext in IMG_EXTS:
            images.extend(split_dir.glob(f"*{ext}"))
    return sorted(images)


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {out_dir} exists. Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)


def sample_and_copy(images: List[Path], root: Path, out_dir: Path, count: int) -> int:
    if not images:
        raise RuntimeError("No images found under images/train or images/val")

    take = min(count, len(images))
    chosen = images[:]
    random.shuffle(chosen)
    chosen = chosen[:take]

    copied = 0
    for img_path in chosen:
        split = img_path.parent.name  # train or val
        dest_name = f"{split}__{img_path.name}"
        dest_img = out_dir / "images" / dest_name

        label_path = root / "labels" / split / (img_path.stem + ".txt")
        dest_label = out_dir / "labels" / f"{split}__{img_path.stem}.txt"

        if not label_path.exists():
            print(f"[WARN] Missing label for {img_path}; skipping")
            continue

        shutil.copy2(img_path, dest_img)
        shutil.copy2(label_path, dest_label)
        copied += 1

    return copied


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    out_dir = Path(args.output)

    random.seed(args.seed)
    images = collect_images(data_root)
    print(f"[INFO] Found {len(images)} images across train/val")

    prepare_output(out_dir, args.overwrite)
    copied = sample_and_copy(images, data_root, out_dir, args.count)
    print(f"[DONE] Copied {copied} image/label pairs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
