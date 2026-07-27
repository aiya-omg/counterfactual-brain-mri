"""
build_atlas.py
正常脳MRIスライスの平均画像（アトラス）を作成する。

使い方:
  conda activate cfmri
  python build_atlas.py --normal_dir ./data/slices/normal --output ./atlas.png

オプション:
  --per_slice    スライス番号ごとに別々のアトラスを作る（atlas_slice000.png など）
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import re


def build_global_atlas(normal_dir: str, output: str, image_size: int = 128):
    """全正常スライスの平均像を1枚作る"""
    paths = sorted(Path(normal_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"PNGが見つかりません: {normal_dir}")

    print(f"正常スライス: {len(paths)} 枚")
    acc = np.zeros((image_size, image_size), dtype=np.float64)

    for p in tqdm(paths, desc="平均計算中"):
        img = Image.open(p).convert("L").resize((image_size, image_size), Image.LANCZOS)
        acc += np.array(img).astype(np.float64)

    atlas = (acc / len(paths)).astype(np.uint8)
    Image.fromarray(atlas).save(output)
    print(f"アトラス保存: {output}  ({image_size}x{image_size})")


def build_per_slice_atlas(normal_dir: str, output_dir: str, image_size: int = 128):
    """スライス番号ごとに別々のアトラスを作る"""
    paths = sorted(Path(normal_dir).glob("*.png"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # スライス番号でグループ化
    groups: dict[int, list] = {}
    for p in paths:
        m = re.search(r"slice(\d+)", p.name)
        if m:
            z = int(m.group(1))
            groups.setdefault(z, []).append(p)

    print(f"スライス番号の種類: {len(groups)}")
    for z, ps in tqdm(sorted(groups.items()), desc="スライス別アトラス作成"):
        acc = np.zeros((image_size, image_size), dtype=np.float64)
        for p in ps:
            img = Image.open(p).convert("L").resize((image_size, image_size), Image.LANCZOS)
            acc += np.array(img).astype(np.float64)
        atlas = (acc / len(ps)).astype(np.uint8)
        Image.fromarray(atlas).save(output_path / f"atlas_slice{z:03d}.png")

    print(f"スライス別アトラス保存: {output_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="正常脳MRIアトラス作成")
    parser.add_argument("--normal_dir",  type=str, default="./data/slices/normal")
    parser.add_argument("--output",      type=str, default="./atlas.png",
                        help="グローバルアトラスの保存先")
    parser.add_argument("--image_size",  type=int, default=128)
    parser.add_argument("--per_slice",   action="store_true",
                        help="スライス番号ごとに別々のアトラスを作る")
    parser.add_argument("--per_slice_dir", type=str, default="./atlas_per_slice")
    args = parser.parse_args()

    build_global_atlas(args.normal_dir, args.output, args.image_size)
    if args.per_slice:
        build_per_slice_atlas(args.normal_dir, args.per_slice_dir, args.image_size)
