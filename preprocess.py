"""
preprocess.py
BraTS NIfTI データ → PNG スライス変換スクリプト

BraTSデータ構造（例）:
  BraTS2021_Training_Data/
    BraTS2021_00000/
      BraTS2021_00000_t1.nii.gz    # T1強調
      BraTS2021_00000_t1ce.nii.gz  # T1造影
      BraTS2021_00000_t2.nii.gz    # T2強調
      BraTS2021_00000_flair.nii.gz # FLAIR
      BraTS2021_00000_seg.nii.gz   # セグメンテーションマスク

使い方:
  python preprocess.py \
    --data_dir /path/to/BraTS2021_Training_Data \
    --output_dir ./data/slices \
    --modality t2 \
    --slice_axis 2 \
    --min_tumor_ratio 0.01
"""

import argparse
import os
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """スライスを0-255にノーマライズ（パーセンタイルクリッピング）"""
    p1, p99 = np.percentile(slice_2d, [1, 99])
    slice_clipped = np.clip(slice_2d, p1, p99)
    if p99 - p1 > 0:
        slice_norm = (slice_clipped - p1) / (p99 - p1)
    else:
        slice_norm = np.zeros_like(slice_clipped)
    return (slice_norm * 255).astype(np.uint8)


def has_enough_tumor(seg_slice: np.ndarray, min_ratio: float = 0.01) -> bool:
    """腫瘍領域が一定割合以上あるスライスかどうか判定"""
    tumor_pixels = np.sum(seg_slice > 0)
    total_pixels = seg_slice.size
    return (tumor_pixels / total_pixels) >= min_ratio


def extract_slices(
    data_dir: str,
    output_dir: str,
    modality: str = "t2",
    slice_axis: int = 2,
    image_size: int = 256,
    min_tumor_ratio: float = 0.01,
    max_patients: int = None,
    skip_patients: int = 0,
):
    """
    NIfTIファイルから2Dスライスを抽出してPNGとして保存

    出力ディレクトリ構成:
      output_dir/
        tumor/     # 腫瘍ありスライス
        normal/    # 腫瘍なしスライス（同一患者の正常領域）
        masks/     # 対応するセグメンテーションマスク
    """
    output_path = Path(output_dir)
    (output_path / "tumor").mkdir(parents=True, exist_ok=True)
    (output_path / "normal").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if max_patients:
        patient_dirs = patient_dirs[skip_patients:skip_patients + max_patients]
    elif skip_patients:
        patient_dirs = patient_dirs[skip_patients:]

    print(f"処理対象患者数: {len(patient_dirs)}")
    tumor_count = 0
    normal_count = 0

    for patient_dir in tqdm(patient_dirs, desc="患者データ処理中"):
        patient_id = patient_dir.name

        # ファイルパス（BraTS2023形式: BraTS-GLI-00000-000-t2f.nii.gz）
        img_path = patient_dir / f"{patient_id}-{modality}.nii.gz"
        seg_path = patient_dir / f"{patient_id}-seg.nii.gz"

        # BraTS2021形式にもフォールバック（BraTS2021_00000_t2.nii.gz）
        if not img_path.exists():
            img_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
        if not seg_path.exists():
            seg_path = patient_dir / f"{patient_id}_seg.nii.gz"

        if not img_path.exists() or not seg_path.exists():
            print(f"  スキップ: {patient_id} (ファイル不足)")
            continue

        # NIfTIロード
        img_nii = nib.load(str(img_path))
        seg_nii = nib.load(str(seg_path))
        img_data = img_nii.get_fdata()  # shape: (H, W, D)
        seg_data = seg_nii.get_fdata()

        num_slices = img_data.shape[slice_axis]

        for idx in range(num_slices):
            # スライス取得
            if slice_axis == 0:
                img_slice = img_data[idx, :, :]
                seg_slice = seg_data[idx, :, :]
            elif slice_axis == 1:
                img_slice = img_data[:, idx, :]
                seg_slice = seg_data[:, idx, :]
            else:
                img_slice = img_data[:, :, idx]
                seg_slice = seg_data[:, :, idx]

            # 空のスライスをスキップ
            if img_slice.max() == 0:
                continue

            # ノーマライズ & リサイズ
            img_norm = normalize_slice(img_slice)
            img_pil = Image.fromarray(img_norm).convert("RGB")
            img_pil = img_pil.resize((image_size, image_size), Image.LANCZOS)

            # マスクリサイズ
            seg_pil = Image.fromarray(
                (seg_slice > 0).astype(np.uint8) * 255
            ).resize((image_size, image_size), Image.NEAREST)

            if has_enough_tumor(seg_slice, min_tumor_ratio):
                # 腫瘍あり
                fname = f"{patient_id}_slice{idx:03d}.png"
                img_pil.save(output_path / "tumor" / fname)
                seg_pil.save(output_path / "masks" / fname)
                tumor_count += 1
            else:
                # 腫瘍なし（正常スライス）
                fname = f"{patient_id}_slice{idx:03d}.png"
                img_pil.save(output_path / "normal" / fname)
                normal_count += 1

    print(f"\n✅ 完了!")
    print(f"  腫瘍ありスライス: {tumor_count}")
    print(f"  正常スライス:     {normal_count}")
    print(f"  保存先: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS NIfTI → PNG スライス変換")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="BraTSデータのルートディレクトリ")
    parser.add_argument("--output_dir", type=str, default="./data/slices",
                        help="出力先ディレクトリ")
    parser.add_argument("--modality", type=str, default="t2f",
                        choices=["t1", "t1ce", "t2", "flair", "t2f", "t1c", "t1n", "t2w"],
                        help="使用するMRIモダリティ（BraTS2023: t2f/t1c/t1n/t2w, BraTS2021: t1/t1ce/t2/flair）")
    parser.add_argument("--slice_axis", type=int, default=2,
                        help="スライス軸（0=矢状面, 1=冠状面, 2=軸位面）")
    parser.add_argument("--image_size", type=int, default=256,
                        help="出力画像サイズ（正方形）")
    parser.add_argument("--min_tumor_ratio", type=float, default=0.01,
                        help="腫瘍スライスと判定する最小腫瘍面積率")
    parser.add_argument("--max_patients",  type=int, default=None,
                        help="処理する最大患者数")
    parser.add_argument("--skip_patients", type=int, default=0,
                        help="スキップする患者数（例: 100 で101人目から開始）")
    args = parser.parse_args()

    extract_slices(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        modality=args.modality,
        slice_axis=args.slice_axis,
        image_size=args.image_size,
        min_tumor_ratio=args.min_tumor_ratio,
        max_patients=args.max_patients,
        skip_patients=args.skip_patients,
    )
