"""
extract_normal_slices.py
BraTS2023 全患者からセグメンテーションマスクが空のスライスを正常スライスとして抽出する。

使い方:
  python extract_normal_slices.py \
    --brats_dir ./data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --output_dir ./data/slices/normal_all \
    --modality t2f \
    --min_brain_ratio 0.05
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(arr[arr > 0], [1, 99]) if arr.max() > 0 else (0, 1)
    arr = np.clip(arr, p1, p99)
    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats_dir",       type=str,
                        default="./data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    parser.add_argument("--output_dir",      type=str, default="./data/slices/normal_all")
    parser.add_argument("--modality",        type=str, default="t2f")
    parser.add_argument("--image_size",      type=int, default=128)
    parser.add_argument("--min_brain_ratio", type=float, default=0.05,
                        help="スライス内の脳ピクセル割合の最低値（空白スライス除外）")
    parser.add_argument("--max_per_patient", type=int, default=50,
                        help="1患者あたりの最大スライス数（-1で無制限）")
    args = parser.parse_args()

    brats_dir  = Path(args.brats_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in brats_dir.iterdir() if d.is_dir()])
    print(f"患者数: {len(patient_dirs)}")

    total_saved = 0

    for patient_dir in tqdm(patient_dirs, desc="患者処理中"):
        pid = patient_dir.name

        # NIfTI ファイル検索
        img_path = patient_dir / f"{pid}-{args.modality}.nii.gz"
        seg_path = patient_dir / f"{pid}-seg.nii.gz"
        if not img_path.exists():
            img_path = patient_dir / f"{pid}_{args.modality}.nii.gz"
            seg_path = patient_dir / f"{pid}_seg.nii.gz"
        if not img_path.exists() or not seg_path.exists():
            continue

        try:
            img_vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
            seg_vol = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        except Exception as e:
            print(f"  スキップ ({pid}): {e}")
            continue

        H, W, D = img_vol.shape
        n_pixels = H * W

        saved_this_patient = 0
        for z in range(D):
            if args.max_per_patient >= 0 and saved_this_patient >= args.max_per_patient:
                break

            seg_sl = seg_vol[:, :, z]
            img_sl = img_vol[:, :, z]

            # 腫瘍なし かつ 脳ピクセルが十分あるスライスのみ
            if seg_sl.max() > 0:
                continue
            brain_ratio = (img_sl > 0).sum() / n_pixels
            if brain_ratio < args.min_brain_ratio:
                continue

            # 保存
            norm = normalize_slice(img_sl)
            pil  = Image.fromarray((norm * 255).astype(np.uint8)).convert("L")
            pil  = pil.resize((args.image_size, args.image_size), Image.LANCZOS)

            fname = output_dir / f"{pid}_slice{z:03d}.png"
            pil.save(str(fname))
            saved_this_patient += 1
            total_saved += 1

    print(f"\n✅ 完了: {total_saved} 枚の正常スライスを {output_dir} に保存しました")


if __name__ == "__main__":
    main()
