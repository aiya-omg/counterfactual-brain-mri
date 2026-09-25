"""
summarize_source_cv.py
Dataset503 のソース内 hold-out 性能を、学習済みの全 fold から症例ごとに集める。

重みの選び方:

  nnU-Net の checkpoint_best.pth は、その fold の検証症例に対する EMA 疑似 Dice で
  選ばれる。同じ症例で評価すると、選択に使ったデータで測ることになり値が楽観的に
  なる。主要な値は最終重み（学習終了時に nnU-Net が書く validation/summary.json）で出す。
  best 重みの値は感度分析として並べる。

  fold 0 だけは `--val --val_best` で validation/summary.json を best 重みの結果で
  上書きしているため、最終重みの結果を summary_final_checkpoint.json に退避してある。

出力:
  results/dataset503/source_cv_final.csv   主要な値
  results/dataset503/source_cv_best.csv    fold 0 のみ（感度分析）

使い方:
  python summarize_source_cv.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_source_baseline import load_summary  # noqa: E402

TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"


def summaries(fold_dir: Path) -> dict[str, Path]:
    """その fold で使える summary.json を重みごとに返す。"""
    found: dict[str, Path] = {}
    final_copy = fold_dir / "summary_final_checkpoint.json"
    validation = fold_dir / "validation" / "summary.json"
    if final_copy.exists():
        found["final"] = final_copy
        if validation.exists():
            found["best"] = validation
    elif validation.exists():
        found["final"] = validation
    return found


def lesion_volume_ml(path: Path) -> float:
    img = nib.load(path)
    voxels = int((np.asarray(img.dataobj) > 0).sum())
    return round(voxels * abs(np.linalg.det(img.affine[:3, :3])) / 1000.0, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    home = Path.home() / "senora_nnunet"
    parser.add_argument("--dataset", default="Dataset503_ISLES22DWIThick")
    parser.add_argument("--results", type=Path, default=home / "nnUNet_results")
    parser.add_argument("--preprocessed", type=Path, default=home / "nnUNet_preprocessed")
    parser.add_argument("--out", type=Path, default=root / "results" / "dataset503")
    args = parser.parse_args()

    model = args.results / args.dataset / TRAINER
    gt = args.preprocessed / args.dataset / "gt_segmentations"
    tables: dict[str, list[pd.DataFrame]] = {"final": [], "best": []}
    for fold_dir in sorted(model.glob("fold_*")):
        # 学習途中の fold は checkpoint_final.pth がないので数えない
        if not (fold_dir / "checkpoint_final.pth").exists():
            continue
        fold = int(fold_dir.name.split("_")[1])
        for weights, path in summaries(fold_dir).items():
            df, _ = load_summary(path)
            df["fold"] = fold
            tables[weights].append(df)

    if not tables["final"]:
        print(f"エラー: 学習済みの fold がありません: {model}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for weights, parts in tables.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["volume_ml"] = [lesion_volume_ml(gt / f"{c}.nii.gz") for c in df["case"]]
        path = args.out / f"source_cv_{weights}.csv"
        df.to_csv(path, index=False)
        q1, q3 = df["dice"].quantile([0.25, 0.75])
        folds = ",".join(str(f) for f in sorted(df["fold"].unique()))
        print(f"[{weights}] fold {folds}: {len(df)} 例、Dice 中位 {df['dice'].median():.3f}"
              f"（IQR {q1:.3f}–{q3:.3f}）、平均 {df['dice'].mean():.3f}、"
              f"Dice 0 は {int((df['dice'] == 0).sum())} 例 → {path}")
        for low, high in ((0, 1), (1, 5), (5, 20), (20, np.inf)):
            group = df[(df["volume_ml"] >= low) & (df["volume_ml"] < high)]
            if len(group):
                label = f"{low}–{high} mL" if np.isfinite(high) else f"{low} mL 以上"
                print(f"    {label}: {len(group)} 例、Dice 中位 {group['dice'].median():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
