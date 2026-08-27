"""
degrade_isles_resolution.py
段階1の再設計（設計書 8.5.6 案E）: ISLES 2022 FLAIR を SENORA と同じ撮像幾何へ
落としてから学習し直すための前処理。

なぜ必要か:

  段階2で、学習済みモデルが SENORA 16例に対して 8例で何も出力しなかった。
  原因は断面外のボクセル間隔で、学習元 ISLES は等方 0.71 mm、SENORA は 6.8 mm。
  約10倍違う。nnU-Net は推論時に入力を学習時の間隔へ補間するため、SENORA の
  20数枚のスライスから200枚以上の断面を作ってモデルに渡していた。
  引き伸ばされた断面には撮像された情報がないので、これは前処理では埋まらない。

  そこで学習元を対象の撮像条件に合わせる。ドメインシフトのうち
  「撮像分解能」の成分を学習側で吸収し、残った差（集団・病期・装置）を
  段階2で測るという切り分けになる。設計書 5.2 の対照実験 C1 を
  評価時ではなく学習時に適用する形でもある。

何を再現するか:

  SENORA アームC 16例の実測値（`mask_characterization.csv`）:
    スライス厚 5.0 mm、間隔 6.8 mm、面内 0.72 mm が16例中12例
    つまり 1.8 mm のギャップがあり、組織の 73.5% しか撮像されていない

  厚みだけを再現して間隔を詰めると、実際には存在しない連続性を与えてしまう。
  ギャップも再現する必要がある（設計書 4.4 の指摘）。

  面内は ISLES 0.71 mm と SENORA 0.72 mm でほぼ一致するので変更しない。

やり方:

  1. RAS 正順（`as_closest_canonical`）に揃え、第3軸を頭尾方向にする
  2. 出力スライスの中心を間隔 6.8 mm ごとに置く
  3. 各中心の前後 2.5 mm（厚み 5.0 mm）に入る元スライスを平均する。
     これが励起されたスラブ内の信号平均に相当する
  4. 中心間の残り 1.8 mm は、どの出力スライスにも寄与しない。撮像されない
  5. マスクは同じ窓の中で陽性ボクセルの割合を取り、閾値で2値化する

マスクの2値化について:

  既定は多数決（0.5）。これは再標本化の慣例だが、薄い病変は消える。
  薄い病変が消えると学習元が大病変に偏り、ソース内 Dice が実態より
  良く見えてしまう。そのため QC には「どれか1ボクセルでも陽性なら陽性」
  とした場合の体積も併記し、影響が見えるようにしている。

使い方:
  python degrade_isles_resolution.py
  python degrade_isles_resolution.py --thickness 4.5 --spacing 6.12
  python degrade_isles_resolution.py --label-threshold 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# SENORA アームC の最頻値（16例中12例）
SENORA_THICKNESS_MM = 5.0
SENORA_SPACING_MM = 6.8


def slab_windows(
    n_slices: int, zoom_mm: float, thickness_mm: float, spacing_mm: float
) -> list[tuple[int, int]]:
    """
    出力スライスごとに、平均する元スライスの範囲を返す。

    元スライス i の中心は i * zoom_mm にある。出力スライス k の中心を
    spacing_mm ごとに置き、その前後 thickness_mm / 2 に中心が入る
    元スライスを集める。

    最初の中心は厚みの半分だけ内側に置く。そうしないと1枚目のスラブが
    ボリュームの外へはみ出し、実際より薄いスラブになる。
    """
    half = thickness_mm / 2.0
    extent = (n_slices - 1) * zoom_mm
    windows: list[tuple[int, int]] = []
    center = half
    while center - half <= extent + 1e-6:
        lo = int(np.ceil((center - half) / zoom_mm - 1e-6))
        hi = int(np.floor((center + half) / zoom_mm + 1e-6))
        lo = max(lo, 0)
        hi = min(hi, n_slices - 1)
        if hi >= lo:
            windows.append((lo, hi))
        center += spacing_mm
    return windows


def degraded_affine(
    affine: np.ndarray, windows: list[tuple[int, int]], spacing_mm: float, zoom_mm: float
) -> np.ndarray:
    """
    落とした後のボクセル→世界座標変換を作る。

    第3軸の方向ベクトルを間隔ぶんに伸ばし、原点を最初のスラブ中心へ移す。
    こうしないとマスクと画像で世界座標がずれる。
    """
    direction = affine[:3, 2] / zoom_mm  # 単位長のスライス方向
    new_affine = affine.copy()
    new_affine[:3, 2] = direction * spacing_mm
    first_center_mm = (windows[0][0] + windows[0][1]) / 2.0 * zoom_mm
    new_affine[:3, 3] = affine[:3, 3] + direction * first_center_mm
    return new_affine


def degrade_case(
    flair_path: Path,
    mask_path: Path,
    out_flair: Path,
    out_mask: Path,
    thickness_mm: float,
    spacing_mm: float,
    label_threshold: float,
) -> dict:
    """1症例を落として書き出し、QC 用の測定値を返す。"""
    flair_img = nib.as_closest_canonical(nib.load(flair_path))
    mask_img = nib.as_closest_canonical(nib.load(mask_path))

    if flair_img.shape[:3] != mask_img.shape[:3]:
        return {
            "subject": flair_path.parents[2].name,
            "error": f"shape mismatch {flair_img.shape[:3]} vs {mask_img.shape[:3]}",
        }

    flair = np.asarray(flair_img.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj)
    if flair.ndim > 3:
        flair = flair[..., 0]
    if mask.ndim > 3:
        mask = mask.reshape(mask.shape[:3] + (-1,)).max(axis=3)
    mask = (mask > 0).astype(np.float32)

    zoom_mm = float(flair_img.header.get_zooms()[2])
    windows = slab_windows(flair.shape[2], zoom_mm, thickness_mm, spacing_mm)
    if not windows:
        return {"subject": flair_path.parents[2].name, "error": "no slab fits"}

    thick_flair = np.stack(
        [flair[:, :, lo : hi + 1].mean(axis=2) for lo, hi in windows], axis=2
    )
    fractions = np.stack(
        [mask[:, :, lo : hi + 1].mean(axis=2) for lo, hi in windows], axis=2
    )
    thick_mask = (fractions >= label_threshold).astype(np.uint8)
    any_mask = (fractions > 0).astype(np.uint8)

    new_affine = degraded_affine(flair_img.affine, windows, spacing_mm, zoom_mm)
    voxel_ml = abs(np.linalg.det(new_affine[:3, :3])) / 1000.0
    original_ml = int(mask.sum()) * abs(np.linalg.det(flair_img.affine[:3, :3])) / 1000.0

    out_flair.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(thick_flair, new_affine), out_flair)
    nib.save(nib.Nifti1Image(thick_mask, new_affine), out_mask)

    return {
        "subject": flair_path.parents[2].name,
        "slices_before": int(flair.shape[2]),
        "slices_after": int(thick_flair.shape[2]),
        "volume_ml_before": round(original_ml, 3),
        "volume_ml_after": round(int(thick_mask.sum()) * voxel_ml, 3),
        "volume_ml_if_any": round(int(any_mask.sum()) * voxel_ml, 3),
        "emptied": bool(thick_mask.sum() == 0 and mask.sum() > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--src", type=Path, default=root / "data" / "isles2022_flair",
        help="register_isles_flair.py の出力（受理された205例）",
    )
    parser.add_argument(
        "--dst", type=Path, default=root / "data" / "isles2022_flair_thick"
    )
    parser.add_argument("--thickness", type=float, default=SENORA_THICKNESS_MM)
    parser.add_argument("--spacing", type=float, default=SENORA_SPACING_MM)
    parser.add_argument(
        "--label-threshold", type=float, default=0.5,
        help="スラブ内の陽性割合がこの値以上なら陽性。既定は多数決",
    )
    parser.add_argument("--out", type=Path, default=root / "results")
    args = parser.parse_args()

    if args.spacing < args.thickness:
        parser.error("間隔がスライス厚より小さいとスラブが重なります")

    cases = sorted(args.src.glob("sub-*/ses-*/anat/*_FLAIR.nii.gz"))
    if not cases:
        print(f"エラー: FLAIR が見つかりません: {args.src}", file=sys.stderr)
        return 1

    gap = args.spacing - args.thickness
    print(f"対象 {len(cases)} 例")
    print(f"厚み {args.thickness} mm / 間隔 {args.spacing} mm "
          f"（ギャップ {gap:.2f} mm、撮像率 {args.thickness / args.spacing:.1%}）")
    print(f"マスクの閾値 {args.label_threshold}")

    records = []
    for flair in tqdm(cases, desc="分解能を落とす"):
        mask = flair.with_name(
            flair.name.replace("_FLAIR.nii.gz", "_label-lesion_roi.nii.gz")
        )
        if not mask.exists():
            print(f"[warn] マスクなし、スキップ: {flair}", file=sys.stderr)
            continue
        relative = flair.relative_to(args.src)
        records.append(
            degrade_case(
                flair,
                mask,
                args.dst / relative,
                args.dst / relative.with_name(mask.name),
                args.thickness,
                args.spacing,
                args.label_threshold,
            )
        )

    df = pd.DataFrame(records)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "isles_resolution_degradation.csv"
    df.to_csv(csv_path, index=False)

    errors = df[df.get("error").notna()] if "error" in df else pd.DataFrame()
    ok = df[df.get("error").isna()] if "error" in df else df

    print(f"\n完了: {len(ok)} 例 → {args.dst}")
    if len(errors):
        print(f"失敗: {len(errors)} 例")
        for _, r in errors.iterrows():
            print(f"  {r['subject']}: {r['error']}")

    if len(ok):
        print(f"スライス数: {ok['slices_before'].median():.0f} "
              f"→ {ok['slices_after'].median():.0f}（中位）")
        print(f"病変体積: {ok['volume_ml_before'].median():.2f} "
              f"→ {ok['volume_ml_after'].median():.2f} mL（中位）")
        print(f"  「どれか1ボクセルでも陽性」とした場合: "
              f"{ok['volume_ml_if_any'].median():.2f} mL")
        emptied = int(ok["emptied"].sum())
        print(f"マスクが空になった症例: {emptied} / {len(ok)}")
        if emptied:
            print("  これらは学習に寄与しない。--label-threshold を下げるか、")
            print("  Dataset 作成時に除外するかを決める必要がある")
    print(f"QC を {csv_path} に出力しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
