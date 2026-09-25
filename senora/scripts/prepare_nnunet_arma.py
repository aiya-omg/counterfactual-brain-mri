"""
prepare_nnunet_arma.py
アームA（急性期・DWI + ADC）の学習データを作る。ISLES 2022 の DWI と ADC を
SENORA アームAの撮像幾何へ落とし、nnU-Net v2 の Dataset 形式に並べる。

なぜアームAを主軸に戻すか（設計書 8.6）:

  アームCの16例は慢性期11例・不明5例で、急性期は0例だった。慢性期の大きな
  梗塞は空洞化して FLAIR で中心が低信号になる（マスク内の30〜70%）。
  急性期しか含まない ISLES で学習したモデルはこれを病変として見たことがなく、
  213 mL の梗塞に 0.04 mL しか出さなかった。分解能を揃えた案Eでも SENORA の
  Dice は床から動かず、残っている差は病期そのものと判断した。

  アームAの7例は全例急性期で、病変はほぼ全体が DWI 高信号。ISLES 2022 の
  本来の課題（DWI + ADC）と一致するので、ドメインシフトを測る対象として
  課題のずれが混ざらない。

何を再現するか:

  SENORA アームA 7例の実測値（DWI の JSON）:
    スライス厚 5.5 mm、間隔 7.15 mm が7例中5例（ほかは 5.0/6.5 と 5.5/7.7）
    面内 1.35 mm（1例 1.43 mm）、1.5T Siemens MAGNETOM ESSENZA

  ISLES 2022 の DWI は 250 例中 193 例が 2 mm 等方、53 例が 4.8 mm 厚。
  断面外だけを案Eと同じ方法（スラブ平均 + ギャップ）で落とす。
  面内は ISLES 2.0 mm の方が SENORA 1.35 mm より粗く、上げても情報は
  増えないので変更しない。推論時に nnU-Net が SENORA 側を学習時の間隔へ
  下げる。4.8 mm 厚の症例は元スライスが1〜2枚しか窓に入らないが、
  同じ処理を通して QC に記録する。

  チャネルは DWI（b1000 トレース）と ADC。ADC の単位は ISLES の症例間でも
  揃っていない（10^-3 と 10^-6 mm²/s が混在）が、nnU-Net は症例ごとに
  z-score 正規化するので尺度の違いは吸収される。

出力:
  <nnUNet_raw>/Dataset503_ISLES22DWIThick/
    imagesTr/  ISLES_XXXX_0000.nii.gz   # DWI
               ISLES_XXXX_0001.nii.gz   # ADC
    labelsTr/  ISLES_XXXX.nii.gz
    dataset.json, case_to_subject.csv
  senora/results/isles_dwi_degradation.csv   # QC

使い方:
  python prepare_nnunet_arma.py
  python prepare_nnunet_arma.py --thickness 5.0 --spacing 6.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade_isles_resolution import degraded_affine, slab_windows  # noqa: E402

# SENORA アームA の最頻値（7例中5例）
SENORA_DWI_THICKNESS_MM = 5.5
SENORA_DWI_SPACING_MM = 7.15


def load_canonical(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3] + (-1,)).max(axis=3)
    return data, img


def slab_mean(volume: np.ndarray, windows: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([volume[:, :, lo : hi + 1].mean(axis=2) for lo, hi in windows], axis=2)


def roundtrip_dice(
    mask: np.ndarray, thick_mask: np.ndarray, windows: list[tuple[int, int]]
) -> float:
    """
    落としたマスクを元の格子へ戻し、元のマスクとの Dice を返す。

    元スライスは、そのスライスを平均に含めた出力スライスの値を受け取る。
    どのスラブにも入らないギャップのスライスは、中心が最も近い出力スライスの値を
    受け取る（撮像されない組織を、隣のスライスで補って読む読影と同じ扱い）。
    体積の保持率は打ち消し合う誤差（薄い病変の消失と、元が厚い症例での膨らみ）を
    区別できないため、位置まで含めて一致を見る。
    """
    centers = np.array([(lo + hi) / 2.0 for lo, hi in windows])
    back = np.zeros(mask.shape, dtype=bool)
    for z in range(mask.shape[2]):
        inside = [k for k, (lo, hi) in enumerate(windows) if lo <= z <= hi]
        k = inside[0] if inside else int(np.argmin(np.abs(centers - z)))
        back[:, :, z] = thick_mask[:, :, k] > 0
    ref = mask > 0
    total = int(ref.sum()) + int(back.sum())
    return 2.0 * int((ref & back).sum()) / total if total else float("nan")


def degrade_case(
    dwi_path: Path,
    adc_path: Path,
    mask_path: Path,
    out_images: list[Path],
    out_label: Path,
    thickness_mm: float,
    spacing_mm: float,
    label_threshold: float,
    write: bool = True,
) -> dict:
    subject = dwi_path.parents[2].name
    dwi, dwi_img = load_canonical(dwi_path)
    adc, _ = load_canonical(adc_path)
    mask, _ = load_canonical(mask_path)
    if not (dwi.shape == adc.shape == mask.shape):
        return {"subject": subject,
                "error": f"shape mismatch {dwi.shape} {adc.shape} {mask.shape}"}
    mask = (mask > 0).astype(np.float32)

    zooms = dwi_img.header.get_zooms()[:3]
    zoom_mm = float(zooms[2])
    windows = slab_windows(dwi.shape[2], zoom_mm, thickness_mm, spacing_mm)
    if not windows:
        return {"subject": subject, "error": "no slab fits"}

    fractions = slab_mean(mask, windows)
    thick_mask = (fractions >= label_threshold).astype(np.uint8)
    any_mask = (fractions > 0).astype(np.uint8)

    new_affine = degraded_affine(dwi_img.affine, windows, spacing_mm, zoom_mm)
    voxel_ml = abs(np.linalg.det(new_affine[:3, :3])) / 1000.0
    original_ml = int(mask.sum()) * abs(np.linalg.det(dwi_img.affine[:3, :3])) / 1000.0

    if write:
        out_label.parent.mkdir(parents=True, exist_ok=True)
        for volume, target in zip((dwi, adc), out_images):
            target.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(slab_mean(volume, windows), new_affine), target)
        nib.save(nib.Nifti1Image(thick_mask, new_affine), out_label)

    return {
        "subject": subject,
        "in_plane_mm": round(float(zooms[0]), 2),
        "through_plane_mm_before": round(zoom_mm, 2),
        "slices_before": int(dwi.shape[2]),
        "slices_after": len(windows),
        "volume_ml_before": round(original_ml, 3),
        "volume_ml_after": round(int(thick_mask.sum()) * voxel_ml, 3),
        "volume_ml_if_any": round(int(any_mask.sum()) * voxel_ml, 3),
        "roundtrip_dice": round(roundtrip_dice(mask, thick_mask, windows), 4),
        "emptied": bool(thick_mask.sum() == 0 and mask.sum() > 0),
        "empty_before": bool(mask.sum() == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--src", type=Path, default=root / "data" / "isles2022" / "ISLES-2022"
    )
    parser.add_argument(
        "--raw", type=Path, default=Path.home() / "senora_nnunet" / "nnUNet_raw",
        help="nnUNet_raw 相当（Windows では日本語パスを避ける）",
    )
    parser.add_argument("--dataset-id", type=int, default=503)
    parser.add_argument("--dataset-name", type=str, default="ISLES22DWIThick")
    parser.add_argument("--thickness", type=float, default=SENORA_DWI_THICKNESS_MM)
    parser.add_argument("--spacing", type=float, default=SENORA_DWI_SPACING_MM)
    parser.add_argument(
        "--label-threshold", type=float, default=0.5,
        help="スラブ内の陽性割合がこの値以上なら陽性。既定は多数決",
    )
    parser.add_argument("--out", type=Path, default=root / "results")
    parser.add_argument(
        "--qc-only", action="store_true",
        help="画像を書かずに QC だけ出す。学習に使っている Dataset を書き換えないため",
    )
    args = parser.parse_args()

    if args.spacing < args.thickness:
        parser.error("間隔がスライス厚より小さいとスラブが重なります")

    dwis = sorted(args.src.glob("sub-*/ses-*/dwi/*_dwi.nii.gz"))
    if not dwis:
        print(f"エラー: DWI が見つかりません: {args.src}", file=sys.stderr)
        return 1

    ds_dir = args.raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images = ds_dir / "imagesTr"
    labels = ds_dir / "labelsTr"

    gap = args.spacing - args.thickness
    print(f"対象 {len(dwis)} 例 → {ds_dir}")
    print(f"厚み {args.thickness} mm / 間隔 {args.spacing} mm "
          f"（ギャップ {gap:.2f} mm、撮像率 {args.thickness / args.spacing:.1%}）")

    records = []
    training = []
    mapping = []
    for dwi in tqdm(dwis, desc="分解能を落とす"):
        subject, session = dwi.parents[2].name, dwi.parents[1].name
        adc = dwi.with_name(dwi.name.replace("_dwi.nii.gz", "_adc.nii.gz"))
        mask = (args.src / "derivatives" / subject / session
                / f"{subject}_{session}_msk.nii.gz")
        if not adc.exists() or not mask.exists():
            records.append({"subject": subject, "error": "ADC かマスクがない"})
            continue

        case_id = f"ISLES_{len(training) + 1:04d}"
        record = degrade_case(
            dwi, adc, mask,
            [images / f"{case_id}_0000.nii.gz", images / f"{case_id}_0001.nii.gz"],
            labels / f"{case_id}.nii.gz",
            args.thickness, args.spacing, args.label_threshold,
            write=not args.qc_only,
        )
        records.append(record)
        if "error" in record:
            continue
        # 落として空になった症例は「病変なし」と教えることになるので除く。
        # 副作用として小病変の学習例が系統的に減る（設計書 8.6.5）
        if record["emptied"] or record["empty_before"]:
            if not args.qc_only:
                for f in (images / f"{case_id}_0000.nii.gz",
                          images / f"{case_id}_0001.nii.gz",
                          labels / f"{case_id}.nii.gz"):
                    f.unlink(missing_ok=True)
            continue
        training.append(case_id)
        mapping.append({"case_id": case_id, "subject": subject})

    suffix = "" if args.label_threshold == 0.5 else f"_t{args.label_threshold}"
    if args.qc_only:
        report_qc(pd.DataFrame(records), args.out / f"isles_dwi_degradation{suffix}.csv",
                  len(training), len(dwis))
        return 0

    dataset = {
        "channel_names": {"0": "DWI", "1": "ADC"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(training),
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": (
            "ISLES 2022 DWI + ADC resampled through-plane to the SENORA Arm A DWI "
            f"geometry ({args.thickness} mm slices, {args.spacing} mm spacing)."
        ),
        "reference": "https://doi.org/10.5281/zenodo.7153326",
        "licence": "CC-BY-4.0",
        "tensorImageSize": "3D",
    }
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (ds_dir / "case_to_subject.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case_id", "subject"])
        writer.writeheader()
        writer.writerows(mapping)

    report_qc(pd.DataFrame(records), args.out / f"isles_dwi_degradation{suffix}.csv",
              len(training), len(dwis))
    print("次:")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0")
    return 0


def report_qc(df: pd.DataFrame, csv_path: Path, n_training: int, n_total: int) -> None:
    """
    QC を書き出して要約する。

    体積は出力スライス間隔（7.15 mm）で数える。等間隔の断面から体積を推定する
    Cavalieri 法と同じで、各スライスがその間隔ぶんの組織を代表する。スラブ厚
    （5.5 mm）で数えるとギャップの組織を0とみなすことになり、系統的に過小になる。
    ただし元が 4.8 mm 厚の症例では窓に元スライスが1〜2枚しか入らず、2枚のとき
    多数決が「どちらか陽性なら陽性」になるので膨らむ。元の厚さで分けて報告する。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    ok = df[df["error"].isna()] if "error" in df else df
    print(f"\n学習に使う症例: {n_training} / {n_total}")
    if "error" in df and df["error"].notna().any():
        for _, r in df[df["error"].notna()].iterrows():
            print(f"  失敗 {r['subject']}: {r['error']}")
    if len(ok):
        print(f"元から病変なし: {int(ok['empty_before'].sum())} 例、"
              f"落として空になった: {int(ok['emptied'].sum())} 例（ともに除外）")
        print(f"スライス数: {ok['slices_before'].median():.0f} → "
              f"{ok['slices_after'].median():.0f}（中位）")
        kept = ok[~ok["emptied"].astype(bool) & ~ok["empty_before"].astype(bool)]
        for thick, group in kept.groupby(kept["through_plane_mm_before"] >= 4.0):
            label = "元が 4 mm 以上の厚さ" if thick else "元が 2 mm 前後"
            retained = group["volume_ml_after"].sum() / group["volume_ml_before"].sum()
            print(f"  {label}（{len(group)} 例）: 総体積の保持率 {retained:.1%}、"
                  f"元の格子に戻した Dice 中位 {group['roundtrip_dice'].median():.3f}"
                  f"（最小 {group['roundtrip_dice'].min():.3f}）")
    print(f"QC を {csv_path} に出力しました")


if __name__ == "__main__":
    sys.exit(main())
