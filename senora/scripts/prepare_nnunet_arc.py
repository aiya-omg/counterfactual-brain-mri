"""
prepare_nnunet_arc.py
ARC（慢性期）の FLAIR を、Dataset502（ISLES の FLAIR）と同じ手順で nnU-Net の Dataset504 にする。
定義は docs/prereg-chronic-model.md。

1症例ごとに:
  1. RAS 正順で頭尾方向の間隔が 2 mm を超える FLAIR は除外する
  2. T2w を FLAIR に剛体登録し（register_isles_flair.py と同じ設定）、T2w 上のマスクを FLAIR の格子へ写す
  3. FLAIR に HD-BET をかけ、脳の外を 0 にする
  4. 登録の QC（体積比、脳内の割合）で外れた症例を除外する
  5. degrade_isles_resolution.py と同じ方法で 5.0 mm 厚・6.8 mm 間隔に落とす。空になった症例は除外する

入力は fetch_arc.py の出力（~/senora_nnunet/arc/raw/<subject>/）。元のファイルは書き換えない。
SimpleITK と nnU-Net は日本語パスを開けないので、作業はすべて ~/senora_nnunet の下で行う。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade_isles_resolution import SENORA_SPACING_MM, SENORA_THICKNESS_MM, degrade_case  # noqa: E402
from register_isles_flair import register_flair_to_dwi  # noqa: E402

MAX_SLICE_MM = 2.0
VOLUME_RATIO = (0.8, 1.25)
MIN_IN_BRAIN = 0.5
DATASET_ID = 504
DATASET_NAME = "ARCFLAIRThick"


def to_3d(image: sitk.Image) -> sitk.Image:
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        size[3] = 0
        return sitk.Extract(image, size, [0, 0, 0, 0])
    return image


def map_mask(case: Path) -> tuple[sitk.Image, sitk.Euler3DTransform, float]:
    """T2w を FLAIR に登録し、マスクを FLAIR の格子へ写す。マスクは 0〜1 の割合で返す。"""
    flair = to_3d(sitk.ReadImage(str(case / "flair.nii.gz")))
    t2w = to_3d(sitk.ReadImage(str(case / "t2w.nii.gz")))
    mask = sitk.Cast(to_3d(sitk.ReadImage(str(case / "mask_t2w.nii.gz"))), sitk.sitkFloat32)
    if mask.GetSize() == t2w.GetSize():
        mask.CopyInformation(t2w)
    tx = register_flair_to_dwi(t2w, flair)
    mapped = sitk.Resample(sitk.Cast(mask > 0, sitk.sitkFloat32), flair, tx,
                           sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    t2_ml = float((sitk.GetArrayFromImage(mask) > 0).sum() * np.prod(mask.GetSpacing()) / 1000.0)
    return mapped, tx, t2_ml


def brain_masks(cases: list[Path]) -> None:
    """HD-BET を同じプロセスで1例ずつ。hd-bet コマンドは子プロセスが多く、学習中に走らせると落ちる。"""
    todo = [c for c in cases if not (c / "flair_brain.nii.gz").exists()]
    if not todo:
        return
    import torch
    from HD_BET.hd_bet_prediction import get_hdbet_predictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    predictor = get_hdbet_predictor(use_tta=True, device=torch.device("cuda"))
    io = SimpleITKIO()
    for index, case in enumerate(todo, start=1):
        image, props = io.read_images([str(case / "flair_input.nii.gz")])
        seg = predictor.predict_single_npy_array(image, props, None, None, False)
        io.write_seg(seg.astype(np.uint8), str(case / "flair_brain.nii.gz"), props)
        print(f"  HD-BET {index}/{len(todo)} {case.name}", flush=True)
    del predictor
    torch.cuda.empty_cache()


def process(raw: Path, work: Path) -> pd.DataFrame:
    rows = []
    cases = []
    for src in sorted(p for p in raw.iterdir() if p.is_dir()):
        subject = src.name
        case = work / subject
        case.mkdir(parents=True, exist_ok=True)
        row = {"subject": subject}
        flair_img = nib.load(src / "flair.nii.gz")
        canonical = nib.as_closest_canonical(flair_img)
        row["slice_mm"] = float(canonical.header.get_zooms()[2])
        row["in_plane_mm"] = float(min(canonical.header.get_zooms()[:2]))
        if row["slice_mm"] > MAX_SLICE_MM:
            row["excluded"] = "slice_too_thick"
            rows.append(row)
            continue

        flair = np.asarray(flair_img.dataobj, dtype=np.float32)
        if flair.ndim > 3:
            flair = flair[..., 0]
        nib.save(nib.Nifti1Image(flair, flair_img.affine, flair_img.header), case / "flair_input.nii.gz")

        mapped_path = case / "mask_flair_frac.nii.gz"
        if not mapped_path.exists():
            mapped, tx, t2_ml = map_mask(src)
            sitk.WriteImage(mapped, str(mapped_path))
            sitk.WriteTransform(tx, str(case / "t2w_to_flair.tfm"))
            (case / "t2_mask_ml.txt").write_text(f"{t2_ml}", encoding="utf-8")
        row["t2_mask_ml"] = float((case / "t2_mask_ml.txt").read_text(encoding="utf-8"))
        cases.append(case)
        rows.append(row)

    print(f"HD-BET: {len(cases)} 例", flush=True)
    brain_masks(cases)

    by_subject = {r["subject"]: r for r in rows}
    for case in cases:
        row = by_subject[case.name]
        flair_img = nib.load(case / "flair_input.nii.gz")
        flair = np.asarray(flair_img.dataobj, dtype=np.float32)
        brain = np.asarray(nib.load(case / "flair_brain.nii.gz").dataobj) > 0
        frac = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(str(case / "mask_flair_frac.nii.gz"))),
                            (2, 1, 0))
        mask = frac >= 0.5
        voxel_ml = abs(np.linalg.det(flair_img.affine[:3, :3])) / 1000.0
        row["flair_mask_ml"] = float(mask.sum() * voxel_ml)
        row["volume_ratio"] = row["flair_mask_ml"] / row["t2_mask_ml"] if row["t2_mask_ml"] else np.nan
        row["in_brain_frac"] = float((mask & brain).sum() / mask.sum()) if mask.any() else np.nan
        row["brain_ml"] = float(brain.sum() * voxel_ml)
        if not (VOLUME_RATIO[0] <= row["volume_ratio"] <= VOLUME_RATIO[1]):
            row["excluded"] = "volume_ratio"
            continue
        if not row["in_brain_frac"] >= MIN_IN_BRAIN:
            row["excluded"] = "outside_brain"
            continue
        nib.save(nib.Nifti1Image(flair * brain, flair_img.affine), case / "flair_stripped.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), flair_img.affine), case / "mask_flair.nii.gz")
        thick = work.parent / "thick" / case.name
        result = degrade_case(case / "flair_stripped.nii.gz", case / "mask_flair.nii.gz",
                              thick / "flair.nii.gz", thick / "mask.nii.gz",
                              SENORA_THICKNESS_MM, SENORA_SPACING_MM, 0.5)
        result.pop("subject", None)
        row.update(result)
        if result.get("error"):
            row["excluded"] = "degrade_error"
        elif result.get("emptied"):
            row["excluded"] = "emptied"
    return pd.DataFrame(rows)


def write_dataset(qc: pd.DataFrame, thick_root: Path, raw_out: Path) -> Path:
    import shutil

    ds = raw_out / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    images, labels = ds / "imagesTr", ds / "labelsTr"
    if ds.exists():
        raise SystemExit(f"{ds} はもうある。上書きしない")
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    used = qc[qc["excluded"].isna()].sort_values("subject")
    mapping = []
    for index, subject in enumerate(used["subject"], start=1):
        case_id = f"ARC_{index:04d}"
        shutil.copy2(thick_root / subject / "flair.nii.gz", images / f"{case_id}_0000.nii.gz")
        shutil.copy2(thick_root / subject / "mask.nii.gz", labels / f"{case_id}.nii.gz")
        mapping.append({"case_id": case_id, "subject": subject})
    dataset = {
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(mapping),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": (
            "Aphasia Recovery Cohort (OpenNeuro ds004884) chronic stroke FLAIR, skull-stripped with HD-BET, "
            "resampled to the SENORA-MRI FLAIR geometry (5.0 mm slabs every 6.8 mm) as in Dataset502. "
            "Lesion masks drawn on T2w were rigidly registered to FLAIR (closest session within 90 days)."
        ),
        "reference": "https://doi.org/10.18112/openneuro.ds004884.v1.0.1",
        "licence": "CC0",
        "tensorImageSize": "3D",
    }
    (ds / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    pd.DataFrame(mapping).to_csv(ds / "case_to_subject.csv", index=False)
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home() / "senora_nnunet"
    parser.add_argument("--arc", type=Path, default=base / "arc")
    parser.add_argument("--raw-out", type=Path, default=base / "nnUNet_raw")
    args = parser.parse_args()

    qc = process(args.arc / "raw", args.arc / "work")
    if "excluded" not in qc:
        qc["excluded"] = np.nan
    qc.to_csv(args.arc / "arc_qc.csv", index=False)
    print(qc["excluded"].fillna("used").value_counts().to_string())
    used = qc[qc["excluded"].isna()]
    print(f"病変体積（劣化前）中央値 {used['flair_mask_ml'].median():.1f} mL、"
          f"体積比 中央値 {used['volume_ratio'].median():.2f}、脳内の割合 中央値 {used['in_brain_frac'].median():.2f}")
    ds = write_dataset(qc, args.arc / "thick", args.raw_out)
    print(f"Dataset → {ds}（{len(used)} 例）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
