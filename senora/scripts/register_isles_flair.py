"""
register_isles_flair.py
段階1: ISLES 2022 の病変マスクを DWI 空間から FLAIR 空間へ剛体登録する。

背景:
  公開データはネイティブ空間のまま。マスクは DWI 空間にある。
  アームCは FLAIR 単一チャネルで学習し、SENORA の FLAIR 空間マスクで評価するため、
  学習前にマスクを FLAIR へ写す必要がある（設計書 4.4 (2)）。

手順:
  1. FLAIR を DWI へ剛体登録（Mattes MI）し、変換を得る
  2. その逆変換でマスクを FLAIR 空間へ写す（最近傍）
  3. 登録品質を round-trip Dice で QC
  4. 採用した FLAIR + 登録済みマスクを出力ツリーにコピー

使い方:
  python register_isles_flair.py
  python register_isles_flair.py --limit 5   # 動作確認
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def read_image(path: Path) -> sitk.Image:
    """NIfTI を読む。日本語パスでも動くよう、失敗時は一時コピー経由。"""
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "img.nii.gz"
            shutil.copy2(path, tmp)
            return sitk.ReadImage(str(tmp))


def write_label(image: sitk.Image, out_path: Path) -> None:
    """ラベル画像を保存。日本語パス対策で常に一時ファイル経由。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    binary = sitk.Cast(image > 0, sitk.sitkUInt8)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "label.nii.gz"
        sitk.WriteImage(binary, str(tmp))
        shutil.copy2(tmp, out_path)


def write_transform(tx: sitk.Transform, out_path: Path) -> None:
    """変換を保存。日本語パス対策で常に一時ファイル経由。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "xfm.txt"
        sitk.WriteTransform(tx, str(tmp))
        shutil.copy2(tmp, out_path)


def find_case_files(raw_root: Path, deriv_root: Path, subject: str) -> dict[str, Path] | None:
    """1症例の FLAIR / DWI / mask パスを探す。レイアウト差を吸収する。"""
    ses_dirs = sorted((raw_root / subject).glob("ses-*"))
    if not ses_dirs:
        ses_dirs = [raw_root / subject]

    for ses in ses_dirs:
        flair_cands = list(ses.rglob("*FLAIR.nii.gz")) + list(ses.rglob("*flair.nii.gz"))
        dwi_cands = [
            p for p in list(ses.rglob("*dwi.nii.gz")) + list(ses.rglob("*_dwi.nii.gz"))
            if "adc" not in p.name.lower()
        ]
        der_ses = (
            deriv_root / subject / ses.name
            if ses.name.startswith("ses-")
            else deriv_root / subject
        )
        mask_cands = list(der_ses.rglob("*msk.nii.gz")) + list(der_ses.rglob("*mask*.nii.gz"))
        if not mask_cands:
            mask_cands = list((deriv_root / subject).rglob("*msk.nii.gz"))

        if flair_cands and dwi_cands and mask_cands:
            return {
                "flair": flair_cands[0],
                "dwi": dwi_cands[0],
                "mask": mask_cands[0],
                "session": ses.name if ses.name.startswith("ses-") else "ses-0001",
            }
    return None


def register_flair_to_dwi(flair: sitk.Image, dwi: sitk.Image) -> sitk.Transform:
    """FLAIR → DWI 剛体登録。戻り値は FLAIR を DWI 空間へ写す変換。"""
    flair_f = sitk.Normalize(sitk.Cast(flair, sitk.sitkFloat32))
    dwi_f = sitk.Normalize(sitk.Cast(dwi, sitk.sitkFloat32))

    init = sitk.CenteredTransformInitializer(
        dwi_f,
        flair_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    return reg.Execute(dwi_f, flair_f)


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom else 1.0


def process_case(
    paths: dict[str, Path],
    out_root: Path,
    subject: str,
    dice_threshold: float,
) -> dict:
    flair_sitk = read_image(paths["flair"])
    dwi_sitk = read_image(paths["dwi"])
    mask_sitk = read_image(paths["mask"])

    same_size = mask_sitk.GetSize() == dwi_sitk.GetSize()
    if same_size:
        mask_sitk.CopyInformation(dwi_sitk)

    tx = register_flair_to_dwi(flair_sitk, dwi_sitk)
    inv = tx.GetInverse()

    mask_on_flair = sitk.Resample(
        mask_sitk,
        flair_sitk,
        inv,
        sitk.sitkNearestNeighbor,
        0.0,
        mask_sitk.GetPixelID(),
    )
    back = sitk.Resample(
        mask_on_flair,
        dwi_sitk,
        tx,
        sitk.sitkNearestNeighbor,
        0.0,
        mask_sitk.GetPixelID(),
    )
    qc_dice = dice_binary(
        sitk.GetArrayFromImage(mask_sitk),
        sitk.GetArrayFromImage(back),
    )

    accepted = same_size and qc_dice >= dice_threshold
    status = "ok" if accepted else ("geom_mismatch" if not same_size else "low_dice")

    ses = paths["session"]
    case_out = out_root / subject / ses
    if accepted:
        flair_out = case_out / "anat" / f"{subject}_{ses}_FLAIR.nii.gz"
        mask_out = case_out / "anat" / f"{subject}_{ses}_label-lesion_roi.nii.gz"
        flair_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["flair"], flair_out)
        write_label(mask_on_flair, mask_out)
        write_transform(tx, case_out / f"{subject}_{ses}_from-FLAIR_to-DWI_xfm.txt")

    return {
        "subject": subject,
        "session": ses,
        "status": status,
        "mask_dwi_same_size": same_size,
        "roundtrip_dice": round(qc_dice, 4),
        "mask_voxels": int((sitk.GetArrayFromImage(mask_sitk) > 0).sum()),
        "flair_shape": "x".join(str(s) for s in flair_sitk.GetSize()),
        "dwi_shape": "x".join(str(s) for s in dwi_sitk.GetSize()),
    }


def discover_subjects(bids_root: Path) -> tuple[Path, Path, list[str]]:
    """画像ルート / derivatives / 被験者一覧を返す。"""
    candidates = [
        bids_root,
        bids_root / "ISLES-2022",
        bids_root / "isles-2022",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        raw = root / "rawdata"
        deriv = root / "derivatives"
        if raw.is_dir() and deriv.is_dir():
            subjects = sorted(p.name for p in raw.glob("sub-*") if p.is_dir())
            if subjects:
                return raw, deriv, subjects
        subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
        if subjects and deriv.is_dir():
            return root, deriv, subjects
    raise FileNotFoundError(f"ISLES 2022 の被験者ディレクトリが見つかりません: {bids_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "isles2022")
    parser.add_argument(
        "--dest",
        type=Path,
        default=root / "data" / "isles2022_flair",
        help="登録済み FLAIR+マスクの出力先",
    )
    parser.add_argument("--out", type=Path, default=root / "results")
    parser.add_argument("--limit", type=int, default=0, help="先頭 N 例だけ（動作確認用）")
    parser.add_argument("--dice-threshold", type=float, default=0.90)
    args = parser.parse_args()

    if not args.data.exists():
        print(
            f"エラー: {args.data} がありません。先に fetch_isles2022.py を実行してください",
            file=sys.stderr,
        )
        return 1

    raw, deriv, subjects = discover_subjects(args.data)
    if args.limit:
        subjects = subjects[: args.limit]

    print(f"rawdata: {raw}")
    print(f"derivatives: {deriv}")
    print(f"対象: {len(subjects)} 例")

    rows: list[dict] = []
    args.dest.mkdir(parents=True, exist_ok=True)

    for subject in tqdm(subjects, desc="登録"):
        paths = find_case_files(raw, deriv, subject)
        if paths is None:
            rows.append({
                "subject": subject,
                "session": "",
                "status": "missing_files",
                "mask_dwi_same_size": False,
                "roundtrip_dice": 0.0,
                "mask_voxels": 0,
                "flair_shape": "",
                "dwi_shape": "",
            })
            continue
        try:
            row = process_case(
                paths, args.dest, subject, dice_threshold=args.dice_threshold
            )
        except Exception as exc:
            row = {
                "subject": subject,
                "session": paths.get("session", ""),
                "status": f"error:{type(exc).__name__}",
                "mask_dwi_same_size": False,
                "roundtrip_dice": 0.0,
                "mask_voxels": 0,
                "flair_shape": "",
                "dwi_shape": "",
                "error": str(exc),
            }
        rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "isles_flair_registration.csv"
    fields = [
        "subject", "session", "status", "mask_dwi_same_size",
        "roundtrip_dice", "mask_voxels", "flair_shape", "dwi_shape",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    dice_vals = [r["roundtrip_dice"] for r in rows if r["status"] == "ok"]
    report = {
        "n_total": len(rows),
        "n_accepted": ok,
        "n_rejected": len(rows) - ok,
        "acceptance_rate": round(ok / len(rows), 4) if rows else 0,
        "roundtrip_dice_median": float(np.median(dice_vals)) if dice_vals else None,
        "roundtrip_dice_min": float(np.min(dice_vals)) if dice_vals else None,
        "dest": str(args.dest),
        "csv": str(csv_path),
        "status_counts": {
            k: sum(1 for r in rows if r["status"] == k)
            for k in sorted({r["status"] for r in rows})
        },
    }
    summary_path = args.out / "isles_flair_registration.json"
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n# 登録結果")
    print(f"- 受理: {ok} / {len(rows)}")
    print(f"- round-trip Dice 中央値: {report['roundtrip_dice_median']}")
    print(f"- 内訳: {report['status_counts']}")
    print(f"- 出力: {args.dest}")
    print(f"- CSV: {csv_path}")
    print("次: python prepare_nnunet_armc.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
