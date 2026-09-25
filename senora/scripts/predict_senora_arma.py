"""
predict_senora_arma.py
段階2（アームA）: ISLES 2022 DWI + ADC を SENORA の DWI 幾何へ落として学習した
nnU-Net（Dataset503）を、SENORA アームA（DWI 上にマスクがある7例）へ適用する。

入力の作り方:

  SENORA の DWI は b0 / b500 / b1000 の4次元で保存されている（bval で確認）。
  ISLES 2022 の DWI チャネルは b1000 のトレース画像なので、第3ボリュームを使う。
  ADC はスキャナが出力したものをそのまま使う。

  ISLES 2022 の DWI / ADC は頭蓋除去済み（背景の約80%が0）で、plans は
  use_mask_for_norm = True。SENORA 側も頭蓋除去が必要になる。HD-BET は
  T1 / T1c / T2 / FLAIR で学習されており、b1000 や ADC には向かない。
  そこで T2 強調に近い b0 に HD-BET をかけ、得た脳マスクを b1000 と ADC に
  適用する。3つとも同じ撮像なので位置合わせは要らない。

  マスクは4次元（DWI の3ボリューム分の複製）で保存されているため、
  最大値を取って3次元に畳む。読影医2名の症例は合意マスクを使う。

使い方:
  python predict_senora_arma.py --all                       # 最終重み、fold 0
  python predict_senora_arma.py --predict --evaluate --folds 0,1,2,3,4
  python predict_senora_arma.py --evaluate --checkpoint best   # 感度分析

予測は <work>/predictions_<重み>_f<fold>、結果は senora_arma_predictions_<同>.csv に出る。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_senora_armc import dice, find_mask  # noqa: E402

DATASET_ID = "503"
TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"


def arm_a_subjects(data_root: Path) -> list[str]:
    seg = pd.read_csv(data_root / "_segmentation_report.csv")
    dwi = seg[seg["sequence"].astype(str).str.upper() == "DWI"]
    return sorted(str(s) for s in dwi["subject_id"])


def b1000_index(bval_path: Path) -> int:
    bvals = [float(v) for v in bval_path.read_text().split()]
    return int(np.argmax(bvals))


def stage(data_root: Path, work: Path, python: Path) -> int:
    b0_dir, brain_dir = work / "b0", work / "brain"
    inputs, masks = work / "inputs", work / "masks"
    for d in (b0_dir, brain_dir, inputs, masks):
        d.mkdir(parents=True, exist_ok=True)

    subjects = arm_a_subjects(data_root)
    staged = []
    for subject in subjects:
        dwi_path = data_root / subject / "dwi" / f"{subject}_dwi.nii.gz"
        adc_path = data_root / subject / "dwi" / f"{subject}_desc-ADC_dwi.nii.gz"
        mask_path = find_mask(data_root, subject)
        if not dwi_path.exists() or not adc_path.exists() or mask_path is None:
            print(f"[warn] {subject}: DWI / ADC / マスクのいずれかがない", file=sys.stderr)
            continue
        img = nib.load(dwi_path)
        dwi = np.asarray(img.dataobj, dtype=np.float32)
        b1000 = b1000_index(dwi_path.with_name(f"{subject}_dwi.bval"))
        header = img.header.copy()
        header.set_data_shape(dwi.shape[:3])
        nib.save(nib.Nifti1Image(dwi[..., 0], img.affine, header), b0_dir / f"{subject}.nii.gz")
        nib.save(nib.Nifti1Image(dwi[..., b1000], img.affine, header),
                 work / f"{subject}_b1000.nii.gz")

        adc_img = nib.load(adc_path)
        nib.save(nib.Nifti1Image(np.asarray(adc_img.dataobj, dtype=np.float32),
                                 adc_img.affine, adc_img.header),
                 work / f"{subject}_adc.nii.gz")

        mask_img = nib.load(mask_path)
        mask = np.asarray(mask_img.dataobj)
        if mask.ndim > 3:
            mask = mask.reshape(mask.shape[:3] + (-1,)).max(axis=3)
        nib.save(nib.Nifti1Image((mask > 0).astype(np.uint8), img.affine),
                 masks / f"{subject}.nii.gz")
        staged.append(subject)

    print(f"staging: {len(staged)} / {len(subjects)} 例")
    if not staged:
        return 1

    hd_bet = python.parent / "Scripts" / "hd-bet.exe"
    if not hd_bet.exists():
        hd_bet = Path("hd-bet")
    cmd = [str(hd_bet), "-i", str(b0_dir), "-o", str(brain_dir),
           "--save_bet_mask", "--no_bet_image"]
    print(f"頭蓋除去（b0）: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("エラー: HD-BET が失敗しました", file=sys.stderr)
        return result.returncode

    for subject in staged:
        bet = brain_dir / f"{subject}_bet.nii.gz"
        if not bet.exists():
            print(f"[warn] {subject}: 脳マスクがない", file=sys.stderr)
            continue
        brain = np.asarray(nib.load(bet).dataobj) > 0
        for channel, name in (("0000", "b1000"), ("0001", "adc")):
            src = nib.load(work / f"{subject}_{name}.nii.gz")
            data = np.asarray(src.dataobj, dtype=np.float32) * brain
            nib.save(nib.Nifti1Image(data, src.affine, src.header),
                     inputs / f"{subject}_{channel}.nii.gz")
    print(f"入力: {len(list(inputs.glob('*_0000.nii.gz')))} 例 → {inputs}")
    return 0


def run_tag(checkpoint: str, folds: str) -> str:
    """予測と結果のファイル名に付ける識別子。例: final_f0、final_f01234"""
    return f"{checkpoint}_f{folds.replace(',', '')}"


def predict(
    work: Path, results: Path, python: Path, folds: str, dataset: str, checkpoint: str
) -> int:
    inputs, out = work / "inputs", work / f"predictions_{run_tag(checkpoint, folds)}"
    out.mkdir(parents=True, exist_ok=True)
    if not list(inputs.glob("*_0000.nii.gz")):
        print("エラー: 入力がありません。先に --stage を実行してください", file=sys.stderr)
        return 1
    if not (results / dataset / TRAINER).is_dir():
        print(f"エラー: 学習済みモデルがありません: {results / dataset}", file=sys.stderr)
        return 1
    predict_exe = python.parent / "Scripts" / "nnUNetv2_predict.exe"
    if not predict_exe.exists():
        predict_exe = Path("nnUNetv2_predict")
    cmd = [str(predict_exe), "-i", str(inputs), "-o", str(out), "-d", DATASET_ID,
           "-c", "3d_fullres", "-f", *folds.split(","), "-chk", f"checkpoint_{checkpoint}.pth"]
    print(f"推論: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def evaluate(work: Path, data_root: Path, out_dir: Path, baseline_csv: Path, tag: str) -> int:
    preds = sorted((work / f"predictions_{tag}").glob("*.nii.gz"))
    if not preds:
        print("エラー: 予測がありません", file=sys.stderr)
        return 1

    rows = []
    for pred_path in preds:
        subject = pred_path.name.replace(".nii.gz", "")
        ref_img = nib.load(work / "masks" / f"{subject}.nii.gz")
        pred = np.asarray(nib.load(pred_path).dataobj) > 0
        ref = np.asarray(ref_img.dataobj) > 0
        tp = int(np.logical_and(pred, ref).sum())
        fp = int(np.logical_and(pred, ~ref).sum())
        fn = int(np.logical_and(~pred, ref).sum())
        voxel_ml = abs(np.linalg.det(ref_img.affine[:3, :3])) / 1000.0
        meta = json.loads((data_root / subject / "dwi" / f"{subject}_dwi.json")
                          .read_text(encoding="utf-8"))
        rows.append({
            "subject": subject,
            "dice": dice(pred, ref),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "volume_ml": round(int(ref.sum()) * voxel_ml, 3),
            "pred_volume_ml": round(int(pred.sum()) * voxel_ml, 3),
            "slice_thickness_mm": meta.get("SliceThickness"),
            "slice_spacing_mm": meta.get("SpacingBetweenSlices"),
        })

    # 読影医2名の症例では、読影医間 Dice が到達可能な上限の目安になる。
    # radA と radB がボクセル単位で同一の症例（sub-086）は独立した2名の読影ではないので、
    # 一致度として扱わず identical_raters で印を付ける
    for row in rows:
        anat = data_root / "derivatives" / "manual_lesion" / row["subject"] / "anat"
        rad_a = sorted(anat.glob("*desc-radA_label-lesion_roi.nii.gz"))
        rad_b = sorted(anat.glob("*desc-radB_label-lesion_roi.nii.gz"))
        if rad_a and rad_b:
            a = np.asarray(nib.load(rad_a[0]).dataobj)
            b = np.asarray(nib.load(rad_b[0]).dataobj)
            a = a.reshape(a.shape[:3] + (-1,)).max(axis=3) > 0 if a.ndim > 3 else a > 0
            b = b.reshape(b.shape[:3] + (-1,)).max(axis=3) > 0 if b.ndim > 3 else b > 0
            row["identical_raters"] = bool(np.array_equal(a, b))
            row["inter_rater_dice"] = float("nan") if row["identical_raters"] else dice(a, b)

    df = pd.DataFrame(rows)
    seg = pd.read_csv(data_root / "_segmentation_report.csv")
    sequence = dict(zip(seg["subject_id"].astype(str), seg["sequence"].astype(str).str.upper()))
    df["mask_sequence"] = df["subject"].map(sequence)
    scored = df.dropna(subset=["dice"])

    rng = np.random.default_rng(0)
    boot = [np.median(rng.choice(scored["dice"].to_numpy(), len(scored)))
            for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    n_dwi = int((df["mask_sequence"] == "DWI").sum())
    n_flair = int((df["mask_sequence"] == "FLAIR").sum())

    lines: list[str] = []
    out = lines.append
    if n_flair == 0:
        out("# 段階2: SENORA-MRI アームA（急性期・DWI + ADC）への適用")
        out("")
        out(f"対象 **{len(df)} 例**（DWI 上にマスクがある症例）。臨床区分は全例 acute だが、"
            "b1000 と ADC で拡散制限を示すのは一部に限られる。")
    else:
        out("# 段階2: SENORA-MRI 全マスクへの DWI + ADC モデルの適用")
        out("")
        out(f"対象 **{len(df)} 例**（DWI 上のマスク {n_dwi} 例、FLAIR 上のマスクを DWI へ写した "
            f"{n_flair} 例。後者は `stage_senora_lesions.py` の出力）。")
    out("学習元は ISLES 2022 DWI + ADC を SENORA の DWI 幾何（5.5 mm 厚 / 7.15 mm 間隔）"
        "へ落としたもの（Dataset503）。頭蓋除去は b0 に HD-BET をかけて得た脳マスク。")
    out(f"重みと fold: `{tag}`。")
    out("")
    out("| 指標 | 値 |")
    out("|---|---|")
    q1, q3 = scored["dice"].quantile([0.25, 0.75])
    out(f"| Dice 中位値 | **{scored['dice'].median():.3f}**（IQR {q1:.3f}–{q3:.3f}、"
        f"bootstrap 95% CI {lo:.3f}–{hi:.3f}） |")
    out(f"| Dice 平均 | {scored['dice'].mean():.3f} |")
    out(f"| Dice 0 の症例 | {int((scored['dice'] == 0).sum())} / {len(scored)} |")
    out(f"| 予測が空の症例 | {int((scored['pred_volume_ml'] == 0).sum())} / {len(scored)} |")
    out(f"| 適合率 中位 | {scored['precision'].median():.3f} |")
    out(f"| 再現率 中位 | {scored['recall'].median():.3f} |")
    out(f"| 病変体積 中位 | {scored['volume_ml'].median():.1f} mL |")
    out("")
    out(f"n = {len(scored)} のため信頼区間は広い。症例ごとの値を主に読む。")
    out("")

    if baseline_csv.exists():
        base = pd.read_csv(baseline_csv)
        out("## ソース内との比較")
        out("")
        out("| 集団 | 例数 | Dice 中位 | 体積 中位(mL) |")
        out("|---|---|---|---|")
        out(f"| ISLES 2022 hold-out（{baseline_csv.stem}） | {len(base)} "
            f"| {base['dice'].median():.3f} "
            f"| {base['volume_ml'].median():.1f} |")
        for label, group in (("SENORA（DWI 上のマスク）", scored[scored["mask_sequence"] == "DWI"]),
                             ("SENORA（FLAIR 上のマスク）", scored[scored["mask_sequence"] == "FLAIR"])):
            if len(group):
                out(f"| {label} | {len(group)} | {group['dice'].median():.3f} "
                    f"| {group['volume_ml'].median():.1f} |")
        out("")

    out("## 症例ごとの結果")
    out("")
    out("| 症例 | マスク | Dice | 適合率 | 再現率 | 病変(mL) | 予測(mL) | 厚/間隔(mm) | 読影医間 Dice |")
    out("|---|---|---|---|---|---|---|---|---|")
    for _, r in scored.sort_values("dice", ascending=False).iterrows():
        precision = f"{r['precision']:.3f}" if pd.notna(r["precision"]) else "—"
        inter = r.get("inter_rater_dice")
        if r.get("identical_raters") is True:
            inter = "同一マスク"
        else:
            inter = f"{inter:.3f}" if pd.notna(inter) else "—"
        out(f"| {r['subject']} | {r['mask_sequence']} | {r['dice']:.3f} | {precision} | {r['recall']:.3f} "
            f"| {r['volume_ml']:.1f} | {r['pred_volume_ml']:.1f} "
            f"| {r['slice_thickness_mm']}/{r['slice_spacing_mm']} | {inter} |")
    out("")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"senora_arma_predictions_{tag}.csv"
    df.to_csv(csv_path, index=False)
    out(f"症例ごとの測定値を `{csv_path}` に出力しました。")
    report = "\n".join(lines)
    target = out_dir / f"senora_arma_predictions_{tag}.md"
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nレポートを {target} に保存しました")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "senora_normalized")
    parser.add_argument(
        "--work", type=Path, default=Path.home() / "senora_nnunet" / "arma_senora",
        help="ASCII のみの作業領域。日本語パスでは SimpleITK が読めない",
    )
    parser.add_argument(
        "--results", type=Path, default=Path.home() / "senora_nnunet" / "nnUNet_results"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--folds", default="0", help="カンマ区切り。0,1,2,3,4 でアンサンブル")
    parser.add_argument(
        "--checkpoint", choices=("final", "best"), default="final",
        help="best は fold の検証症例で選んだ重みで、同じ症例での評価が楽観的になる。"
             "主要な値は final で出す",
    )
    parser.add_argument("--dataset", default="Dataset503_ISLES22DWIThick")
    parser.add_argument("--out", type=Path, default=root / "results" / "dataset503")
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="ソース内の症例ごとの表。既定は summarize_source_cv.py の出力",
    )
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any((args.stage, args.predict, args.evaluate, args.all)):
        parser.error("--stage / --predict / --evaluate / --all のいずれかを指定してください")
    if args.stage or args.all:
        if code := stage(args.data, args.work, args.python):
            return code
    tag = run_tag(args.checkpoint, args.folds)
    if args.predict or args.all:
        if code := predict(args.work, args.results, args.python, args.folds, args.dataset,
                           args.checkpoint):
            return code
    if args.evaluate or args.all:
        baseline = args.baseline or args.out / f"source_cv_{args.checkpoint}.csv"
        if code := evaluate(args.work, args.data, args.out, baseline, tag):
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
