"""
predict_senora_armc.py
段階2: ISLES 2022 FLAIR で学習した nnU-Net を SENORA-MRI のアームC
（FLAIR 上に病変マスクがある16例）へ適用し、症例ごとの Dice を測る。

前処理を揃える理由:

  学習元の ISLES 2022 は主催者が頭蓋除去したうえで公開している。
  nnU-Net の plans は use_mask_for_norm = True で計画されており、
  これは背景が 0 で埋まっていることを前提に、非ゼロ領域だけで
  z-score 正規化するという意味である。SENORA は頭蓋除去されていないため、
  そのまま入力すると頭皮と頭蓋を含めた統計で正規化され、
  学習時とは違う輝度分布がモデルに渡る。
  ドメインシフトを測るはずの実験に、前処理の不一致という別の要因が
  混ざってしまうので、推論前に頭蓋除去が必須になる。

  設計書 8章は SynthStrip を指定しているが、FreeSurfer は Windows では
  WSL を要する。ここでは HD-BET を使う。nnU-Net と同じ研究室が出している
  脳抽出モデルで、FLAIR を含む複数コントラストで学習されており、
  pip で入る。設計書からの逸脱として記録する。

Windows での注意:

  SimpleITK と nnU-Net は日本語を含むパスを開けない。リポジトリが
  `AI医学` 配下にあるため、作業ファイルはすべて ASCII のみの
  作業領域（既定では ~/senora_nnunet/armc_senora）に置いて処理する。

使い方:
  # 1. 頭蓋除去して staging する
  python predict_senora_armc.py --stage
  # 2. 学習済みモデルで推論する
  python predict_senora_armc.py --predict
  # 3. マスクと突き合わせて評価する
  python predict_senora_armc.py --evaluate
  # まとめて
  python predict_senora_armc.py --all
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# 学習済みモデルの識別子
DATASET = "Dataset501_ISLES22FLAIR"
TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"


def arm_c_subjects(data_root: Path) -> list[str]:
    """
    FLAIR 上にマスクが描かれた症例を `_segmentation_report.csv` から取る。

    段階0.6 で確認したとおり、読影医は発症状態に応じて参照系列を
    選んでいる。アームCの対象は sequence 列が FLAIR の症例のみ。
    """
    report = data_root / "_segmentation_report.csv"
    if not report.exists():
        raise SystemExit(f"{report} がありません")
    seg = pd.read_csv(report)
    flair = seg[seg["sequence"].astype(str).str.upper() == "FLAIR"]
    return sorted(str(s) for s in flair["subject_id"])


def find_mask(data_root: Path, subject: str) -> Path | None:
    """読影医2名の症例は合意マスクを優先する。characterize_masks.py と同じ規則。"""
    anat = data_root / "derivatives" / "manual_lesion" / subject / "anat"
    if not anat.is_dir():
        return None
    files = sorted(anat.glob("*_label-lesion_roi.nii.gz"))
    if not files:
        return None
    for f in files:
        if "desc-consensus" in f.name:
            return f
    plain = [f for f in files if "desc-" not in f.name]
    return plain[0] if plain else files[0]


def stage(data_root: Path, work: Path, python: Path) -> int:
    """
    FLAIR とマスクを ASCII パスへ複製し、FLAIR に頭蓋除去をかける。

    HD-BET は入力ディレクトリを一括で処理できるので、症例ごとに
    呼ばずにまとめて渡す。GPU が無い環境では -device cpu になるが、
    16例なら現実的な時間で終わる。
    """
    raw = work / "flair_raw"
    stripped = work / "flair_stripped"
    masks = work / "masks"
    for d in (raw, stripped, masks):
        d.mkdir(parents=True, exist_ok=True)

    subjects = arm_c_subjects(data_root)
    staged = []
    for subject in subjects:
        flair = data_root / subject / "anat" / f"{subject}_FLAIR.nii.gz"
        mask = find_mask(data_root, subject)
        if not flair.exists():
            print(f"[warn] {subject}: FLAIR がありません", file=sys.stderr)
            continue
        if mask is None:
            print(f"[warn] {subject}: マスクがありません", file=sys.stderr)
            continue
        # nnU-Net の入力規約に合わせ、チャネル番号 _0000 を付ける
        shutil.copy2(flair, raw / f"{subject}_0000.nii.gz")
        shutil.copy2(mask, masks / f"{subject}.nii.gz")
        staged.append(subject)

    print(f"staging: {len(staged)} / {len(subjects)} 例")
    if not staged:
        return 1

    hd_bet = python.parent / "Scripts" / "hd-bet.exe"
    if not hd_bet.exists():
        hd_bet = Path("hd-bet")
    cmd = [str(hd_bet), "-i", str(raw), "-o", str(stripped)]
    print(f"頭蓋除去: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("エラー: HD-BET が失敗しました", file=sys.stderr)
        return result.returncode

    produced = sorted(stripped.glob("*_0000.nii.gz"))
    print(f"頭蓋除去済み: {len(produced)} 本 → {stripped}")
    return 0


def predict(work: Path, results: Path, python: Path, folds: str) -> int:
    """学習済みモデルで推論する。best 重みを使う。"""
    stripped = work / "flair_stripped"
    out = work / "predictions"
    out.mkdir(parents=True, exist_ok=True)

    inputs = sorted(stripped.glob("*_0000.nii.gz"))
    if not inputs:
        print(f"エラー: {stripped} に入力がありません。先に --stage を実行してください",
              file=sys.stderr)
        return 1

    model = results / DATASET / TRAINER
    if not model.is_dir():
        print(f"エラー: 学習済みモデルがありません: {model}", file=sys.stderr)
        return 1

    predict_exe = python.parent / "Scripts" / "nnUNetv2_predict.exe"
    if not predict_exe.exists():
        predict_exe = Path("nnUNetv2_predict")
    cmd = [
        str(predict_exe),
        "-i", str(stripped),
        "-o", str(out),
        "-d", "501",
        "-c", "3d_fullres",
        "-f", *folds.split(","),
        "-chk", "checkpoint_best.pth",
    ]
    print(f"推論: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode
    print(f"予測: {len(sorted(out.glob('*.nii.gz')))} 本 → {out}")
    return 0


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    inter = int(np.logical_and(pred, ref).sum())
    total = int(pred.sum()) + int(ref.sum())
    if total == 0:
        return float("nan")
    return 2.0 * inter / total


def training_spacing(results: Path) -> list[float] | None:
    """
    学習時に nnU-Net が再標本化の目標としたボクセル間隔を返す。

    SENORA との差を測る基準になる。plans.json の 3d_fullres 構成に入っている。
    """
    plans = results / DATASET / TRAINER / "plans.json"
    if not plans.exists():
        return None
    import json

    data = json.loads(plans.read_text(encoding="utf-8"))
    spacing = data.get("configurations", {}).get("3d_fullres", {}).get("spacing")
    return list(spacing) if spacing else None


def evaluate(work: Path, out_dir: Path, baseline_csv: Path, results: Path) -> int:
    """
    予測とマスクを突き合わせ、症例ごとの Dice を出す。

    マスクは4次元で保存されている症例があるため、全ボリュームの
    最大値を取って3次元に畳む（段階0.6 で確認した仕様）。

    あわせて撮像幾何を記録する。学習元と対象で断面外のボクセル間隔が
    大きく違う場合、nnU-Net は対象を学習時の間隔へ補間してから推論するため、
    実際には撮像されていない断面が引き伸ばして作られる。
    ドメインシフトの内訳を語るにはこの差を数字で持っておく必要がある。
    """
    preds = sorted((work / "predictions").glob("*.nii.gz"))
    if not preds:
        print("エラー: 予測がありません。先に --predict を実行してください",
              file=sys.stderr)
        return 1

    rows = []
    for pred_path in preds:
        subject = pred_path.name.replace(".nii.gz", "")
        mask_path = work / "masks" / f"{subject}.nii.gz"
        if not mask_path.exists():
            print(f"[warn] {subject}: マスクがありません", file=sys.stderr)
            continue

        pred_img = nib.load(pred_path)
        ref_img = nib.load(mask_path)
        pred = np.asarray(pred_img.dataobj) > 0
        ref = np.asarray(ref_img.dataobj)
        if ref.ndim > 3:
            ref = ref.reshape(ref.shape[:3] + (-1,)).max(axis=3)
        ref = ref > 0

        if pred.shape != ref.shape:
            print(f"[warn] {subject}: 形が違う pred={pred.shape} ref={ref.shape}",
                  file=sys.stderr)
            rows.append({"subject": subject, "shape_mismatch": True})
            continue

        tp = int(np.logical_and(pred, ref).sum())
        fp = int(np.logical_and(pred, ~ref).sum())
        fn = int(np.logical_and(~pred, ref).sum())
        voxel_ml = abs(np.linalg.det(ref_img.affine[:3, :3])) / 1000.0
        zooms = ref_img.header.get_zooms()[:3]
        rows.append(
            {
                "subject": subject,
                "dice": dice(pred, ref),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "volume_ml": round(int(ref.sum()) * voxel_ml, 3),
                "pred_volume_ml": round(int(pred.sum()) * voxel_ml, 3),
                "in_plane_mm": round(float(min(zooms)), 2),
                "through_plane_mm": round(float(max(zooms)), 2),
                "slices": int(min(ref_img.shape[:3])),
            }
        )

    df = pd.DataFrame(rows)
    scored = df.dropna(subset=["dice"]) if "dice" in df else pd.DataFrame()

    lines: list[str] = []
    out = lines.append
    out("# 段階2: SENORA-MRI アームCへの適用")
    out("")
    out(f"対象 **{len(df)} 例**（FLAIR 上にマスクがある症例）。")
    out("学習元は ISLES 2022 FLAIR、頭蓋除去は HD-BET。")
    out("")

    if scored.empty:
        out("> 評価できた症例がありません。")
    else:
        q1, q3 = scored["dice"].quantile([0.25, 0.75])
        out("| 指標 | 値 |")
        out("|---|---|")
        out(f"| Dice 中位値 | **{scored['dice'].median():.3f}**（IQR {q1:.3f}–{q3:.3f}） |")
        out(f"| Dice 平均 | {scored['dice'].mean():.3f} |")
        out(f"| Dice 0 の症例 | {int((scored['dice'] == 0).sum())} / {len(scored)} |")
        out(f"| 適合率 中位 | {scored['precision'].median():.3f} |")
        out(f"| 再現率 中位 | {scored['recall'].median():.3f} |")
        out(f"| 病変体積 中位 | {scored['volume_ml'].median():.1f} mL |")
        out("")

        if baseline_csv.exists():
            base = pd.read_csv(baseline_csv)
            out("## ソース内との比較")
            out("")
            out("| 集団 | 例数 | Dice 中位 | 体積 中位(mL) |")
            out("|---|---|---|---|")
            out(f"| ISLES 2022 hold-out | {len(base)} | {base['dice'].median():.3f} "
                f"| {base['volume_ml'].median():.1f} |")
            out(f"| SENORA アームC | {len(scored)} | {scored['dice'].median():.3f} "
                f"| {scored['volume_ml'].median():.1f} |")
            out("")
            out("体積分布が違う場合、Dice の差はドメインシフトだけを表さない。")
            out("段階1で示したとおり Dice は体積に強く依存する（ρ = 0.76）ため、")
            out("比較の際は体積を層別する必要がある。")
            out("")

        empty = scored[scored["pred_volume_ml"] == 0]
        out("## 何が起きているか")
        out("")
        out(f"予測が完全に空だった症例が **{len(empty)} / {len(scored)}**。")
        out("性能が下がったのではなく、モデルが何も出していない。")
        out("低下幅を測る対象になっていないので、原因を先に切り分ける。")
        out("")

        target = training_spacing(results)
        out("### 撮像分解能の差")
        out("")
        if target:
            out(f"学習時の再標本化目標: {target[0]:.2f} × {target[1]:.2f} × {target[2]:.2f} mm"
                "（ISLES 2022 FLAIR はほぼ等方）")
        out(f"SENORA アームC: 面内 中位 {scored['in_plane_mm'].median():.2f} mm、"
            f"断面外 中位 **{scored['through_plane_mm'].median():.2f} mm**、"
            f"スライス数 中位 {scored['slices'].median():.0f}")
        out("")
        if target:
            ratio = scored["through_plane_mm"].median() / max(target)
            out(f"断面外の間隔が学習時の約 **{ratio:.0f} 倍**ある。")
        out("")
        out("nnU-Net は推論時に入力を学習時の間隔へ補間する。つまり SENORA の")
        out("20数枚のスライスから200枚以上の断面を作って 128³ のパッチに渡している。")
        out("引き伸ばされた断面には実際に撮像された情報がないため、")
        out("モデルには学習時と似ても似つかない立体が入力される。")
        out("")
        out("**これは前処理で吸収できるずれではない。** 学習元が等方 0.71 mm である限り、")
        out("5〜7 mm 厚の対象に対しては入力分布が一致しない。")
        out("学習元を SENORA と同等の分解能に落としてから学習し直す必要がある。")
        out("")

        out("## 症例ごとの結果")
        out("")
        out("| 症例 | Dice | 適合率 | 再現率 | 病変(mL) | 予測(mL) | 断面外(mm) |")
        out("|---|---|---|---|---|---|---|")
        for _, r in scored.sort_values("dice", ascending=False).iterrows():
            precision = (
                f"{r['precision']:.3f}" if pd.notna(r["precision"]) else "—"
            )
            out(f"| {r['subject']} | {r['dice']:.3f} | {precision} "
                f"| {r['recall']:.3f} | {r['volume_ml']:.1f} "
                f"| {r['pred_volume_ml']:.1f} | {r['through_plane_mm']:.1f} |")
        out("")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "senora_armc_predictions.csv"
    df.to_csv(csv_path, index=False)
    out(f"症例ごとの測定値を `{csv_path}` に出力しました。")

    report = "\n".join(lines)
    target = out_dir / "senora_armc_predictions.md"
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nレポートを {target} に保存しました")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "senora_normalized")
    parser.add_argument(
        "--work",
        type=Path,
        default=Path.home() / "senora_nnunet" / "armc_senora",
        help="ASCII のみの作業領域。日本語パスでは SimpleITK が読めない",
    )
    parser.add_argument(
        "--results", type=Path, default=Path.home() / "senora_nnunet" / "nnUNet_results"
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="nnU-Net / HD-BET の実行ファイルを探す基準",
    )
    parser.add_argument("--folds", default="0", help="使う fold。カンマ区切り")
    parser.add_argument("--out", type=Path, default=root / "results")
    parser.add_argument("--stage", action="store_true", help="複製と頭蓋除去")
    parser.add_argument("--predict", action="store_true", help="推論")
    parser.add_argument("--evaluate", action="store_true", help="評価")
    parser.add_argument("--all", action="store_true", help="上記を順に実行")
    args = parser.parse_args()

    if not any((args.stage, args.predict, args.evaluate, args.all)):
        parser.error("--stage / --predict / --evaluate / --all のいずれかを指定してください")

    if args.stage or args.all:
        code = stage(args.data, args.work, args.python)
        if code:
            return code
    if args.predict or args.all:
        code = predict(args.work, args.results, args.python, args.folds)
        if code:
            return code
    if args.evaluate or args.all:
        code = evaluate(
            args.work, args.out, args.out / "source_baseline_fold0.csv", args.results
        )
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
