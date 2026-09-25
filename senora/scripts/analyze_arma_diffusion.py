"""
analyze_arma_diffusion.py
段階2（アームA）の結果を、病変内の拡散の状態で分解する（病期別の評価指標）。

なぜ必要か:

  SENORA アームAの7例は臨床区分では全例 acute だが、b1000 と ADC で見ると
  拡散制限を示すのは一部に限られる。読影医は新旧の梗塞をまとめて描いており
  （sub-014、sub-052 はマスクが新しい梗塞と古い梗塞の両方にかかる）、
  ISLES 2022 のラベル（拡散制限域 = 梗塞コア）とは定義が違う。
  Dice 1つでは「モデルが外した」のか「参照の定義が違う」のかが区別できない。

指標（設計書 8.6.3 で事前に固定）:

  参照マスクを2つに分ける。
    急性コア     マスク内で ADC が ADC_CORE 未満のボクセル
    それ以外     マスクの残り（亜急性期後半〜慢性期の成分）

  症例ごとに出すもの:
    dice              マスク全体に対する Dice（従来の指標）
    dice_core         急性コアに対する Dice
    recall_core       急性コアのうち予測されたもの（課題が合っている部分での性能）
    recall_noncore    それ以外のうち予測されたもの（病期ずれの大きさ）
    pred_core_frac    予測のうち ADC が ADC_CORE 未満の割合。マスク外の予測が
                      急性コア相当の組織かどうかを見る
    lesion_f1         病変単位の検出。参照の連結成分が予測と1ボクセルでも重なれば検出、
                      参照と重ならない予測成分は偽陽性。ISLES'22 と同じ定義
    fp_*              マスク外の予測（はみ出し）の性質

  ADC_CORE = 620 × 10^-6 mm²/s は梗塞コアの推定で広く使われる閾値。
  ADC の単位が違うと意味を失うので、正常脳の ADC 中央値が想定範囲にあるかを
  症例ごとに確かめてから使う。

  n = 7 のため、相関係数は「事前に予測した関係と一致したか」を見る記述に使い、
  検定の p 値は主張の根拠にしない。

使い方:
  python analyze_arma_diffusion.py                  # 最終重み、fold 0
  python analyze_arma_diffusion.py --tag final_f01234
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

ADC_CORE = 620.0  # × 10^-6 mm²/s
RESTRICTED_ADC = 0.8
RESTRICTED_B1000 = 1.3

# 脳マスク内・病変外の ADC 中央値の許容範囲（× 10^-6 mm²/s）。
# 実質だけなら 700〜900 程度だが、脳マスクは脳室と脳溝の CSF（約 3000）を含むため
# SENORA の7例では 855〜1154 になる。単位が 10^-3 なら 1 前後、10^-9 なら 10^-3 前後に
# なるので、この範囲で単位の取り違えは確実に弾ける
NORMAL_ADC_RANGE = (500.0, 1500.0)

CONNECTIVITY = ndimage.generate_binary_structure(3, 3)


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / total if total else float("nan")


def lesion_detection(ref: np.ndarray, pred: np.ndarray) -> dict:
    ref_lab, n_ref = ndimage.label(ref, CONNECTIVITY)
    pred_lab, n_pred = ndimage.label(pred, CONNECTIVITY)
    detected = len(set(np.unique(ref_lab[pred])) - {0})
    pred_hit = len(set(np.unique(pred_lab[ref])) - {0})
    tp, fn, fp = detected, n_ref - detected, n_pred - pred_hit
    denom = 2 * tp + fp + fn
    return {
        "ref_lesions": n_ref,
        "pred_lesions": n_pred,
        "lesions_detected": tp,
        "lesions_fp": fp,
        "lesion_f1": 2 * tp / denom if denom else float("nan"),
    }


def measure(work: Path, pred_dir: Path, subject: str) -> tuple[dict, dict]:
    ref_img = nib.load(work / "masks" / f"{subject}.nii.gz")
    ref = np.asarray(ref_img.dataobj) > 0
    pred = load(pred_dir / f"{subject}.nii.gz") > 0
    brain = load(work / "brain" / f"{subject}_bet.nii.gz") > 0
    b0 = load(work / "b0" / f"{subject}.nii.gz").astype(np.float32)
    b1000 = load(work / "inputs" / f"{subject}_0000.nii.gz")
    adc = load(work / "inputs" / f"{subject}_0001.nii.gz")
    voxel_ml = abs(np.linalg.det(ref_img.affine[:3, :3])) / 1000.0

    normal = brain & ~ref
    nb, na = float(np.median(b1000[normal])), float(np.median(adc[normal]))
    if not NORMAL_ADC_RANGE[0] <= na <= NORMAL_ADC_RANGE[1]:
        raise ValueError(
            f"{subject}: 正常脳の ADC 中央値 {na:.3g} が {NORMAL_ADC_RANGE} の外。"
            f"ADC の単位が × 10^-6 mm²/s でない可能性があり、ADC_CORE = {ADC_CORE} を使えない"
        )

    restricted = (adc < RESTRICTED_ADC * na) & (b1000 > RESTRICTED_B1000 * nb)
    core_tissue = brain & (adc > 0) & (adc < ADC_CORE)
    core = ref & core_tissue
    rest = ref & ~core

    row = {
        "subject": subject,
        "dice": dice(pred, ref),
        "volume_ml": round(float(ref.sum() * voxel_ml), 3),
        "pred_volume_ml": round(float(pred.sum() * voxel_ml), 3),
        "normal_adc": round(na, 1),
        "adc_ratio": float(np.median(adc[ref]) / na),
        "b1000_ratio": float(np.median(b1000[ref]) / nb),
        "restricted_frac": float(restricted[ref].mean()),
        "core_ml": round(float(core.sum() * voxel_ml), 3),
        "core_frac": float(core.sum() / ref.sum()),
        "dice_core": dice(pred, core) if core.any() else float("nan"),
        "recall_core": float((pred & core).sum() / core.sum()) if core.any() else float("nan"),
        "recall_noncore": float((pred & rest).sum() / rest.sum()) if rest.any() else float("nan"),
        "pred_core_frac": float(core_tissue[pred].mean()) if pred.any() else float("nan"),
    }
    row.update(lesion_detection(ref, pred))

    fp = pred & ~ref
    if fp.any():
        lab, n = ndimage.label(fp, CONNECTIVITY)
        near = ndimage.binary_dilation(ref, iterations=2)
        touching = sum(int((lab == i).sum()) for i in range(1, n + 1) if (near & (lab == i)).any())
        row.update({
            "fp_ml": round(float(fp.sum() * voxel_ml), 3),
            "fp_touching_frac": touching / int(fp.sum()),
            "fp_adc_ratio": float(np.median(adc[fp]) / na),
            "fp_restricted_frac": float(restricted[fp].mean()),
            "fp_core_frac": float(core_tissue[fp].mean()),
        })
    images = {"b0": b0, "b1000": b1000, "ADC": adc, "ref": ref, "pred": pred}
    return row, images


def draw(panels: list[tuple[str, float, dict]], path: Path, tag: str) -> None:
    fig, axes = plt.subplots(len(panels), 3, figsize=(9, 3.1 * len(panels)))
    for r, (subject, score, im) in enumerate(panels):
        ref, pred = im["ref"], im["pred"]
        z = int(np.argmax(ref.sum(axis=(0, 1)) + 0.5 * pred.sum(axis=(0, 1))))
        for c, name in enumerate(("b0", "b1000", "ADC")):
            ax = axes[r, c]
            img = im[name].astype(float)
            vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
            ax.imshow(np.rot90(img[:, :, z]), cmap="gray", vmin=0, vmax=vmax)
            ax.contour(np.rot90(ref[:, :, z]), levels=[0.5], colors="lime", linewidths=1.0)
            ax.contour(np.rot90(pred[:, :, z]), levels=[0.5], colors="red", linewidths=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{subject}\nDice {score:.3f}\nslice {z}", fontsize=10)
            if r == 0:
                ax.set_title(name, fontsize=11)
    fig.suptitle(f"SENORA Arm A ({tag}): green = reference mask, red = prediction",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def report(df: pd.DataFrame, tag: str) -> str:
    lines = [
        f"# 病期別の評価（{tag}、{len(df)} 例）",
        "",
        f"急性コア = マスク内で ADC が {ADC_CORE:.0f} × 10⁻⁶ mm²/s 未満。"
        "定義は設計書 8.6.3。",
        "",
        "| 症例 | Dice | コアの割合 | Dice（コア） | 再現率（コア） | 再現率（それ以外） "
        "| 予測のコア組織割合 | 病変 F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['subject']} | {fmt(r['dice'])} | {r['core_frac']:.0%} | {fmt(r['dice_core'])} "
            f"| {fmt(r['recall_core'])} | {fmt(r['recall_noncore'])} "
            f"| {fmt(r['pred_core_frac'], 2)} | {fmt(r['lesion_f1'], 2)} |"
        )
    with_core = df[df["core_ml"] >= 0.5]
    lines += [
        "",
        f"- Dice 中位（マスク全体）: **{df['dice'].median():.3f}**",
        f"- 急性コアが 0.5 mL 以上ある症例: {len(with_core)} / {len(df)}",
    ]
    if len(with_core):
        lines += [
            f"  - Dice（コア）中位 **{with_core['dice_core'].median():.3f}**、"
            f"再現率（コア）中位 **{with_core['recall_core'].median():.3f}**",
            f"  - 再現率（それ以外）中位 {with_core['recall_noncore'].median():.3f}",
        ]
    no_core = df[df["core_ml"] < 0.5]
    if len(no_core):
        silent = int((no_core["pred_volume_ml"] <= 1.0).sum())
        lines.append(
            f"- 急性コアが 0.5 mL 未満の症例: {len(no_core)} 例。予測体積 中位 "
            f"{no_core['pred_volume_ml'].median():.2f} mL、1 mL 以下 {silent} / {len(no_core)}"
        )
    lines.append(f"- 病変 F1 中位: {df['lesion_f1'].median():.2f}")
    lines += ["", "記述のための順位相関（n が小さいので p 値は根拠にしない）:", ""]
    for col in ("restricted_frac", "core_frac", "volume_ml"):
        rho, _ = stats.spearmanr(df[col], df["dice"])
        lines.append(f"- Dice と {col}: ρ = {rho:+.2f}")
    rho_v, _ = stats.spearmanr(df["volume_ml"], df["pred_volume_ml"])
    rho_c, _ = stats.spearmanr(df["core_ml"], df["pred_volume_ml"])
    lines.append(f"- 予測体積とマスク体積: ρ = {rho_v:+.2f}、予測体積と急性コア体積: ρ = {rho_c:+.2f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--work", type=Path, default=Path.home() / "senora_nnunet" / "arma_senora"
    )
    parser.add_argument("--out", type=Path, default=root / "results" / "dataset503")
    parser.add_argument(
        "--tag", default="final_f0",
        help="predict_senora_arma.py の重みと fold の識別子（例: final_f0、final_f01234）",
    )
    args = parser.parse_args()

    pred_dir = args.work / f"predictions_{args.tag}"
    subjects = sorted(p.name.replace(".nii.gz", "") for p in pred_dir.glob("*.nii.gz"))
    if not subjects:
        print(f"エラー: 予測がありません: {pred_dir}", file=sys.stderr)
        return 1

    rows, panels = [], []
    for subject in subjects:
        row, images = measure(args.work, pred_dir, subject)
        rows.append(row)
        panels.append((subject, row["dice"], images))
    order = np.argsort([-r["dice"] for r in rows])
    df = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    panels = [panels[i] for i in order]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"senora_arma_stage_{args.tag}.csv"
    df.to_csv(csv_path, index=False)
    draw(panels, args.out / f"arma_overlay_{args.tag}.png", args.tag)
    text = report(df, args.tag)
    (args.out / f"senora_arma_stage_{args.tag}.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n出力: {csv_path}、arma_overlay_{args.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
