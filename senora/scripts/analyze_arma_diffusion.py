"""
analyze_arma_diffusion.py
段階2（アームA）の結果を、病変内の拡散の状態で分解する。

なぜ必要か:

  SENORA アームAの7例は臨床区分では全例 acute だが、b1000 と ADC で見ると
  拡散制限を示すのは一部に限られる。読影医は新旧の梗塞をまとめて描いており
  （sub-014、sub-052 はマスクが新しい梗塞と古い梗塞の両方にかかる）、
  ISLES 2022 のラベル（拡散制限域 = 梗塞コア）とは定義が違う。
  Dice 1つでは「モデルが外した」のか「参照の定義が違う」のかが区別できない。

測るもの:

  1. マスクのうち拡散制限を示すボクセルの割合（ADC < 0.8 × 正常、
     b1000 > 1.3 × 正常。正常 = 脳マスク内でマスク外の中央値）と Dice の関係
  2. 偽陽性の性質。マスクに接しているか、拡散制限を示しているか
  3. マスクを急性コア（ADC < 620 × 10^-6 mm²/s）とそれ以外に分けた再現率。
     620 は梗塞コアの推定で広く使われる閾値
  4. 症例ごとの断面図（b0 / b1000 / ADC に参照と予測の輪郭）

使い方:
  python analyze_arma_diffusion.py
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

ADC_CORE = 620.0
RESTRICTED_ADC = 0.8
RESTRICTED_B1000 = 1.3


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj)


def measure(work: Path, subject: str) -> tuple[dict, dict]:
    ref_img = nib.load(work / "masks" / f"{subject}.nii.gz")
    ref = np.asarray(ref_img.dataobj) > 0
    pred = load(work / "predictions" / f"{subject}.nii.gz") > 0
    brain = load(work / "brain" / f"{subject}_bet.nii.gz") > 0
    b0 = load(work / "b0" / f"{subject}.nii.gz").astype(np.float32)
    b1000 = load(work / "inputs" / f"{subject}_0000.nii.gz")
    adc = load(work / "inputs" / f"{subject}_0001.nii.gz")
    voxel_ml = abs(np.linalg.det(ref_img.affine[:3, :3])) / 1000.0

    normal = brain & ~ref
    nb, na = np.median(b1000[normal]), np.median(adc[normal])
    restricted = (adc < RESTRICTED_ADC * na) & (b1000 > RESTRICTED_B1000 * nb)
    core = ref & (adc > 0) & (adc < ADC_CORE)
    rest = ref & ~core

    row = {
        "subject": subject,
        "adc_ratio": float(np.median(adc[ref]) / na),
        "b1000_ratio": float(np.median(b1000[ref]) / nb),
        "restricted_frac": float(restricted[ref].mean()),
        "core_ml": round(float(core.sum() * voxel_ml), 2),
        "core_frac": float(core.sum() / ref.sum()),
        "recall_core": float((pred & core).sum() / core.sum()) if core.any() else np.nan,
        "recall_noncore": float((pred & rest).sum() / rest.sum()) if rest.any() else np.nan,
    }
    fp = pred & ~ref
    if fp.any():
        lab, n = ndimage.label(fp)
        near = ndimage.binary_dilation(ref, iterations=2)
        touching = sum(int((lab == i).sum()) for i in range(1, n + 1) if (near & (lab == i)).any())
        row.update({
            "fp_ml": round(float(fp.sum() * voxel_ml), 2),
            "fp_touching_frac": touching / int(fp.sum()),
            "fp_adc_ratio": float(np.median(adc[fp]) / na),
            "fp_restricted_frac": float(restricted[fp].mean()),
        })
    images = {"b0": b0, "b1000": b1000, "ADC": adc, "ref": ref, "pred": pred}
    return row, images


def draw(panels: list[tuple[str, float, dict]], path: Path) -> None:
    fig, axes = plt.subplots(len(panels), 3, figsize=(9, 3.1 * len(panels)))
    for r, (subject, dice, im) in enumerate(panels):
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
                ax.set_ylabel(f"{subject}\nDice {dice:.3f}\nslice {z}", fontsize=10)
            if r == 0:
                ax.set_title(name, fontsize=11)
    fig.suptitle("SENORA Arm A: green = reference mask, red = Dataset503 prediction",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--work", type=Path, default=Path.home() / "senora_nnunet" / "arma_senora"
    )
    parser.add_argument("--out", type=Path, default=root / "results" / "dataset503")
    args = parser.parse_args()

    results = pd.read_csv(args.out / "senora_arma_predictions.csv")
    results = results.sort_values("dice", ascending=False)
    rows, panels = [], []
    for _, r in results.iterrows():
        row, images = measure(args.work, r["subject"])
        rows.append(row)
        panels.append((r["subject"], r["dice"], images))

    df = results.merge(pd.DataFrame(rows), on="subject")
    df.to_csv(args.out / "senora_arma_diffusion.csv", index=False)
    draw(panels, args.out / "arma_overlay.png")

    cols = ["subject", "dice", "volume_ml", "restricted_frac", "adc_ratio", "core_frac",
            "recall_core", "recall_noncore", "fp_ml", "fp_touching_frac", "fp_restricted_frac"]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    for col in ("restricted_frac", "adc_ratio", "volume_ml"):
        rho, p = stats.spearmanr(df[col], df["dice"])
        print(f"Spearman Dice vs {col}: rho = {rho:+.2f} (p = {p:.3f}, n = {len(df)})")
    print(f"出力: {args.out / 'senora_arma_diffusion.csv'}, {args.out / 'arma_overlay.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
