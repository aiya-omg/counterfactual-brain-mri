"""
make_review_sheets.py
臨床医による判定（設計書 8.6.6 の優先度4）の資料を作る。

アームAでモデルの予測がマスクの外に出た領域（sub-052 で 112 mL、sub-037 で 51 mL）が、
参照マスクの描き漏れなのか、モデルの過検出なのかは画像だけでは決められない。
判定者が予測を知っていると判定が引きずられるので、予測は一切載せない。
対象も予測のはみ出しが大きい症例に絞らず、アームAの7例すべてを同じ形式で出す。

各症例について、脳を含む全スライスの b1000 と ADC に参照マスクの輪郭（緑）だけを重ねる。
記入票（review_form.csv）には症例とスライスごとに
「緑の輪郭の外に急性梗塞（b1000 高信号かつ ADC 低下）があるか」を書いてもらう。
症例の並びは乱数で固定し、症例番号の順序から情報が漏れないようにする。

使い方:
  python make_review_sheets.py
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


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj)


def sheet(work: Path, subject: str, code: str, path: Path) -> list[int]:
    brain = load(work / "brain" / f"{subject}_bet.nii.gz") > 0
    b1000 = load(work / "inputs" / f"{subject}_0000.nii.gz").astype(float)
    adc = load(work / "inputs" / f"{subject}_0001.nii.gz").astype(float)
    mask = load(work / "masks" / f"{subject}.nii.gz") > 0
    slices = [int(z) for z in np.flatnonzero(brain.sum(axis=(0, 1)) > 500)]

    vmax_b = np.percentile(b1000[brain], 99.5)
    cols = 4
    rows = int(np.ceil(len(slices) / (cols // 2)))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.1 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for i, z in enumerate(slices):
        for j, (name, img, vmin, vmax) in enumerate((("b1000", b1000, 0, vmax_b),
                                                     ("ADC", adc, 0, 2000))):
            ax = axes.ravel()[2 * i + j]
            ax.imshow(np.rot90(img[:, :, z]), cmap="gray", vmin=vmin, vmax=vmax)
            if mask[:, :, z].any():
                ax.contour(np.rot90(mask[:, :, z]), levels=[0.5], colors="lime", linewidths=0.9)
            ax.set_title(f"slice {z}  {name}", fontsize=9)
    fig.suptitle(f"Case {code}: green = reference mask. Is there acute infarction "
                 "(bright b1000 and dark ADC) OUTSIDE the green outline?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return slices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--work", type=Path, default=Path.home() / "senora_nnunet" / "arma_senora"
    )
    parser.add_argument("--out", type=Path, default=root / "results" / "review")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    subjects = sorted(p.name.replace(".nii.gz", "") for p in (args.work / "masks").glob("*.nii.gz"))
    if not subjects:
        print(f"エラー: マスクがありません: {args.work / 'masks'}", file=sys.stderr)
        return 1
    order = np.random.default_rng(args.seed).permutation(len(subjects))

    args.out.mkdir(parents=True, exist_ok=True)
    form, key = [], []
    for rank, idx in enumerate(order, start=1):
        subject, code = subjects[idx], f"R{rank:02d}"
        slices = sheet(args.work, subject, code, args.out / f"case_{code}.png")
        key.append({"code": code, "subject": subject})
        form += [{"case": code, "slice": z, "acute_outside_mask (yes/no/unsure)": "",
                  "location / comment": ""} for z in slices]
        print(f"{code}: {len(slices)} スライス")

    pd.DataFrame(form).to_csv(args.out / "review_form.csv", index=False, encoding="utf-8-sig")
    # 対応表は判定者に渡さない
    pd.DataFrame(key).to_csv(args.out / "key_do_not_share.csv", index=False)
    (args.out / "README.txt").write_text(
        "SENORA-MRI DWI review\n"
        "\n"
        "Each case_Rxx.png shows every slice of one patient: b1000 (left) and ADC (right).\n"
        "The green outline is the reference lesion mask drawn by the original radiologists.\n"
        "\n"
        "For each slice, please answer in review_form.csv:\n"
        "  Is there acute infarction (high signal on b1000 AND low ADC) outside the green outline?\n"
        "  yes / no / unsure, and a short location note if yes.\n"
        "\n"
        "Please do not consult any other annotation or model output while reviewing.\n",
        encoding="utf-8",
    )
    print(f"出力: {args.out}（判定者に渡すのは case_*.png、review_form.csv、README.txt のみ）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
