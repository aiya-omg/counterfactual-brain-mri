"""
apply_unlabeled.py
マスクのない SENORA 症例に Dataset503 の DWI モデルをかけ、ADC から作る参照で評価する（設計書 8.6.11）。

読影医の代わりの参照（23例で閾値を決め、80例を見る前に固定した）:
  急性コアの塊  脳マスク内で ADC < 620 かつ b1000 > 1.3 × 正常実質のボクセルを 26 近傍でつなぎ、
                0.5 mL 以上で、重心が脳の下から 30% より上にある連結成分
  コアあり      急性コアの塊が1つ以上ある症例

下 30% を除くのは頭蓋底・後頭蓋窩の磁化率アーチファクトが同じ条件を満たすため。
参照もモデルも ADC を使うので、一致は部分的に構成上のもの。示すのは「急性の拡散制限に反応するか」。

作業領域は arma_senora と同じ構成（b0 / brain / inputs / predictions_<tag>）。

使い方:
  python apply_unlabeled.py --stage --predict --analyze
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_arma_diffusion import ADC_CORE, CONNECTIVITY, RESTRICTED_B1000, normal_reference  # noqa: E402
from predict_senora_arma import b1000_index, predict, run_tag  # noqa: E402
from stage_senora_lesions import brain_masks, save  # noqa: E402

COMPONENT_MIN_ML = 0.5
INFERIOR_EXCLUDED = 0.30
PRED_EMPTY_ML = 1.0
OFF_REFERENCE_ML = 1.0
EXCLUDED = {"sub-026": "sub-027 と同じ2検査を含む重複フォルダ（設計書 8.6.11）"}


def subjects(data_root: Path) -> tuple[list[str], dict[str, str]]:
    participants = pd.read_csv(data_root / "participants.tsv", sep="\t")
    unlabeled = participants[participants["has_lesion_mask"].astype(str).str.lower() == "no"]
    keep, skipped = [], dict(EXCLUDED)
    for s in sorted(unlabeled["participant_id"].astype(str)):
        if s in skipped:
            continue
        dwi = data_root / s / "dwi"
        if not (dwi / f"{s}_dwi.nii.gz").exists() or not (dwi / f"{s}_desc-ADC_dwi.nii.gz").exists():
            skipped[s] = "正規化後に DWI か ADC がない"
            continue
        keep.append(s)
    return keep, skipped


def stage(data_root: Path, work: Path, cases: list[str]) -> None:
    for d in ("b0", "brain", "inputs"):
        (work / d).mkdir(parents=True, exist_ok=True)
    for s in cases:
        dwi_dir = data_root / s / "dwi"
        img = nib.load(dwi_dir / f"{s}_dwi.nii.gz")
        dwi = np.asarray(img.dataobj, dtype=np.float32)
        save(dwi[..., 0], img, work / "b0" / f"{s}.nii.gz")
        save(dwi[..., b1000_index(dwi_dir / f"{s}_dwi.bval")], img, work / f"{s}_b1000.nii.gz")
        adc = nib.load(dwi_dir / f"{s}_desc-ADC_dwi.nii.gz")
        save(np.asarray(adc.dataobj, dtype=np.float32), adc, work / f"{s}_adc.nii.gz")
    todo = [s for s in cases if not (work / "brain" / f"{s}_bet.nii.gz").exists()]
    if todo:
        print(f"頭蓋除去（b0、{len(todo)} 例）")
        brain_masks(work, todo)
    for s in cases:
        brain = np.asarray(nib.load(work / "brain" / f"{s}_bet.nii.gz").dataobj) > 0
        for channel, name in (("0000", "b1000"), ("0001", "adc")):
            src = nib.load(work / f"{s}_{name}.nii.gz")
            save(np.asarray(src.dataobj, dtype=np.float32) * brain, src,
                 work / "inputs" / f"{s}_{channel}.nii.gz")


def core_components(adc: np.ndarray, b1000: np.ndarray, brain: np.ndarray, nb: float,
                    voxel_ml: float) -> tuple[np.ndarray, int]:
    """急性コアの塊にラベルを振った配列と個数を返す。"""
    candidate = brain & (adc > 0) & (adc < ADC_CORE) & (b1000 > RESTRICTED_B1000 * nb)
    z = np.where(brain.any(axis=(0, 1)))[0]
    z_cut = z.min() + INFERIOR_EXCLUDED * (z.max() - z.min())
    lab, n = ndimage.label(candidate, CONNECTIVITY)
    out = np.zeros(lab.shape, dtype=np.int32)
    kept = 0
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() * voxel_ml < COMPONENT_MIN_ML or ndimage.center_of_mass(comp)[2] < z_cut:
            continue
        kept += 1
        out[comp] = kept
    return out, kept


def measure(work: Path, pred_dir: Path, s: str) -> tuple[dict, dict]:
    def load(path: Path) -> np.ndarray:
        return np.asarray(nib.load(path).dataobj)

    b0 = nib.load(work / "b0" / f"{s}.nii.gz")
    voxel_ml = abs(np.linalg.det(b0.affine[:3, :3])) / 1000.0
    brain = load(work / "brain" / f"{s}_bet.nii.gz") > 0
    adc = load(work / f"{s}_adc.nii.gz").astype(np.float64)
    b1000 = load(work / f"{s}_b1000.nii.gz").astype(np.float64)
    pred = load(pred_dir / f"{s}.nii.gz") > 0
    zooms = b0.header.get_zooms()[:3]
    row = {"subject": s, "slice_mm": round(float(zooms[2]), 2),
           "pred_ml": float(pred.sum() * voxel_ml)}
    try:
        na, nb = normal_reference(adc, b1000, brain, np.zeros_like(brain), s)
    except ValueError as exc:
        row["error"] = str(exc)
        return row, {}
    comps, n = core_components(adc, b1000, brain, nb, voxel_ml)
    core = comps > 0
    detected = [bool((pred & (comps == k)).any()) for k in range(1, n + 1)]
    pred_lab, n_pred = ndimage.label(pred, CONNECTIVITY)
    off = [float((pred_lab == k).sum() * voxel_ml) for k in range(1, n_pred + 1)
           if not (core & (pred_lab == k)).any()]
    row.update({
        "normal_adc": round(na, 1),
        "core_lesions": n,
        "core_ml": float(core.sum() * voxel_ml),
        "core_detected": int(sum(detected)),
        "core_recall": float((pred & core).sum() / core.sum()) if n else float("nan"),
        "pred_on_core_ml": float((pred & core).sum() * voxel_ml),
        "off_reference_ml": float(sum(off)),
        "off_reference_max_ml": float(max(off)) if off else 0.0,
    })
    return row, {"adc": adc, "b1000": b1000, "brain": brain, "pred": pred, "comps": comps}


def draw(panels: list[tuple[str, str, dict, np.ndarray]], path: Path, title: str) -> None:
    if not panels:
        return
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (s, label, v, focus) in zip(axes.ravel(), panels):
        z = int(np.argmax(focus.sum(axis=(0, 1)))) if focus.any() else v["b1000"].shape[2] // 2
        sl = v["b1000"][:, :, z]
        inside = v["brain"][:, :, z]
        vmax = np.percentile(sl[inside], 99.5) if inside.any() else sl.max()
        ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=vmax)
        ax.contour(np.rot90(v["comps"][:, :, z] > 0), levels=[0.5], colors="red", linewidths=1.0)
        ax.contour(np.rot90(v["pred"][:, :, z]), levels=[0.5], colors="yellow", linewidths=0.8)
        ax.set_title(f"{s} z={z} {label}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def verdict(ok: bool) -> str:
    return "当たり" if ok else "**外れ**"


def report(df: pd.DataFrame, skipped: dict[str, str], tag: str) -> str:
    lines = [f"# マスクのない症例への DWI モデルの適用（{tag}）", ""]
    out = lines.append
    out("参照は ADC から作る急性コアの塊（設計書 8.6.11）。参照もモデルも ADC を使うので、"
        "一致は部分的に構成上のもの。")
    out("")
    if skipped:
        out("除外: " + "、".join(f"{s}（{why}）" for s, why in skipped.items()))
        out("")
    errors = df[df["error"].notna()] if "error" in df else df.iloc[0:0]
    for _, r in errors.iterrows():
        out(f"- {r['subject']}: {r['error']}")
    ok = df[df["error"].isna()] if "error" in df else df
    main = ok[ok["stroke_subtype"] == "ischemic"]
    other = ok[ok["stroke_subtype"] != "ischemic"]

    positive = main[main["core_lesions"] > 0]
    negative = main[main["core_lesions"] == 0]
    lesions, detected = int(main["core_lesions"].sum()), int(main["core_detected"].sum())
    acute = main[main["presentation"] == "acute"]
    q1 = detected / lesions if lesions else float("nan")
    q2 = float((negative["pred_ml"] <= PRED_EMPTY_ML).mean()) if len(negative) else float("nan")
    q3 = float((acute["core_lesions"] == 0).mean()) if len(acute) else float("nan")
    rho = stats.spearmanr(main["pred_ml"], main["core_ml"]).statistic if len(main) > 2 else float("nan")

    out(f"## 予測の検証（主解析: 虚血性 {len(main)} 例）")
    out("")
    out("| # | 予測 | 結果 | 判定 |")
    out("|---|---|---|---|")
    out(f"| Q1 | 急性コアの塊の 80% 以上に予測が重なる | {detected} / {lesions}（{q1:.0%}） "
        f"| {verdict(q1 >= 0.8)} |")
    out(f"| Q2 | コアなし症例の 85% 以上で予測 ≤ {PRED_EMPTY_ML:g} mL "
        f"| {int((negative['pred_ml'] <= PRED_EMPTY_ML).sum())} / {len(negative)}（{q2:.0%}） "
        f"| {verdict(q2 >= 0.85)} |")
    out(f"| Q3 | 臨床区分が急性の症例の 25% 以上がコアなし "
        f"| {int((acute['core_lesions'] == 0).sum())} / {len(acute)}（{q3:.0%}） "
        f"| {verdict(q3 >= 0.25)} |")
    out(f"| Q4 | 予測体積とコア体積の Spearman ρ ≥ 0.6 | ρ = {rho:.2f} | {verdict(rho >= 0.6)} |")
    out("")

    table = pd.crosstab(main["presentation"].replace("", "（空欄）"),
                        np.where(main["core_lesions"] > 0, "コアあり", "コアなし"),
                        margins=True, margins_name="計")
    out("### 臨床区分と画像上のコア")
    out("")
    out("| 臨床区分 | " + " | ".join(str(c) for c in table.columns) + " |")
    out("|---|" + "---|" * len(table.columns))
    for idx, r in table.iterrows():
        out(f"| {idx} | " + " | ".join(str(v) for v in r) + " |")
    out("")
    if len(positive):
        out(f"- コアあり {len(positive)} 例: コア体積 中位 {positive['core_ml'].median():.1f} mL、"
            f"予測体積 中位 {positive['pred_ml'].median():.1f} mL、"
            f"コアの再現率 中位 {positive['core_recall'].median():.2f}")
    if len(negative):
        out(f"- コアなし {len(negative)} 例: 予測体積 中位 {negative['pred_ml'].median():.2f} mL、"
            f"最大 {negative['pred_ml'].max():.1f} mL")
    out("")

    out("## 症例ごとの値")
    out("")
    out("| 症例 | 年齢 | 区分 | 病型 | 間隔 (mm) | コアの塊 | 検出 | コア (mL) | 予測 (mL) "
        "| 再現率 | コア外の予測 (mL) |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ok.sort_values(["core_ml", "pred_ml"], ascending=False).iterrows():
        recall = "—" if pd.isna(r["core_recall"]) else f"{r['core_recall']:.2f}"
        out(f"| {r['subject']} | {r['age']} | {r['presentation'] or '—'} | {r['stroke_subtype']} "
            f"| {r['slice_mm']:g} | {r['core_lesions']:.0f} | {r['core_detected']:.0f} "
            f"| {r['core_ml']:.1f} | {r['pred_ml']:.1f} | {recall} | {r['off_reference_ml']:.1f} |")
    out("")
    flagged = ok[ok["off_reference_max_ml"] >= OFF_REFERENCE_ML]
    out(f"## コアの塊に重ならない {OFF_REFERENCE_ML:g} mL 以上の予測（探索的、目視用）")
    out("")
    out("、".join(f"{r['subject']}（{r['off_reference_max_ml']:.1f} mL）"
                 for _, r in flagged.iterrows()) or "なし")
    out("")
    if len(other):
        out(f"## 虚血性以外（{len(other)} 例、値のみ）")
        out("")
        out("、".join(f"{r['subject']}（{r['stroke_subtype']}、コア {r['core_ml']:.1f} mL、"
                     f"予測 {r['pred_ml']:.1f} mL）" for _, r in other.iterrows()))
        out("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    home = Path.home() / "senora_nnunet"
    parser.add_argument("--data", type=Path, default=root / "data" / "senora_normalized")
    parser.add_argument("--work", type=Path, default=home / "senora_unlabeled")
    parser.add_argument("--results", type=Path, default=home / "nnUNet_results")
    parser.add_argument("--dataset", default="Dataset503_ISLES22DWIThick")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", type=Path, default=root / "results" / "unlabeled")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    cases, skipped = subjects(args.data)
    print(f"対象 {len(cases)} 例、除外 {len(skipped)} 例")
    if args.stage:
        stage(args.data, args.work, cases)
    if args.predict:
        code = predict(args.work, args.results, args.python, args.folds, args.dataset,
                       args.checkpoint)
        if code:
            return code
    if not args.analyze:
        return 0

    tag = run_tag(args.checkpoint, args.folds)
    pred_dir = args.work / f"predictions_{tag}"
    participants = pd.read_csv(args.data / "participants.tsv", sep="\t").set_index("participant_id")
    rows, positive_panels, off_panels = [], [], []
    for s in cases:
        row, v = measure(args.work, pred_dir, s)
        p = participants.loc[s]
        row["age"] = p["age"]
        row["presentation"] = "" if pd.isna(p["presentation_status"]) else str(p["presentation_status"]).strip().lower()
        row["stroke_subtype"] = "（なし）" if pd.isna(p["stroke_subtype"]) else str(p["stroke_subtype"]).strip().lower()
        rows.append(row)
        if not v:
            continue
        if row["core_lesions"]:
            label = f"core {row['core_ml']:.1f} / pred {row['pred_ml']:.1f} mL"
            positive_panels.append((s, label, v, v["comps"] > 0))
        if row["off_reference_max_ml"] >= OFF_REFERENCE_ML:
            off = v["pred"] & ~ndimage.binary_dilation(v["comps"] > 0, CONNECTIVITY)
            off_panels.append((s, f"off-ref {row['off_reference_max_ml']:.1f} mL", v, off))
    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = np.nan

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"unlabeled_{tag}.csv", index=False)
    draw(positive_panels, args.out / f"unlabeled_core_{tag}.png",
         "Cases with ADC core: b1000, red = ADC core component, yellow = model")
    draw(off_panels, args.out / f"unlabeled_offref_{tag}.png",
         "Predictions not touching an ADC core component (>= 1 mL)")
    text = report(df, skipped, tag)
    (args.out / f"unlabeled_{tag}.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
