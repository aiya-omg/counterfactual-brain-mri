"""
stage_senora_lesions.py
SENORA-MRI のマスク23本すべてを DWI の格子に揃え、画像上の病期を測る。

なぜ必要か:

  アームAの7例では、読影医のマスクが新旧の梗塞をまとめて描いており、
  ISLES 2022 のラベル（拡散制限域）と定義が違うことが Dice 低下の主因だった
  （設計書 8.6.4）。この病期ずれが SENORA 全体に言えるのかを知るには、
  FLAIR 上に描かれたアームCの16例も、同じ検査の b1000 / ADC で測る必要がある。
  participants.tsv には発症から撮像までの時間がないため、画像上の病期で代用する。

位置合わせ:

  FLAIR と DWI は同じ検査内で数分違いに撮られている（sub-027 のみ約9時間離れる）。
  ヘッダの座標だけで揃う可能性が高いが、頭の動きは残りうるので、
  FLAIR → b0 の剛体変換（相互情報量）を推定する。

  register_isles_flair.py の QC は変換とその逆で往復させたもので、
  変換が誤っていても 1 に近くなる。ここでは変換の推定に使っていない解剖構造で測る。
    脳 Dice   FLAIR と b0 それぞれに別々に HD-BET をかけた脳マスクの重なり
    CSF Dice  FLAIR の低信号（抑制された CSF）と ADC の高値（自由拡散）の重なり
  さらに採用した変換を 3 mm ずらしたときの CSF Dice も出し、この指標が
  ずれに反応することを症例ごとに示す。反応しない指標なら QC の役に立たない。

  判定: 推定した変換の CSF Dice がヘッダだけの場合より 0.02 以上悪ければ
  ヘッダの座標を使う。どちらを使ったかは registration_qc.csv に残す。
  撮像時刻が 60 分以上離れている、または採用した変換の CSF Dice が 3 mm ずらしを
  0.02 以上上回らない症例は「未検証」とし、病期の集計から除く。
  この2条件は16例の QC を見たあとに決めた（sub-027 が該当）。

病期の指標（analyze_arma_diffusion.py と同じ定義）:
  core_frac         マスク内で ADC < 620 × 10^-6 mm²/s の割合（急性コア）
  restricted_frac   ADC < 0.8 × 正常 かつ b1000 > 1.3 × 正常 の割合
  cavity_frac       ADC > 2000 の割合（CSF 様の自由拡散。慢性期の空洞化）
  画像上の病期      急性コア優位（core_frac ≥ 0.5）／ 急性コアを一部含む
                    （core ≥ 0.5 mL）／ 急性コアなし

出力の作業領域は arma_senora と同じ構成（b0 / brain / inputs / masks）なので、
predict_senora_arma.py と analyze_arma_diffusion.py に --work で渡せば23例に適用できる。

前提: predict_senora_armc.py --stage（FLAIR の頭蓋除去）と
      predict_senora_arma.py --stage（アームAの b0 脳マスク）が済んでいること。
      後者がなければ HD-BET を全例にかける。

使い方:
  python stage_senora_lesions.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_arma_diffusion import (  # noqa: E402
    ADC_CORE, normal_reference, restricted_tissue,
)
from predict_senora_arma import b1000_index  # noqa: E402
from predict_senora_armc import find_mask  # noqa: E402

CSF_ADC = 2000.0  # × 10^-6 mm²/s。CSF は約 3000、実質は 700〜900
CSF_FLAIR = 0.5   # 脳内の FLAIR 中央値に対する比。FLAIR では CSF が抑制されて暗い
SHIFT_MM = 3.0
QC_TOLERANCE = 0.02
QC_RESPONSE = 0.02    # 採用した変換の CSF Dice が 3 mm ずらしをこれ以上上回らなければ検証できていない
SAME_SESSION_MIN = 60.0
CORE_MIN_ML = 0.5


def to_3d(data: np.ndarray) -> np.ndarray:
    return data.reshape(data.shape[:3] + (-1,)).max(axis=3) if data.ndim > 3 else data


def save(data: np.ndarray, like: nib.Nifti1Image, path: Path) -> None:
    header = like.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, like.affine, header), path)


def prepare(data_root: Path, work: Path, armc_work: Path, arma_work: Path) -> pd.DataFrame:
    for d in ("b0", "brain", "inputs", "masks", "flair", "flair_brain", "flair_mask"):
        (work / d).mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(data_root / "_segmentation_report.csv")
    rows = []
    for _, r in seg.sort_values("subject_id").iterrows():
        s, reference = str(r["subject_id"]), str(r["sequence"]).strip().upper()
        dwi_dir = data_root / s / "dwi"
        img = nib.load(dwi_dir / f"{s}_dwi.nii.gz")
        dwi = np.asarray(img.dataobj, dtype=np.float32)
        b1000 = dwi[..., b1000_index(dwi_dir / f"{s}_dwi.bval")]
        adc = np.asarray(nib.load(dwi_dir / f"{s}_desc-ADC_dwi.nii.gz").dataobj, dtype=np.float32)
        save(dwi[..., 0], img, work / "b0" / f"{s}.nii.gz")
        save(b1000, img, work / f"{s}_b1000.nii.gz")
        save(adc, img, work / f"{s}_adc.nii.gz")

        mask = to_3d(np.asarray(nib.load(find_mask(data_root, s)).dataobj)) > 0
        if reference == "DWI":
            save(mask.astype(np.uint8), img, work / "masks" / f"{s}.nii.gz")
        else:
            flair_img = nib.load(data_root / s / "anat" / f"{s}_FLAIR.nii.gz")
            stripped = armc_work / "flair_stripped" / f"{s}_0000.nii.gz"
            if not stripped.exists():
                raise FileNotFoundError(f"{stripped} がない。predict_senora_armc.py --stage が先")
            save(np.asarray(flair_img.dataobj, dtype=np.float32), flair_img,
                 work / "flair" / f"{s}.nii.gz")
            save((np.asarray(nib.load(stripped).dataobj) > 0).astype(np.uint8), flair_img,
                 work / "flair_brain" / f"{s}.nii.gz")
            save(mask.astype(np.uint8), flair_img, work / "flair_mask" / f"{s}.nii.gz")

        bet = work / "brain" / f"{s}_bet.nii.gz"
        reuse = arma_work / "brain" / f"{s}_bet.nii.gz"
        if not bet.exists() and reuse.exists():
            shutil.copy(reuse, bet)
        rows.append({"subject": s, "reference": reference})

    todo = [r["subject"] for r in rows if not (work / "brain" / f"{r['subject']}_bet.nii.gz").exists()]
    if todo:
        print(f"頭蓋除去（b0、{len(todo)} 例）")
        brain_masks(work, todo)
    return pd.DataFrame(rows)


def brain_masks(work: Path, subjects: list[str]) -> None:
    """
    b0 に HD-BET をかけて脳マスクを作る。

    hd-bet コマンドは前処理と書き出しに計12個の子プロセスを起こし、学習と同時に走らせると
    Windows のコミットメモリを使い切って学習ごと落ちる（2026-09-26 に fold 1 が落ちた）。
    同じプロセス内で1例ずつ推論する。
    """
    import torch
    from HD_BET.hd_bet_prediction import get_hdbet_predictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    predictor = get_hdbet_predictor(use_tta=True, device=torch.device("cuda"))
    io = SimpleITKIO()
    for s in subjects:
        image, props = io.read_images([str(work / "b0" / f"{s}.nii.gz")])
        seg = predictor.predict_single_npy_array(image, props, None, None, False)
        io.write_seg(seg.astype(np.uint8), str(work / "brain" / f"{s}_bet.nii.gz"), props)
        print(f"  {s}: 脳 {int((seg > 0).sum())} ボクセル")
    del predictor
    torch.cuda.empty_cache()


def read(path: Path, pixel=sitk.sitkFloat32) -> sitk.Image:
    return sitk.Cast(sitk.ReadImage(str(path)), pixel)


def resample(moving: sitk.Image, fixed: sitk.Image, tx: sitk.Transform) -> np.ndarray:
    """moving を fixed の格子へ線形補間で写し、nibabel と同じ (x, y, z) 順で返す。"""
    out = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / total if total else float("nan")


def shifted(tx: sitk.Euler3DTransform, offset) -> sitk.Euler3DTransform:
    out = sitk.Euler3DTransform(tx)
    out.SetTranslation(tuple(np.add(tx.GetTranslation(), offset)))
    return out


def register(work: Path, s: str) -> tuple[dict, np.ndarray, np.ndarray]:
    fixed = read(work / "b0" / f"{s}.nii.gz")
    moving = read(work / "flair" / f"{s}.nii.gz")
    fixed_brain = read(work / "brain" / f"{s}_bet.nii.gz", sitk.sitkUInt8)
    moving_brain = read(work / "flair_brain" / f"{s}.nii.gz", sitk.sitkUInt8)

    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(fixed_brain)
    tx0 = sitk.Euler3DTransform()
    tx0.SetCenter(stats_filter.GetCentroid(1))

    # 再現性のため seed を固定する（0 は sitkWallClock で毎回変わる）。
    # 相互情報量の並列計算も加算順が揺れるので1スレッドにする
    reg = sitk.ImageRegistrationMethod()
    reg.SetNumberOfThreads(1)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(0.5, seed=20260926)
    reg.SetMetricFixedMask(sitk.BinaryDilate(fixed_brain, [3, 3, 1]))
    reg.SetMetricMovingMask(sitk.BinaryDilate(moving_brain, [6, 6, 1]))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=0.01, numberOfIterations=300, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(tx0, inPlace=False)
    result = reg.Execute(fixed, moving)
    if result.GetName() == "CompositeTransform":
        result = sitk.CompositeTransform(result).GetNthTransform(0)
    return sitk.Euler3DTransform(result), fixed, moving


def acquisition_gap_min(data_root: Path, s: str) -> float:
    """FLAIR と DWI の AcquisitionTime の差（分）。日付は sidecar にないので同日を仮定する。"""
    def minutes(path: Path) -> float:
        h, m, sec = json.loads(path.read_text(encoding="utf-8"))["AcquisitionTime"].split(":")
        return int(h) * 60 + int(m) + float(sec) / 60
    return abs(minutes(data_root / s / "dwi" / f"{s}_dwi.json")
               - minutes(data_root / s / "anat" / f"{s}_FLAIR.json"))


def qc_case(data_root: Path, work: Path, s: str) -> tuple[dict, np.ndarray, np.ndarray]:
    tx, fixed, moving = register(work, s)

    b0_brain = np.asarray(nib.load(work / "brain" / f"{s}_bet.nii.gz").dataobj) > 0
    adc = np.asarray(nib.load(work / f"{s}_adc.nii.gz").dataobj)
    csf_dwi = b0_brain & (adc > CSF_ADC)

    flair = np.asarray(nib.load(work / "flair" / f"{s}.nii.gz").dataobj, dtype=np.float32)
    flair_brain = np.asarray(nib.load(work / "flair_brain" / f"{s}.nii.gz").dataobj) > 0
    csf_flair = flair_brain & (flair < CSF_FLAIR * np.median(flair[flair_brain]))
    like = nib.load(work / "flair" / f"{s}.nii.gz")
    csf_path = work / "flair_brain" / f"{s}_csf.nii.gz"
    save(csf_flair.astype(np.float32), like, csf_path)
    csf_moving = read(csf_path)
    brain_moving = read(work / "flair_brain" / f"{s}.nii.gz")

    def score(t: sitk.Transform) -> tuple[float, float]:
        brain = resample(brain_moving, fixed, t) > 0.5
        csf = resample(csf_moving, fixed, t) > 0.5
        both = brain & b0_brain
        return dice(brain, b0_brain), dice(csf & both, csf_dwi & both)

    identity = sitk.Euler3DTransform()
    identity.SetCenter(tx.GetCenter())
    brain_id, csf_id = score(identity)
    brain_reg, csf_reg = score(tx)
    use_registered = csf_reg >= csf_id - QC_TOLERANCE
    chosen = tx if use_registered else identity
    offsets = [np.eye(3)[i] * sign * SHIFT_MM for i in range(3) for sign in (1, -1)]
    csf_shift = float(np.mean([score(shifted(chosen, o))[1] for o in offsets]))

    physical = (fixed.TransformContinuousIndexToPhysicalPoint([float(v) for v in q])
                for q in np.argwhere(b0_brain)[::50])
    displacement = np.linalg.norm([np.subtract(tx.TransformPoint(p), p) for p in physical], axis=1)

    mask_moving = read(work / "flair_mask" / f"{s}.nii.gz")
    mask_frac = resample(mask_moving, fixed, chosen)
    flair_on_dwi = resample(moving, fixed, chosen)

    flair_vox = abs(np.linalg.det(like.affine[:3, :3])) / 1000.0
    dwi_vox = abs(np.linalg.det(nib.load(work / "b0" / f"{s}.nii.gz").affine[:3, :3])) / 1000.0
    flair_mask = np.asarray(nib.load(work / "flair_mask" / f"{s}.nii.gz").dataobj) > 0
    angles = np.degrees(tx.GetParameters()[:3])
    csf_used = csf_reg if use_registered else csf_id
    gap = acquisition_gap_min(data_root, s)
    row = {
        "subject": s,
        "acquisition_gap_min": gap,
        "brain_dice_header": brain_id, "brain_dice_registered": brain_reg,
        "csf_dice_header": csf_id, "csf_dice_registered": csf_reg,
        f"csf_dice_shift{SHIFT_MM:g}mm": csf_shift,
        "rotation_max_deg": float(np.abs(angles).max()),
        "displacement_median_mm": float(np.median(displacement)),
        "displacement_max_mm": float(displacement.max()),
        "used": "registered" if use_registered else "header",
        "csf_dice_used": csf_used,
        "verified": bool(gap <= SAME_SESSION_MIN and csf_used - csf_shift >= QC_RESPONSE),
        "flair_mask_ml": float(flair_mask.sum() * flair_vox),
        "dwi_mask_ml": float((mask_frac > 0.5).sum() * dwi_vox),
    }
    return row, mask_frac, flair_on_dwi


def stage_case(work: Path, s: str) -> dict:
    ref_img = nib.load(work / "masks" / f"{s}.nii.gz")
    ref = np.asarray(ref_img.dataobj) > 0
    brain = np.asarray(nib.load(work / "brain" / f"{s}_bet.nii.gz").dataobj) > 0
    b1000 = np.asarray(nib.load(work / f"{s}_b1000.nii.gz").dataobj)
    adc = np.asarray(nib.load(work / f"{s}_adc.nii.gz").dataobj)
    voxel_ml = abs(np.linalg.det(ref_img.affine[:3, :3])) / 1000.0

    na, nb = normal_reference(adc, b1000, brain, ref, s)
    lesion = ref & brain
    if not lesion.any():
        return {"subject": s, "volume_ml": float(ref.sum() * voxel_ml), "in_brain_frac": 0.0}
    core = lesion & (adc > 0) & (adc < ADC_CORE)
    restricted = restricted_tissue(adc, b1000, brain, na, nb)
    core_ml = float(core.sum() * voxel_ml)
    core_frac = float(core.sum() / lesion.sum())
    if core_frac >= 0.5:
        imaging_stage = "急性コア優位"
    elif core_ml >= CORE_MIN_ML:
        imaging_stage = "急性コアを一部含む"
    else:
        imaging_stage = "急性コアなし"
    return {
        "subject": s,
        "volume_ml": round(float(ref.sum() * voxel_ml), 3),
        "in_brain_frac": float(lesion.sum() / ref.sum()),
        "normal_adc": round(na, 1),
        "adc_ratio": float(np.median(adc[lesion]) / na),
        "b1000_ratio": float(np.median(b1000[lesion]) / nb),
        "core_ml": round(core_ml, 3),
        "core_frac": core_frac,
        "restricted_frac": float(restricted[lesion].mean()),
        "cavity_frac": float((adc[lesion] > CSF_ADC).mean()),
        "imaging_stage": imaging_stage,
    }


def write_inputs(work: Path, arma_work: Path, s: str) -> None:
    # アームAの7例は arma_senora の入力（int16 で保存、量子化誤差は ADC で最大 0.03）を
    # そのまま使い、同じ症例の予測が2つの報告で食い違わないようにする
    existing = [arma_work / "inputs" / f"{s}_{c}.nii.gz" for c in ("0000", "0001")]
    if all(p.exists() for p in existing):
        for p in existing:
            shutil.copy(p, work / "inputs" / p.name)
        return
    brain = np.asarray(nib.load(work / "brain" / f"{s}_bet.nii.gz").dataobj) > 0
    for channel, name in (("0000", "b1000"), ("0001", "adc")):
        src = nib.load(work / f"{s}_{name}.nii.gz")
        save(np.asarray(src.dataobj, dtype=np.float32) * brain, src,
             work / "inputs" / f"{s}_{channel}.nii.gz")


def draw_registration(work: Path, panels: list, path: Path, background: str) -> None:
    """
    background = "flair": DWI 格子へ写した FLAIR に ADC 由来の CSF（水色）を重ねる。位置合わせの目視用
    background = "adc":   ADC に写したマスク（緑）を重ねる。病期の目視用
    """
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.6 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (s, row, mask_frac, flair_on_dwi) in zip(axes.ravel(), panels):
        adc = np.asarray(nib.load(work / f"{s}_adc.nii.gz").dataobj, dtype=float)
        mask = mask_frac > 0.5
        z = int(np.argmax(mask.sum(axis=(0, 1)))) if mask.any() else adc.shape[2] // 2
        img = flair_on_dwi if background == "flair" else adc
        sl = img[:, :, z]
        vmax = np.percentile(sl[sl > 0], 99.5) if (sl > 0).any() else 1.0
        ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=vmax)
        if background == "flair":
            ax.contour(np.rot90(adc[:, :, z] > CSF_ADC), levels=[0.5], colors="cyan",
                       linewidths=0.6)
        ax.contour(np.rot90(mask[:, :, z]), levels=[0.5], colors="lime", linewidths=1.0)
        ax.set_title(f"{s} z={z}  CSF Dice {row['csf_dice_used']:.2f} ({row['used']})",
                     fontsize=9)
    if background == "flair":
        title = "Arm C: FLAIR resampled to DWI grid; cyan = ADC > 2000 (CSF), green = lesion mask"
    else:
        title = "Arm C: ADC with FLAIR lesion mask mapped to DWI grid (green)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fmt(v, digits=2) -> str:
    return "—" if pd.isna(v) else f"{v:.{digits}f}"


def report(qc: pd.DataFrame, st: pd.DataFrame) -> str:
    lines = ["# SENORA-MRI 23例の画像上の病期", ""]
    out = lines.append
    out("## 1. アームC（FLAIR 上のマスク）の DWI への位置合わせ")
    out("")
    out("QC は変換の推定に使っていない構造で測る（往復 Dice ではない）。")
    out(f"「ずらし」は採用した変換を各軸 ±{SHIFT_MM:g} mm 動かした6通りの CSF Dice の平均で、"
        "指標がずれに反応するかを見る対照。")
    out(f"検証済み = 同じ検査（撮像時刻の差 {SAME_SESSION_MIN:.0f} 分以内）かつ、"
        f"採用した変換の CSF Dice がずらしを {QC_RESPONSE} 以上上回る。")
    out("")
    out("| 症例 | 撮像の差 (分) | 脳 Dice（ヘッダ / 推定） | CSF Dice（ヘッダ / 推定） | 採用 "
        f"| ずらし | 補正量 中位 / 最大 (mm) | 検証 | マスク体積 FLAIR → DWI (mL) |")
    out("|---|---|---|---|---|---|---|---|---|")
    shift_col = f"csf_dice_shift{SHIFT_MM:g}mm"
    for _, r in qc.iterrows():
        out(f"| {r['subject']} | {r['acquisition_gap_min']:.0f} "
            f"| {r['brain_dice_header']:.3f} / {r['brain_dice_registered']:.3f} "
            f"| {r['csf_dice_header']:.3f} / {r['csf_dice_registered']:.3f} | {r['used']} "
            f"| {r[shift_col]:.3f} "
            f"| {r['displacement_median_mm']:.1f} / {r['displacement_max_mm']:.1f} "
            f"| {'済' if r['verified'] else '**未**'} "
            f"| {r['flair_mask_ml']:.1f} → {r['dwi_mask_ml']:.1f} |")
    out("")
    better = int((qc["csf_dice_registered"] > qc["csf_dice_header"]).sum())
    out(f"- 採用した変換で CSF Dice 中位 **{qc['csf_dice_used'].median():.3f}**、"
        f"{SHIFT_MM:g} mm ずらすと {qc[shift_col].median():.3f}")
    out(f"- 剛体補正で CSF Dice が上がった症例: {better} / {len(qc)}"
        f"（補正量の中位 {qc['displacement_median_mm'].median():.1f} mm）")
    out(f"- 検証済み: **{int(qc['verified'].sum())} / {len(qc)}**")
    out(f"- マスク体積の保持率（DWI / FLAIR）中位 "
        f"{(qc['dwi_mask_ml'] / qc['flair_mask_ml']).median():.0%}")
    out("")

    out("## 2. 画像上の病期")
    out("")
    out(f"急性コア = ADC < {ADC_CORE:.0f}、空洞 = ADC > {CSF_ADC:.0f}（× 10⁻⁶ mm²/s）。"
        "割合は脳マスク内の病変に対するもの。")
    out("")
    out("| 症例 | アーム | 臨床区分 | 体積 (mL) | コア (mL) | コアの割合 | 拡散制限の割合 "
        "| 空洞の割合 | ADC 比 | 画像上の病期 |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in st.iterrows():
        stage_label = r.get("imaging_stage", "—")
        if not r["verified"]:
            stage_label = f"（{stage_label}、位置合わせ未検証）"
        out(f"| {r['subject']} | {r['arm']} | {r['presentation'] or '—'} | {r['volume_ml']:.1f} "
            f"| {fmt(r.get('core_ml'), 1)} | {fmt(r.get('core_frac'))} "
            f"| {fmt(r.get('restricted_frac'))} | {fmt(r.get('cavity_frac'))} "
            f"| {fmt(r.get('adc_ratio'))} | {stage_label} |")
    out("")
    excluded = st.loc[~st["verified"], "subject"].tolist()
    st = st[st["verified"]]
    out("### 集計")
    out("")
    if excluded:
        out(f"位置合わせを検証できなかった {', '.join(excluded)} を除く {len(st)} 例。")
        out("")
    table = pd.crosstab(st["imaging_stage"], st["arm"], margins=True, margins_name="計")
    out("| 画像上の病期 | " + " | ".join(str(c) for c in table.columns) + " |")
    out("|---|" + "---|" * len(table.columns))
    for idx, r in table.iterrows():
        out(f"| {idx} | " + " | ".join(str(v) for v in r) + " |")
    out("")
    table = pd.crosstab(st["imaging_stage"], st["presentation"].replace("", "（空欄）"),
                        margins=True, margins_name="計")
    out("| 画像上の病期 \\ 臨床区分 | " + " | ".join(str(c) for c in table.columns) + " |")
    out("|---|" + "---|" * len(table.columns))
    for idx, r in table.iterrows():
        out(f"| {idx} | " + " | ".join(str(v) for v in r) + " |")
    out("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    home = Path.home() / "senora_nnunet"
    parser.add_argument("--data", type=Path, default=root / "data" / "senora_normalized")
    parser.add_argument("--work", type=Path, default=home / "senora_all")
    parser.add_argument("--armc-work", type=Path, default=home / "armc_senora")
    parser.add_argument("--arma-work", type=Path, default=home / "arma_senora")
    parser.add_argument("--out", type=Path, default=root / "results" / "staging")
    args = parser.parse_args()

    cases = prepare(args.data, args.work, args.armc_work, args.arma_work)
    participants = pd.read_csv(args.data / "participants.tsv", sep="\t")
    presentation = {str(r["participant_id"]): ("" if pd.isna(r["presentation_status"])
                                               else str(r["presentation_status"]).strip().lower())
                    for _, r in participants.iterrows()}

    qc_rows, panels = [], []
    for s in cases.loc[cases["reference"] == "FLAIR", "subject"]:
        row, mask_frac, flair_on_dwi = qc_case(args.data, args.work, s)
        like = nib.load(args.work / "b0" / f"{s}.nii.gz")
        save((mask_frac > 0.5).astype(np.uint8), like, args.work / "masks" / f"{s}.nii.gz")
        qc_rows.append(row)
        panels.append((s, row, mask_frac, flair_on_dwi))
        print(f"{s}: CSF Dice {row['csf_dice_header']:.3f} → {row['csf_dice_registered']:.3f}"
              f"（採用 {row['used']}、ずらし {row[f'csf_dice_shift{SHIFT_MM:g}mm']:.3f}）、"
              f"補正 {row['displacement_median_mm']:.1f} mm、検証 {row['verified']}")
    qc = pd.DataFrame(qc_rows)
    verified = dict(zip(qc["subject"], qc["verified"]))

    st_rows = []
    for _, c in cases.iterrows():
        s = c["subject"]
        write_inputs(args.work, args.arma_work, s)
        row = stage_case(args.work, s)
        row["arm"] = "A" if c["reference"] == "DWI" else "C"
        row["presentation"] = presentation.get(s, "")
        row["verified"] = bool(verified.get(s, True))
        st_rows.append(row)
    st = pd.DataFrame(st_rows).sort_values(["arm", "subject"]).reset_index(drop=True)

    args.out.mkdir(parents=True, exist_ok=True)
    qc.to_csv(args.out / "registration_qc.csv", index=False)
    st.to_csv(args.out / "lesion_stage.csv", index=False)
    draw_registration(args.work, panels, args.out / "armc_registration_overlay.png", "flair")
    draw_registration(args.work, panels, args.out / "armc_stage_overlay.png", "adc")
    text = report(qc, st)
    (args.out / "lesion_stage.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
