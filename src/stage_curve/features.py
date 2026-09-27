"""病変単位の特徴量。閾値は事前登録（docs/prereg-stage-curve.md）と config から受け取る。"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

CONNECTIVITY_26 = ndimage.generate_binary_structure(3, 3)
STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)

COLUMNS = [
    "lesion_id", "patient_id", "dataset", "vol_ml", "log_vol_c", "adc_normal",
    "adc_ratio_median", "adc_ratio_p10", "adc_ratio_p90", "adc_ratio_eroded",
    "core_fraction", "dwi_ratio_median", "overlap_frac", "detected", "detected_any",
    "detected_half", "infratentorial", "mask_sequence", "clinical_stage", "qc_flag",
]


@dataclass(frozen=True)
class Thresholds:
    overlap_detected: float = 0.10
    overlap_half: float = 0.5
    min_volume_ml: float = 0.1
    dilation_voxels: int = 3
    erosion_min_voxels: int = 27
    inferior_fraction: float = 0.30
    adc_core: float = 620.0
    adc_noise: float = 200.0
    adc_csf: float = 1500.0
    adc_normal_qc: tuple[float, float] = (600.0, 1000.0)
    restricted_b1000: float = 1.3
    unlabeled_whole_adc_range: tuple[float, float] = (500.0, 1500.0)
    unlabeled_min_volume_ml: float = 0.5


def voxel_volume_ml(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3])) / 1000.0)


def assert_same_grid(reference: nib.Nifti1Image, *others: nib.Nifti1Image) -> None:
    """shape と affine が一致しなければ止める。ADC はリサンプリングしない。"""
    ref_shape = reference.shape[:3]
    ref_affine = np.asarray(reference.affine, dtype=float)
    for other in others:
        if other.shape[:3] != ref_shape or not np.allclose(other.affine, ref_affine, rtol=1e-4, atol=1e-3):
            raise ValueError(
                f"グリッドが一致しない: shape {other.shape[:3]} vs {ref_shape}"
            )


def adc_scale_to_1e6(raw_median: float, bands: list[tuple[float, float, float]]) -> float | None:
    """変換前の中央値が入った帯の倍率を返す。重なる帯や該当なしは None。"""
    hits = [scale for low, high, scale in bands if low <= raw_median <= high]
    if len(hits) == 1:
        return hits[0]
    return None


def inferior_slice_mask(brain: np.ndarray, fraction: float) -> np.ndarray:
    """脳が存在する z 範囲の下側 fraction を True にする。"""
    z = np.where(brain.any(axis=(0, 1)))[0]
    below = np.zeros(brain.shape, dtype=bool)
    if len(z) == 0 or z.max() == z.min():
        return below
    z_cut = z.min() + fraction * (z.max() - z.min())
    below[:, :, np.arange(brain.shape[2]) < z_cut] = True
    return below


def z_cut_of(brain: np.ndarray, fraction: float) -> float:
    z = np.where(brain.any(axis=(0, 1)))[0]
    if len(z) == 0:
        return 0.0
    return float(z.min() + fraction * (z.max() - z.min()))


def dilate(mask: np.ndarray, voxels: int) -> np.ndarray:
    if voxels <= 0 or not mask.any():
        return mask.astype(bool)
    return ndimage.binary_dilation(mask.astype(bool), structure=STRUCTURE_6, iterations=voxels)


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=CONNECTIVITY_26)


def core_clusters(adc_1e6: np.ndarray, b1000: np.ndarray, brain: np.ndarray,
                  normal_b1000: float, voxel_ml: float, thresholds: Thresholds) -> np.ndarray:
    """09/27 に固定した急性コアの塊（0.5 mL 以上、下 30% を除く）。"""
    candidate = (brain & (adc_1e6 > 0) & (adc_1e6 < thresholds.adc_core)
                 & (b1000 > thresholds.restricted_b1000 * normal_b1000))
    z_cut = z_cut_of(brain, thresholds.inferior_fraction)
    labels, count = component_labels(candidate)
    kept = np.zeros(labels.shape, dtype=np.int32)
    next_id = 0
    for label in range(1, count + 1):
        component = labels == label
        if component.sum() * voxel_ml < thresholds.unlabeled_min_volume_ml:
            continue
        if ndimage.center_of_mass(component)[2] < z_cut:
            continue
        next_id += 1
        kept[component] = next_id
    return kept


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def lesion_rows(adc_1e6: np.ndarray, b1000: np.ndarray, brain: np.ndarray,
                reference: np.ndarray, prediction: np.ndarray, voxel_ml: float,
                thresholds: Thresholds, dataset: str, subject: str,
                mask_sequence: str, clinical_stage: str,
                scale_unknown: bool) -> list[dict]:
    """1症例の参照を連結成分に分け、1行1病変の辞書を返す。"""
    brain = brain.astype(bool)
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)
    if not (adc_1e6.shape == b1000.shape == brain.shape == reference.shape == prediction.shape):
        raise ValueError(f"{subject}: 配列の shape が一致しない")

    below = inferior_slice_mask(brain, thresholds.inferior_fraction)
    z_cut = z_cut_of(brain, thresholds.inferior_fraction)
    exclude = dilate(reference | prediction, thresholds.dilation_voxels)
    parenchyma = (brain & ~exclude & ~below
                  & (adc_1e6 > thresholds.adc_noise) & (adc_1e6 < thresholds.adc_csf))
    if parenchyma.any() and np.isfinite(_median(b1000[parenchyma])):
        rough_b = _median(b1000[parenchyma])
        clusters = core_clusters(adc_1e6, b1000, brain, rough_b, voxel_ml, thresholds)
        exclude = dilate(reference | prediction | (clusters > 0), thresholds.dilation_voxels)
        parenchyma = (brain & ~exclude & ~below
                      & (adc_1e6 > thresholds.adc_noise) & (adc_1e6 < thresholds.adc_csf))
    adc_normal = _median(adc_1e6[parenchyma]) if parenchyma.any() else float("nan")
    b1000_normal = _median(b1000[parenchyma]) if parenchyma.any() else float("nan")

    flags = []
    low, high = thresholds.adc_normal_qc
    if not (low <= adc_normal <= high):
        flags.append("adc_normal_out_of_range")
    if scale_unknown:
        flags.append("adc_unit_unknown")
    qc = ";".join(flags)

    labels, count = component_labels(reference)
    rows = []
    patient_id = f"{dataset}_{subject}"
    serial = 0
    for label in range(1, count + 1):
        component = labels == label
        volume = float(component.sum() * voxel_ml)
        if volume < thresholds.min_volume_ml:
            continue
        serial += 1
        adc_values = adc_1e6[component]
        overlap = float((prediction & component).sum() / component.sum())
        eroded = ndimage.binary_erosion(component, structure=STRUCTURE_6, iterations=1)
        if int(eroded.sum()) < thresholds.erosion_min_voxels:
            eroded_ratio = float("nan")
        else:
            eroded_ratio = _median(adc_1e6[eroded]) / adc_normal
        rows.append({
            "lesion_id": f"{dataset}_{subject}_{serial:03d}",
            "patient_id": patient_id,
            "dataset": dataset,
            "vol_ml": volume,
            "log_vol_c": float(np.log(volume)),
            "adc_normal": adc_normal,
            "adc_ratio_median": _median(adc_values) / adc_normal,
            "adc_ratio_p10": float(np.percentile(adc_values, 10)) / adc_normal,
            "adc_ratio_p90": float(np.percentile(adc_values, 90)) / adc_normal,
            "adc_ratio_eroded": eroded_ratio,
            "core_fraction": float("nan") if scale_unknown else float((adc_values < thresholds.adc_core).mean()),
            "dwi_ratio_median": _median(b1000[component]) / b1000_normal,
            "overlap_frac": overlap,
            "detected": int(overlap >= thresholds.overlap_detected),
            "detected_any": int(overlap > 0),
            "detected_half": int(overlap >= thresholds.overlap_half),
            "infratentorial": int(ndimage.center_of_mass(component)[2] < z_cut),
            "mask_sequence": mask_sequence,
            "clinical_stage": clinical_stage,
            "qc_flag": qc,
        })
    return rows


def empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)
