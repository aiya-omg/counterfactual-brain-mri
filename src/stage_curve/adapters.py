"""データセットごとのパス解決。SOOP と JHU は枠だけ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.stage_curve.features import (
    Thresholds,
    adc_scale_to_1e6,
    assert_same_grid,
    core_clusters,
    voxel_volume_ml,
)


@dataclass(frozen=True)
class LoadedCase:
    dataset: str
    subject: str
    adc_1e6: np.ndarray
    b1000: np.ndarray
    brain: np.ndarray
    reference: np.ndarray
    prediction: np.ndarray
    voxel_ml: float
    mask_sequence: str
    clinical_stage: str
    scale_unknown: bool
    affine: np.ndarray


class DatasetAdapter:
    """後から SOOP や JHU を足すための入口。onset_to_mri_hours は今回は読まない。"""

    name: str

    def load(self, config: dict, thresholds: Thresholds) -> tuple[list[LoadedCase], list[dict]]:
        raise NotImplementedError


def _image(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(path)


def _array(image: nib.Nifti1Image) -> np.ndarray:
    return np.asarray(image.dataobj)


def require_path(value, label: str) -> Path:
    if value is None or (isinstance(value, str) and value.strip().upper() == "TODO"):
        raise SystemExit(
            f"{label} が TODO のままです。パスを config/stage_curve.yaml に入れてから再実行してください。"
        )
    return Path(value)


def _scale_adc(adc: np.ndarray, brain: np.ndarray, bands: list[tuple[float, float, float]]) -> tuple[np.ndarray, bool]:
    tissue = brain & (adc > 0)
    raw_median = float(np.median(adc[tissue])) if tissue.any() else float("nan")
    scale = adc_scale_to_1e6(raw_median, bands)
    if scale is None:
        return adc, True
    return adc * scale, False


class IslesAdapter(DatasetAdapter):
    """ISLES は各 fold の検証予測だけを使う。アンサンブルは学習症例へのリークになる。"""

    name = "ISLES"

    def load(self, config: dict, thresholds: Thresholds) -> tuple[list[LoadedCase], list[dict]]:
        cfg = config["isles"]
        images = Path(cfg["images_dir"])
        labels = Path(cfg["labels_dir"])
        results = Path(cfg["results_dir"])
        brain_dir = require_path(cfg.get("brain_dir"), "isles.brain_dir")
        bands = [tuple(row) for row in config["adc_scale_bands"]]
        predictions = {}
        for fold_dir in sorted(results.glob("fold_*")):
            validation = fold_dir / "validation"
            if not validation.is_dir():
                continue
            for path in validation.glob("ISLES_*.nii.gz"):
                if path.stem in predictions:
                    raise ValueError(f"{path.stem} が複数の fold の検証予測にある")
                predictions[path.stem] = path
        cases, notes = [], []
        for label_path in sorted(labels.glob("ISLES_*.nii.gz")):
            subject = label_path.name.replace(".nii.gz", "")
            prediction = predictions.get(subject)
            if prediction is None:
                raise FileNotFoundError(f"{subject} の out-of-fold 予測がない")
            dwi_img = _image(images / f"{subject}_0000.nii.gz")
            adc_img = _image(images / f"{subject}_0001.nii.gz")
            label_img = _image(label_path)
            pred_img = _image(prediction)
            brain_name = cfg.get("brain_name", "{subject}_brain.nii.gz").format(subject=subject)
            brain_img = _image(brain_dir / brain_name)
            assert_same_grid(dwi_img, adc_img, label_img, pred_img, brain_img)
            brain = _array(brain_img) > 0
            adc, unknown = _scale_adc(_array(adc_img).astype(np.float64), brain, bands)
            cases.append(LoadedCase(
                self.name, subject, adc, _array(dwi_img).astype(np.float64), brain,
                _array(label_img) > 0, _array(pred_img) > 0,
                voxel_volume_ml(dwi_img.affine), "", "", unknown,
                np.asarray(dwi_img.affine, dtype=float),
            ))
        note = {"dataset": self.name, "cases": len(cases),
                "prediction": "fold_*/validation の out-of-fold。アンサンブルは使っていない。"}
        final_summary = results / "fold_0" / "summary_final_checkpoint.json"
        fold0_nifti = next((results / "fold_0" / "validation").glob("ISLES_*.nii.gz"), None)
        if final_summary.exists() and fold0_nifti is not None and fold0_nifti.stat().st_mtime > final_summary.stat().st_mtime:
            note["fold_0_weights"] = (
                "fold_0/validation の NIfTI は summary_final_checkpoint.json より新しい。"
                "fold 0 だけ best 重みの予測で上書きされている。fold 1〜4 は最終重み。"
            )
        notes.append(note)
        return cases, notes


class SenoraMaskedAdapter(DatasetAdapter):
    name = "SENORA_masked"

    def load(self, config: dict, thresholds: Thresholds) -> tuple[list[LoadedCase], list[dict]]:
        cfg = config["senora_masked"]
        work = Path(cfg["work_dir"])
        pred_dir = Path(cfg["predictions_dir"])
        excluded = set(cfg.get("exclude") or [])
        report = pd.read_csv(config_path(cfg["segmentation_report"]))
        participants = pd.read_csv(config_path(cfg["participants"]), sep="\t")
        stage = participants.set_index("participant_id")["presentation_status"].to_dict()
        sequence = report.set_index("subject_id")["sequence"].to_dict()
        bands = [tuple(row) for row in config["adc_scale_bands"]]
        cases = []
        for mask_path in sorted((work / "masks").glob("sub-*.nii.gz")):
            subject = mask_path.name.replace(".nii.gz", "")
            if subject in excluded:
                continue
            dwi_img = _image(work / f"{subject}_b1000.nii.gz")
            adc_img = _image(work / f"{subject}_adc.nii.gz")
            brain_img = _image(work / "brain" / f"{subject}_bet.nii.gz")
            pred_img = _image(pred_dir / f"{subject}.nii.gz")
            mask_img = _image(mask_path)
            assert_same_grid(dwi_img, adc_img, brain_img, pred_img, mask_img)
            brain = _array(brain_img) > 0
            adc, unknown = _scale_adc(_array(adc_img).astype(np.float64), brain, bands)
            cases.append(LoadedCase(
                self.name, subject, adc, _array(dwi_img).astype(np.float64), brain,
                _array(mask_img) > 0, _array(pred_img) > 0,
                voxel_volume_ml(dwi_img.affine),
                str(sequence.get(subject, "")).strip().upper(),
                _stage(stage.get(subject)), unknown,
                np.asarray(dwi_img.affine, dtype=float),
            ))
        return cases, [{"dataset": self.name, "cases": len(cases), "excluded": sorted(excluded)}]


class SenoraClusterAdapter(DatasetAdapter):
    """マスクのない虚血性症例。参照は 302daae で固定した急性コアの塊。主解析には入れない。"""

    name = "SENORA_cluster"

    def load(self, config: dict, thresholds: Thresholds) -> tuple[list[LoadedCase], list[dict]]:
        cfg = config["senora_cluster"]
        work = Path(cfg["work_dir"])
        pred_dir = Path(cfg["predictions_dir"])
        excluded = set(cfg.get("exclude") or [])
        participants = pd.read_csv(config_path(cfg["participants"]), sep="\t")
        ischemic = set(participants.loc[
            participants["stroke_subtype"].astype(str) == "ischemic", "participant_id"].astype(str))
        stage = participants.set_index("participant_id")["presentation_status"].to_dict()
        bands = [tuple(row) for row in config["adc_scale_bands"]]
        low, high = thresholds.unlabeled_whole_adc_range
        cases, skipped = [], []
        for adc_path in sorted(work.glob("sub-*_adc.nii.gz")):
            subject = adc_path.name.replace("_adc.nii.gz", "")
            if subject in excluded or subject not in ischemic:
                continue
            prediction_path = pred_dir / f"{subject}.nii.gz"
            if not prediction_path.exists():
                skipped.append({"subject": subject, "reason": "予測がない"})
                continue
            dwi_img = _image(work / f"{subject}_b1000.nii.gz")
            adc_img = _image(adc_path)
            brain_img = _image(work / "brain" / f"{subject}_bet.nii.gz")
            pred_img = _image(prediction_path)
            assert_same_grid(dwi_img, adc_img, brain_img, pred_img)
            brain = _array(brain_img) > 0
            raw_adc = _array(adc_img).astype(np.float64)
            whole = float(np.median(raw_adc[brain])) if brain.any() else float("nan")
            if not (low <= whole <= high):
                skipped.append({"subject": subject, "reason": f"ADC 単位確認で除外（脳全体の中央値 {whole:.4g}）"})
                continue
            adc, unknown = _scale_adc(raw_adc, brain, bands)
            b1000 = _array(dwi_img).astype(np.float64)
            parenchyma = brain & (adc > thresholds.adc_noise) & (adc < thresholds.adc_csf)
            normal_b = float(np.median(b1000[parenchyma])) if parenchyma.any() else float("nan")
            reference = core_clusters(
                adc, b1000, brain, normal_b, voxel_volume_ml(dwi_img.affine), thresholds) > 0
            cases.append(LoadedCase(
                self.name, subject, adc, b1000, brain, reference, _array(pred_img) > 0,
                voxel_volume_ml(dwi_img.affine), "cluster", _stage(stage.get(subject)), unknown,
                np.asarray(dwi_img.affine, dtype=float),
            ))
        return cases, [{"dataset": self.name, "cases": len(cases), "skipped": skipped}]


class PendingAdapter(DatasetAdapter):
    """SOOP と JHU。発症からの時間があるデータでは onset_to_mri_hours を後で足す。"""

    def __init__(self, name: str):
        self.name = name

    def load(self, config: dict, thresholds: Thresholds) -> tuple[list[LoadedCase], list[dict]]:
        raise NotImplementedError(f"{self.name} は今回は実装しない")


def config_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _stage(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    return text


ADAPTERS = {
    "ISLES": IslesAdapter,
    "SENORA_masked": SenoraMaskedAdapter,
    "SENORA_cluster": SenoraClusterAdapter,
}
