"""病変特徴量を CSV にする。既存の予測とマスクは読むだけ。"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.stage_curve.adapters import ADAPTERS
from src.stage_curve.features import COLUMNS, Thresholds, empty_table, lesion_rows


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def thresholds_from(config: dict) -> Thresholds:
    low, high = config["adc_normal_qc"]
    whole_low, whole_high = config["unlabeled_whole_adc_range"]
    return Thresholds(
        overlap_detected=float(config["overlap_detected"]),
        overlap_half=float(config["overlap_half"]),
        min_volume_ml=float(config["min_volume_ml"]),
        dilation_voxels=int(config["dilation_voxels"]),
        erosion_min_voxels=int(config["erosion_min_voxels"]),
        inferior_fraction=float(config["inferior_fraction"]),
        adc_core=float(config["adc_core"]),
        adc_noise=float(config["adc_noise"]),
        adc_csf=float(config["adc_csf"]),
        adc_normal_qc=(float(low), float(high)),
        restricted_b1000=float(config["restricted_b1000"]),
        unlabeled_whole_adc_range=(float(whole_low), float(whole_high)),
        unlabeled_min_volume_ml=float(config["unlabeled_min_volume_ml"]),
    )


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def qc_summary(table: pd.DataFrame) -> dict:
    summary = {}
    for dataset, group in table.groupby("dataset"):
        normal = group.drop_duplicates("patient_id")["adc_normal"]
        summary[dataset] = {
            "lesions": int(len(group)),
            "patients": int(group["patient_id"].nunique()),
            "adc_normal_median": float(normal.median()),
            "adc_normal_min": float(normal.min()),
            "adc_normal_max": float(normal.max()),
            "detected_fraction": float(group["detected"].mean()),
        }
    isles = table[table["dataset"] == "ISLES"]
    if len(isles):
        summary["isles_detected_fraction"] = float(isles["detected"].mean())
    chronic = table[(table["dataset"] == "SENORA_masked") & (table["clinical_stage"] == "chronic")]
    if len(chronic):
        median = float(chronic["adc_ratio_median"].median())
        summary["senora_chronic_adc_ratio_median"] = median
        summary["senora_chronic_adc_ratio_min"] = float(chronic["adc_ratio_median"].min())
        summary["senora_chronic_adc_ratio_max"] = float(chronic["adc_ratio_median"].max())
        # 空洞化した陳旧性梗塞は正常実質の 2.1〜3.3 倍だった。中央値がその付近にあれば矛盾しない。
        summary["senora_chronic_consistent_with_2.1_to_3.3"] = bool(1.5 <= median <= 4.0)
    transition = table[table["dataset"].isin(["ISLES", "SENORA_masked"])]
    band = transition["adc_ratio_median"].between(0.8, 1.5)
    summary["transition_lesions_adc_ratio_0.8_to_1.5"] = int(band.sum())
    summary["transition_by_dataset"] = (
        transition.loc[band, "dataset"].value_counts().to_dict() if band.any() else {})
    return summary


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo_root() / "config" / "stage_curve.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    thresholds = thresholds_from(config)
    root = repo_root()
    rows: list[dict] = []
    notes: list[dict] = []
    for name in ("ISLES", "SENORA_masked", "SENORA_cluster"):
        loaded, note = ADAPTERS[name]().load(config, thresholds)
        notes.extend(note)
        for case in loaded:
            rows.extend(lesion_rows(
                case.adc_1e6, case.b1000, case.brain, case.reference, case.prediction,
                case.voxel_ml, thresholds, case.dataset, case.subject,
                case.mask_sequence, case.clinical_stage, case.scale_unknown,
            ))
    table = pd.DataFrame(rows, columns=COLUMNS) if rows else empty_table()
    out = root / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "lesions.csv", index=False)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(root),
        "config": config,
        "notes": notes,
        "qc": qc_summary(table) if len(table) else {},
        "lesions": int(len(table)),
        "patients": int(table["patient_id"].nunique()) if len(table) else 0,
        "nonpositive_adc_ratio": int((~np.isfinite(table["adc_ratio_median"])
                                      | (table["adc_ratio_median"] <= 0)).sum()) if len(table) else 0,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    print(f"{len(table)} 病変 → {out / 'lesions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
