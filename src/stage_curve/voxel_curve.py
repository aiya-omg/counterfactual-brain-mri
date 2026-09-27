"""病変内ボクセルを ADC 比のビンに分け、予測に含まれる割合を出す。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.stage_curve.adapters import ADAPTERS
from src.stage_curve.extract_lesions import load_config, repo_root, thresholds_from
from src.stage_curve.features import component_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo_root() / "config" / "stage_curve.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    thresholds = thresholds_from(config)
    root = repo_root()
    lesions = pd.read_csv(root / config["output_dir"] / "lesions.csv")
    normal = lesions.drop_duplicates("patient_id").set_index("patient_id")["adc_normal"]
    edges = np.arange(config["voxel_ratio_min"], config["voxel_ratio_max"] + 1e-9,
                      config["voxel_bin_width"])
    # 患者 × ビンの（ボクセル数、予測数）
    counts: dict[str, np.ndarray] = {}
    predicted: dict[str, np.ndarray] = {}
    # マスクなし症例の塊は ADC で選んであるので、主解析と同じ2データだけを使う。
    for name in ("ISLES", "SENORA_masked"):
        loaded, _ = ADAPTERS[name]().load(config, thresholds)
        for case in loaded:
            patient = f"{case.dataset}_{case.subject}"
            if patient not in normal.index or not np.isfinite(normal.loc[patient]) or normal.loc[patient] <= 0:
                continue
            labels, count = component_labels(case.reference)
            keep = np.zeros(case.reference.shape, dtype=bool)
            for label in range(1, count + 1):
                component = labels == label
                if component.sum() * case.voxel_ml >= thresholds.min_volume_ml:
                    keep |= component
            if not keep.any():
                continue
            ratio = case.adc_1e6[keep] / float(normal.loc[patient])
            bins = np.digitize(ratio, edges) - 1
            valid = (bins >= 0) & (bins < len(edges) - 1) & np.isfinite(ratio)
            hit = case.prediction[keep][valid]
            bins = bins[valid]
            total = np.bincount(bins, minlength=len(edges) - 1).astype(float)
            hits = np.bincount(bins, weights=hit.astype(float), minlength=len(edges) - 1)
            counts[patient] = total
            predicted[patient] = hits
    patients = sorted(counts)
    rng = np.random.default_rng(int(config["seed"]))
    draws = int(config["bootstrap_draws"])
    point = np.vstack([predicted[p] for p in patients]).sum(axis=0)
    base = np.vstack([counts[p] for p in patients]).sum(axis=0)
    rate = np.divide(point, base, out=np.full_like(point, np.nan), where=base > 0)
    boot = np.full((draws, len(rate)), np.nan)
    index = np.arange(len(patients))
    for draw in range(draws):
        chosen = rng.choice(index, size=len(patients), replace=True)
        total = np.zeros_like(rate)
        hits = np.zeros_like(rate)
        for i in chosen:
            total += counts[patients[i]]
            hits += predicted[patients[i]]
        boot[draw] = np.divide(hits, total, out=np.full_like(hits, np.nan), where=total > 0)
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    centers = (edges[:-1] + edges[1:]) / 2
    pd.DataFrame({
        "adc_ratio_center": centers,
        "voxel_fraction_predicted": rate,
        "ci_low": low,
        "ci_high": high,
        "voxels": base.astype(int),
    }).to_csv(root / config["output_dir"] / "voxel_curve.csv", index=False)
    print(f"ビン {len(centers)} → {root / config['output_dir'] / 'voxel_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
