"""検出曲線、ボクセル曲線、ADC 比の分布、QC 図。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from src.stage_curve.adapters import ADAPTERS
from src.stage_curve.extract_lesions import load_config, repo_root, thresholds_from
from src.stage_curve.features import component_labels


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def main_figure(lesions: pd.DataFrame, draws: pd.DataFrame, out: Path, seed: int) -> None:
    main = lesions[lesions["dataset"].isin(["ISLES", "SENORA_masked"])].copy()
    grid = np.logspace(np.log10(0.4), np.log10(3.5), 200)
    log_grid = np.log(grid)
    curves = _logistic(draws["intercept"].to_numpy()[:, None]
                       + draws["log_adc"].to_numpy()[:, None] * log_grid)
    center = np.nanmedian(curves, axis=0)
    low, high = np.nanpercentile(curves, [2.5, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(grid, low, high, color="#9bb8d3", alpha=0.7, label="95% CI")
    ax.plot(grid, center, color="#1f4e79", label="1 mL")
    rng = np.random.default_rng(seed)
    colors = {"ISLES": "#d4762c", "SENORA_masked": "#2f6f4e"}
    for dataset, group in main.groupby("dataset"):
        y = group["detected"].to_numpy(dtype=float) + rng.normal(0, 0.015, len(group))
        ax.scatter(group["adc_ratio_median"], y, s=12, alpha=0.45, c=colors[dataset],
                   label=dataset, linewidths=0)
    if "ed50" in draws.columns and np.isfinite(draws["ed50"]).any():
        ed50 = float(np.nanmedian(draws["ed50"]))
        ed90 = float(np.nanmedian(draws["ed90"]))
        ax.axvline(ed50, color="#1f4e79", ls="--", lw=1, label=f"ED50 {ed50:.2f}")
        ax.axvline(ed90, color="#1f4e79", ls=":", lw=1, label=f"ED90 {ed90:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("ADC ratio")
    ax.set_ylabel("Detection probability")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "detection_curve.png", dpi=150)
    plt.close(fig)


def voxel_figure(curve: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(curve["adc_ratio_center"], curve["ci_low"], curve["ci_high"], color="#9bb8d3")
    ax.plot(curve["adc_ratio_center"], curve["voxel_fraction_predicted"], color="#1f4e79")
    ax.set_xlabel("ADC ratio")
    ax.set_ylabel("Fraction of voxels predicted")
    fig.tight_layout()
    fig.savefig(out / "voxel_curve.png", dpi=150)
    plt.close(fig)


def histogram(lesions: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for dataset, group in lesions.groupby("dataset"):
        ax.hist(group["adc_ratio_median"].dropna(), bins=30, histtype="step", density=True, label=dataset)
    ax.axvspan(0.8, 1.5, color="#dddddd", zorder=0)
    ax.set_xlabel("ADC ratio")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "adc_ratio_hist.png", dpi=150)
    plt.close(fig)


def qc_figure(lesions: pd.DataFrame, config: dict, out: Path) -> None:
    thresholds = thresholds_from(config)
    chosen = lesions.sample(n=min(int(config["qc_figures"]), len(lesions)),
                            random_state=int(config["seed"]))
    wanted = set(chosen["lesion_id"])
    panels = {}
    for name in ("ISLES", "SENORA_masked", "SENORA_cluster"):
        loaded, _ = ADAPTERS[name]().load(config, thresholds)
        for case in loaded:
            labels, count = component_labels(case.reference)
            serial = 0
            for label in range(1, count + 1):
                component = labels == label
                if component.sum() * case.voxel_ml < thresholds.min_volume_ml:
                    continue
                serial += 1
                lesion_id = f"{case.dataset}_{case.subject}_{serial:03d}"
                if lesion_id not in wanted:
                    continue
                z = int(np.clip(round(ndimage.center_of_mass(component)[2]), 0, component.shape[2] - 1))
                row = chosen.loc[chosen["lesion_id"] == lesion_id].iloc[0]
                panels[lesion_id] = (case.b1000[:, :, z], component[:, :, z], case.prediction[:, :, z],
                                     f"{lesion_id}  ratio {row['adc_ratio_median']:.2f}  det {int(row['detected'])}")
    n = len(panels)
    if n == 0:
        return
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, lesion_id in zip(axes, chosen["lesion_id"]):
        ax.axis("off")
        if lesion_id not in panels:
            continue
        image, mask, pred, title = panels[lesion_id]
        show = image.T
        ax.imshow(show, cmap="gray", origin="lower")
        if mask.any():
            ax.contour(mask.T, levels=[0.5], colors="#d14b4b", linewidths=0.6)
        if pred.any():
            ax.contour(pred.T, levels=[0.5], colors="#e0b000", linewidths=0.6)
        ax.set_title(title, fontsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "qc_lesions.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo_root() / "config" / "stage_curve.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = repo_root()
    out = root / config["output_dir"]
    lesions = pd.read_csv(out / "lesions.csv")
    draws = pd.read_csv(out / "coef_draws.csv")
    main_figure(lesions, draws, out, int(config["seed"]))
    voxel = out / "voxel_curve.csv"
    if voxel.exists():
        voxel_figure(pd.read_csv(voxel), out)
    histogram(lesions, out)
    qc_figure(lesions, config, out)
    print(f"図 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
