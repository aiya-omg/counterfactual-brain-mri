"""
batch_evaluate.py
demo_output/ 内の全フォルダに対して Dice/IoU/Precision/Recall を遡り計算し、
CSV と updated result_analysis.png に保存する。

使い方:
  python batch_evaluate.py --gt_mask ./data/slices/masks/BraTS-GLI-00000-000_slice055.png
  python batch_evaluate.py --gt_mask ./data/slices/masks/BraTS-GLI-00000-000_slice055.png --output_dir ./demo_output --update_images
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu as sk_otsu


# ── 評価関数 ──────────────────────────────────────────────────────────────────

def threshold_otsu(diff_map: Image.Image) -> np.ndarray:
    arr = np.array(diff_map.convert("L")).astype(float) / 255.0
    try:
        thresh = sk_otsu(arr)
    except Exception:
        thresh = 0.5
    return arr >= thresh


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    return {
        "Dice":      round(float(2*tp / (2*tp+fp+fn+1e-8)), 4),
        "IoU":       round(float(tp   / (tp+fp+fn+1e-8)),   4),
        "Precision": round(float(tp   / (tp+fp+1e-8)),      4),
        "Recall":    round(float(tp   / (tp+fn+1e-8)),      4),
    }


def apply_heatmap_overlay(original: Image.Image, diff_map: Image.Image, alpha=0.55):
    orig = np.array(original.convert("RGB")).astype(float)
    diff = np.array(diff_map.convert("L")).astype(float)
    thr  = np.percentile(diff[diff > 0], 70) if diff.max() > 0 else 0
    diff_m = np.where(diff >= thr, diff, 0)
    cmap   = plt.get_cmap("jet")
    dn     = diff_m / diff_m.max() if diff_m.max() > 0 else diff_m
    hm     = (cmap(dn)[:, :, :3] * 255).astype(float)
    mask   = (diff_m > 0).astype(float)[:, :, np.newaxis]
    return Image.fromarray((orig*(1-alpha*mask) + hm*alpha*mask).astype(np.uint8))


def parse_folder_info(folder_name: str) -> dict:
    """フォルダ名からモデル名とパラメータを解析"""
    info = {"model": "unknown", "t_noise": None, "t_start": None}
    if "Flow-Matching" in folder_name:
        info["model"] = "Flow-Matching"
        m = re.search(r"_t([\d.]+)$", folder_name)
        if m:
            info["t_start"] = float(m.group(1))
    elif "MONAI-DDPM" in folder_name:
        info["model"] = "MONAI-DDPM"
        m = re.search(r"_t(\d+)$", folder_name)
        if m:
            info["t_noise"] = int(m.group(1))
    elif "simulation" in folder_name:
        info["model"] = "simulation"
    return info


def update_result_image(run_dir: Path, gt_mask: Image.Image, metrics: dict, params: dict):
    """result_analysis.png にメトリクスを上書きして再保存"""
    orig  = Image.open(run_dir / "01_original.png").convert("RGB")
    cf    = Image.open(run_dir / "02_counterfactual.png").convert("RGB")
    diff  = Image.open(run_dir / "03_difference_map.png").convert("L")
    overlay = apply_heatmap_overlay(orig, diff)

    gt_resized = gt_mask.resize(orig.size, Image.NEAREST)
    gt_arr   = np.array(gt_resized.convert("L"))
    diff_arr = np.array(diff)

    ncols = 5
    fig, axes = plt.subplots(1, ncols, figsize=(25, 5.6))
    fig.patch.set_facecolor("#1a1a2e")

    model  = params.get("model", "")
    t_info = (f"t_noise={params['t_noise']}" if params.get("t_noise") is not None
              else f"t_start={params['t_start']}" if params.get("t_start") is not None
              else "")
    fig.suptitle(f"Counterfactual Visual Attribution - Brain Tumor MRI  |  model={model}  /  {t_info}",
                 color="white", fontsize=11, y=1.01)

    panels = [
        (np.array(orig),    "Original (Tumor MRI)",    None),
        (np.array(cf),      "Counterfactual (Normal)",  None),
        (diff_arr,          "Difference Map",            "hot"),
        (np.array(overlay), "Lesion Highlight",          None),
        (gt_arr,            "GT Segmentation",           "gray"),
    ]

    for ax, (img, lbl, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(lbl, color="white", fontsize=11, pad=6)
        ax.axis("off")

        if lbl == "Difference Map":
            info = "\n".join(f"{k}: {v}" for k, v in params.items() if v is not None)
            ax.text(0.03, 0.97, info, transform=ax.transAxes,
                    fontsize=8, color="white", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

        if lbl == "GT Segmentation":
            met_str = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            ax.text(0.03, 0.97, met_str, transform=ax.transAxes,
                    fontsize=8, color="lime", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))

    plt.tight_layout()
    fig.savefig(str(run_dir / "result_analysis.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="demo_output 全フォルダの遡り評価")
    parser.add_argument("--gt_mask",      type=str, required=True,
                        help="GTマスク画像のパス")
    parser.add_argument("--normal_diff",  type=str, default=None,
                        help="正常画像の差分マップのパス（指定時はOtsuの代わりに適応閾値を使用）")
    parser.add_argument("--percentile",   type=float, default=95.0,
                        help="正常差分の閾値パーセンタイル (デフォルト: 95)")
    parser.add_argument("--output_dir",   type=str, default="./demo_output")
    parser.add_argument("--update_images", action="store_true",
                        help="result_analysis.png にメトリクスを上書き保存")
    parser.add_argument("--csv_out",      type=str, default="./demo_output/metrics_log.csv",
                        help="CSV保存先")
    args = parser.parse_args()

    gt_mask_orig = Image.open(args.gt_mask).convert("L")
    normal_diff_orig = (Image.open(args.normal_diff).convert("L")
                        if args.normal_diff else None)
    thresh_method = f"normal_ref(p={args.percentile})" if normal_diff_orig else "otsu"
    print(f"閾値方法: {thresh_method}")

    output_dir = Path(args.output_dir)
    run_dirs = sorted([d for d in output_dir.iterdir()
                       if d.is_dir() and (d / "03_difference_map.png").exists()])

    print(f"対象フォルダ数: {len(run_dirs)}")

    rows = []
    for run_dir in run_dirs:
        diff_map = Image.open(run_dir / "03_difference_map.png").convert("L")
        # diff_mapのサイズにGTマスクを合わせる
        gt_mask = gt_mask_orig.resize(diff_map.size, Image.NEAREST)
        gt_binary = np.array(gt_mask) > 127

        if normal_diff_orig is not None:
            # 正常差分を基準にした適応閾値
            normal_diff = normal_diff_orig.resize(diff_map.size, Image.NEAREST)
            tumor_arr  = np.array(diff_map).astype(float) / 255.0
            normal_arr = np.array(normal_diff).astype(float) / 255.0
            threshold  = np.percentile(normal_arr, args.percentile)
            pred_mask  = tumor_arr > threshold
        else:
            pred_mask = threshold_otsu(diff_map)
        metrics = compute_metrics(pred_mask, gt_binary)
        info = parse_folder_info(run_dir.name)

        row = {
            "folder":    run_dir.name,
            "model":     info["model"],
            "t_noise":   info["t_noise"] if info["t_noise"] is not None else "",
            "t_start":   info["t_start"] if info["t_start"] is not None else "",
            **metrics,
        }
        rows.append(row)

        t_str = (f"t_noise={info['t_noise']}" if info["t_noise"] is not None
                 else f"t_start={info['t_start']}" if info["t_start"] is not None
                 else "")
        print(f"  {run_dir.name}  Dice={metrics['Dice']:.4f}  IoU={metrics['IoU']:.4f}  "
              f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}")

        if args.update_images:
            params = {"model": info["model"]}
            if info["t_noise"] is not None:
                params["t_noise"] = info["t_noise"]
            if info["t_start"] is not None:
                params["t_start"] = info["t_start"]
            update_result_image(run_dir, gt_mask, metrics, params)

    # CSV保存
    csv_path = Path(args.csv_out)
    fieldnames = ["folder", "model", "t_noise", "t_start", "Dice", "IoU", "Precision", "Recall"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ CSV保存: {csv_path}")

    if args.update_images:
        print(f"✅ result_analysis.png を全フォルダで更新完了")


if __name__ == "__main__":
    main()
