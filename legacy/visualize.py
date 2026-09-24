"""
visualize.py
Counterfactual差分マップの高度な可視化ツール

機能:
  - ヒートマップオーバーレイ（元画像に病変領域を重ね表示）
  - 閾値ベースの病変セグメンテーション
  - 評価指標の計算（GT maskがある場合）
  - 結果サマリーのHTMLレポート生成

使い方:
  python visualize.py \
    --original_dir ./data/slices/tumor \
    --cf_dir ./results/counterfactual \
    --diff_dir ./results/difference \
    --mask_dir ./data/slices/masks \
    --output_dir ./results/report
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, label


def apply_heatmap_overlay(
    original: np.ndarray,
    diff_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    threshold_percentile: float = 70.0,
) -> np.ndarray:
    """
    元画像に差分ヒートマップをオーバーレイ

    Args:
        original: 元画像 (H, W, 3) uint8
        diff_map: 差分マップ (H, W) float or uint8
        alpha: ヒートマップの透明度
        colormap: カラーマップ名
        threshold_percentile: 表示するパーセンタイル閾値

    Returns:
        オーバーレイ画像 (H, W, 3) uint8
    """
    diff_float = diff_map.astype(float)

    # 閾値以下を0に（ノイズ除去）
    threshold = np.percentile(diff_float[diff_float > 0], threshold_percentile) \
        if diff_float.max() > 0 else 0
    diff_masked = np.where(diff_float >= threshold, diff_float, 0)

    # カラーマップ適用
    cmap = plt.get_cmap(colormap)
    if diff_masked.max() > 0:
        diff_norm = diff_masked / diff_masked.max()
    else:
        diff_norm = diff_masked

    heatmap_rgba = (cmap(diff_norm) * 255).astype(np.uint8)
    heatmap_rgb = heatmap_rgba[:, :, :3]

    # マスク（閾値以上の部分のみ）
    mask = (diff_masked > 0).astype(float)[:, :, np.newaxis]

    # アルファブレンディング
    original_float = original.astype(float)
    overlay = original_float * (1 - alpha * mask) + heatmap_rgb.astype(float) * alpha * mask

    return overlay.astype(np.uint8)


def threshold_to_binary(
    diff_map: np.ndarray,
    method: str = "otsu",
    manual_threshold: float = 0.5,
) -> np.ndarray:
    """
    差分マップを閾値処理して2値セグメンテーションマスクを生成

    Args:
        diff_map: 差分マップ (H, W) 0-255
        method: "otsu" | "percentile" | "manual"
        manual_threshold: methodが"manual"の場合の閾値 (0-1)

    Returns:
        2値マスク (H, W) bool
    """
    diff_norm = diff_map.astype(float) / 255.0

    if method == "otsu":
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(diff_norm)
        except Exception:
            thresh = 0.5
        return diff_norm >= thresh

    elif method == "percentile":
        thresh = np.percentile(diff_norm, 80)
        return diff_norm >= thresh

    else:  # manual
        return diff_norm >= manual_threshold


def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict:
    """
    予測マスクとGTマスクを比較して評価指標を計算

    Args:
        pred_mask: 予測マスク (H, W) bool
        gt_mask: GTマスク (H, W) bool

    Returns:
        評価指標の辞書（Dice, IoU, Precision, Recall）
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "Dice": round(float(dice), 4),
        "IoU": round(float(iou), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
    }


def create_analysis_figure(
    original: Image.Image,
    counterfactual: Image.Image,
    diff_map: Image.Image,
    gt_mask: Optional[Image.Image] = None,
    title: str = "",
) -> plt.Figure:
    """
    分析結果を1枚の図にまとめる

    パネル構成:
      [元画像] [反実仮想] [差分マップ] [ヒートマップOL] ([GTマスク])
    """
    orig_arr = np.array(original.convert("RGB"))
    cf_arr = np.array(counterfactual.convert("RGB"))
    diff_arr = np.array(diff_map.convert("L"))

    # ヒートマップオーバーレイ
    overlay = apply_heatmap_overlay(orig_arr, diff_arr)

    # 2値マスク
    binary_mask = threshold_to_binary(diff_arr, method="otsu")

    ncols = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.patch.set_facecolor("#1a1a2e")
    if title:
        fig.suptitle(title, color="white", fontsize=13, y=1.02)

    panels = [
        (orig_arr, "元画像（腫瘍MRI）", None),
        (cf_arr, "反実仮想（正常脳）", None),
        (diff_arr, "差分マップ", "hot"),
        (overlay, "病変ハイライト", None),
    ]
    if gt_mask is not None:
        gt_arr = np.array(gt_mask.convert("L"))
        panels.append((gt_arr, "GT セグメンテーション", "gray"))

    for ax, (img, label_text, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label_text, color="white", fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout()
    return fig


def generate_html_report(
    results: list,
    output_path: str,
):
    """
    処理結果のHTMLサマリーレポートを生成

    Args:
        results: [{filename, dice, iou, ...}, ...]
        output_path: 出力HTMLファイルパス
    """
    rows = ""
    for r in results:
        dice_color = "#4ade80" if r.get("Dice", 0) > 0.5 else "#f87171"
        rows += f"""
        <tr>
          <td>{r['filename']}</td>
          <td style="color:{dice_color}">{r.get('Dice', 'N/A')}</td>
          <td>{r.get('IoU', 'N/A')}</td>
          <td>{r.get('Precision', 'N/A')}</td>
          <td>{r.get('Recall', 'N/A')}</td>
        </tr>"""

    avg_dice = np.mean([r.get('Dice', 0) for r in results if 'Dice' in r])

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Counterfactual MRI - 評価レポート</title>
<style>
  body {{ background: #0f0f1a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 2rem; }}
  h1 {{ color: #818cf8; border-bottom: 1px solid #374151; padding-bottom: 0.5rem; }}
  .summary {{ background: #1e1e3a; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th {{ background: #312e81; padding: 0.75rem; text-align: left; }}
  td {{ padding: 0.6rem; border-bottom: 1px solid #374151; }}
  tr:hover {{ background: #1e2a3a; }}
  .metric {{ font-size: 2rem; font-weight: bold; color: #a5b4fc; }}
</style>
</head>
<body>
<h1>🧠 Counterfactual Visual Attribution - 評価レポート</h1>
<div class="summary">
  <p>処理枚数: <span class="metric">{len(results)}</span></p>
  <p>平均 Dice スコア: <span class="metric">{avg_dice:.4f}</span></p>
</div>
<table>
  <thead>
    <tr><th>ファイル名</th><th>Dice</th><th>IoU</th><th>Precision</th><th>Recall</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📊 レポート保存: {output_path}")


# ─── バッチ可視化 ───────────────────────────────────────────────────────────────

def run_visualization(
    original_dir: str,
    cf_dir: str,
    diff_dir: str,
    output_dir: str,
    mask_dir: Optional[str] = None,
    num_images: int = 20,
):
    orig_path = Path(original_dir)
    cf_path = Path(cf_dir)
    diff_path = Path(diff_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    orig_files = sorted(orig_path.glob("*.png"))[:num_images]
    all_metrics = []

    for orig_file in orig_files:
        stem = orig_file.stem.replace("_cf", "")
        cf_file = cf_path / f"{stem}_cf.png"
        diff_file = diff_path / f"{stem}_diff.png"

        if not cf_file.exists() or not diff_file.exists():
            continue

        original = Image.open(orig_file)
        counterfactual = Image.open(cf_file)
        diff_map = Image.open(diff_file)

        gt_mask = None
        metrics = {}
        if mask_dir:
            mask_file = Path(mask_dir) / f"{stem}.png"
            if mask_file.exists():
                gt_mask = Image.open(mask_file)
                diff_arr = np.array(diff_map.convert("L"))
                gt_arr = np.array(gt_mask.convert("L"))
                pred_mask = threshold_to_binary(diff_arr, method="otsu")
                gt_binary = gt_arr > 127
                metrics = compute_metrics(pred_mask, gt_binary)

        # 分析図の生成と保存
        fig = create_analysis_figure(
            original, counterfactual, diff_map, gt_mask, title=stem
        )
        fig.savefig(
            out_path / f"{stem}_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        plt.close(fig)

        all_metrics.append({"filename": stem, **metrics})
        print(f"  {stem}: {metrics}")

    # HTMLレポート
    if any("Dice" in m for m in all_metrics):
        generate_html_report(all_metrics, str(out_path / "report.html"))

    print(f"\n✅ 可視化完了: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual MRI可視化・評価")
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--cf_dir", type=str, required=True)
    parser.add_argument("--diff_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/report")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="GTセグメンテーションマスクのディレクトリ（評価用）")
    parser.add_argument("--num_images", type=int, default=20)
    args = parser.parse_args()

    run_visualization(
        original_dir=args.original_dir,
        cf_dir=args.cf_dir,
        diff_dir=args.diff_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        num_images=args.num_images,
    )
