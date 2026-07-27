"""
demo.py  -  Counterfactual Visual Attribution クイックデモ
torch / diffusers 不要で --use_synthetic オプションで即実行可能。

使い方:
  # 合成データでテスト（ライブラリ最小構成）
  python demo.py --use_synthetic

  # 実データで実行
  python demo.py --input_image ./data/slices/tumor/sample.png

  # img2img方式（推奨: DDIM inversionより自然な出力）
  python demo.py --input_image ./data/slices/tumor/sample.png --use_real_sd --use_img2img --strength 0.55

  # Stable Diffusion 本番版（GPU必要、DDIM inversion方式）
  python demo.py --use_synthetic --use_real_sd
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# torch は --use_real_sd 時のみ必要
try:
    import torch
except ImportError:
    torch = None


# ── 合成脳MRI生成 ──────────────────────────────────────────────────────────────

def create_synthetic_brain_mri(size=512, add_tumor=True, seed=42):
    """合成脳MRI画像を生成（BraTSデータがない場合のテスト用）"""
    from scipy.ndimage import gaussian_filter
    np.random.seed(seed)
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    for y in range(size):
        for x in range(size):
            dx = (x - cx) / (size * 0.45)
            dy = (y - cy) / (size * 0.48)
            r2 = dx**2 + dy**2
            if r2 < 1.0:
                img[y, x] = 0.60 + np.random.normal(0, 0.02) if r2 > 0.72**2 \
                             else 0.75 + np.random.normal(0, 0.02)

    # 脳室（暗い領域）
    for y in range(size):
        for x in range(size):
            dx = (x - cx) / (size * 0.08)
            dy = (y - cy) / (size * 0.15)
            if dx**2 + dy**2 < 1.0:
                img[y, x] = 0.10 + np.random.normal(0, 0.01)

    gt_mask = np.zeros((size, size), dtype=np.uint8)
    if add_tumor:
        tcx = cx + int(size * 0.12)
        tcy = cy - int(size * 0.08)
        tr  = int(size * 0.07)
        for y in range(size):
            for x in range(size):
                d1 = ((x-tcx)/tr)**2 + ((y-tcy)/(tr*0.8))**2
                d2 = ((x-tcx-8)/(tr*0.7))**2 + ((y-tcy+5)/(tr*0.6))**2
                if d1 < 1.0 or d2 < 0.9:
                    img[y, x] = 0.95 + np.random.normal(0, 0.03)
                    gt_mask[y, x] = 255

    img = gaussian_filter(np.clip(img, 0, 1), sigma=1.5)
    img_u8 = (img * 255).astype(np.uint8)
    return Image.fromarray(np.stack([img_u8]*3, axis=2)), Image.fromarray(gt_mask)


# ── シミュレーション版反実仮想（torch不要）──────────────────────────────────────

def simulate_counterfactual(tumor_image):
    """SD未使用のシミュレーション版（動作確認用）"""
    from scipy.ndimage import gaussian_filter, binary_dilation
    img_arr  = np.array(tumor_image).astype(float)
    img_gray = img_arr.mean(axis=2)
    threshold    = np.percentile(img_gray[img_gray > 10], 90)
    tumor_region = img_gray > threshold
    dilated      = binary_dilation(tumor_region, iterations=5)
    surrounding  = dilated & ~tumor_region
    surr_mean    = img_gray[surrounding].mean() if surrounding.sum() > 0 else 150
    cf = img_arr.copy()
    np.random.seed(0)
    cf[tumor_region] = surr_mean + np.random.normal(0, 5, cf[tumor_region].shape)
    cf = gaussian_filter(cf, sigma=[1.5, 1.5, 0])
    return Image.fromarray(np.clip(cf, 0, 255).astype(np.uint8))


# ── 差分マップ ─────────────────────────────────────────────────────────────────

def compute_difference_map(original, counterfactual, smooth_sigma=2.0):
    """元画像と反実仮想の差分から病変マップを生成"""
    from scipy.ndimage import gaussian_filter
    orig = np.array(original).astype(float)
    cf   = np.array(counterfactual).astype(float)
    diff = gaussian_filter(np.abs(orig - cf).mean(axis=2), sigma=smooth_sigma)
    if diff.max() > 0:
        diff = (diff / diff.max() * 255).astype(np.uint8)
    return Image.fromarray(diff)


# ── ヒートマップオーバーレイ ────────────────────────────────────────────────────

def apply_heatmap_overlay(original, diff_map, alpha=0.55):
    orig = np.array(original).astype(float)
    diff = np.array(diff_map).astype(float)
    thr  = np.percentile(diff[diff > 0], 70) if diff.max() > 0 else 0
    diff_m = np.where(diff >= thr, diff, 0)
    cmap   = plt.get_cmap("jet")
    dn     = diff_m / diff_m.max() if diff_m.max() > 0 else diff_m
    hm     = (cmap(dn)[:, :, :3] * 255).astype(float)
    mask   = (diff_m > 0).astype(float)[:, :, np.newaxis]
    return Image.fromarray((orig*(1-alpha*mask) + hm*alpha*mask).astype(np.uint8))


# ── 2値マスク（Otsu）──────────────────────────────────────────────────────────

def threshold_otsu(diff_map):
    from skimage.filters import threshold_otsu as sk_otsu
    arr = np.array(diff_map).astype(float) / 255.0
    try:
        thresh = sk_otsu(arr)
    except Exception:
        thresh = 0.5
    return arr >= thresh


def threshold_by_normal_ref(
    tumor_diff: Image.Image,
    normal_diff: Image.Image,
    percentile: float = 95.0,
) -> np.ndarray:
    """
    正常画像の差分（モデルの再構成誤差）を基準にした適応閾値処理。

    閾値 = normal_diff の上位 percentile% の値
    → 腫瘍差分がその閾値を超えた画素のみを病変候補とする。

    Args:
        tumor_diff  : 腫瘍MRIの差分マップ (PIL, L mode)
        normal_diff : 正常MRIを同モデルで処理した差分マップ (PIL, L mode)
        percentile  : 正常差分の何パーセンタイルを閾値とするか (デフォルト: 95)

    Returns:
        binary mask (H, W) bool
    """
    tumor_arr  = np.array(tumor_diff.convert("L")).astype(float) / 255.0
    normal_arr = np.array(
        normal_diff.convert("L").resize(tumor_diff.size, Image.NEAREST)
    ).astype(float) / 255.0

    threshold = np.percentile(normal_arr, percentile)
    return tumor_arr > threshold


# ── 評価指標 ───────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt)
    return {
        "Dice":      round(float(2*tp / (2*tp+fp+fn+1e-8)), 4),
        "IoU":       round(float(tp   / (tp+fp+fn+1e-8)),   4),
        "Precision": round(float(tp   / (tp+fp+1e-8)),      4),
        "Recall":    round(float(tp   / (tp+fn+1e-8)),      4),
    }


# ── 分析図作成 ─────────────────────────────────────────────────────────────────

def create_analysis_figure(original, counterfactual, diff_map, gt_mask=None, title="",
                           params: dict = None, metrics: dict = None):
    """
    params に渡したキー・値が差分マップパネルに表示される。
    metrics に渡した評価結果（Dice/IoU/Precision/Recall）がGTパネルに表示される。
    """
    orig_arr = np.array(original.convert("RGB"))
    cf_arr   = np.array(counterfactual.convert("RGB"))
    diff_arr = np.array(diff_map.convert("L"))
    overlay  = apply_heatmap_overlay(original, diff_map)

    ncols  = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5.6))
    fig.patch.set_facecolor("#1a1a2e")

    # タイトル（パラメータ情報付き）
    param_str = ""
    if params:
        param_str = "  |  " + "  /  ".join(f"{k}={v}" for k, v in params.items())
    full_title = title + param_str
    if full_title:
        fig.suptitle(full_title, color="white", fontsize=11, y=1.01)

    panels = [
        (orig_arr,         "Original (Tumor MRI)",   None),
        (cf_arr,           "Counterfactual (Normal)", None),
        (diff_arr,         "Difference Map",          "hot"),
        (np.array(overlay),"Lesion Highlight",        None),
    ]
    if gt_mask is not None:
        panels.append((np.array(gt_mask.convert("L")), "GT Segmentation", "gray"))

    for ax, (img, lbl, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(lbl, color="white", fontsize=11, pad=6)
        ax.axis("off")

        # 差分マップパネルにパラメータをテキストで重ねる
        if lbl == "Difference Map" and params:
            info = "\n".join(f"{k}: {v}" for k, v in params.items())
            ax.text(0.03, 0.97, info,
                    transform=ax.transAxes,
                    fontsize=8, color="white", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

        # GTパネルに評価結果を表示
        if lbl == "GT Segmentation" and metrics:
            met_str = "\n".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            ax.text(0.03, 0.97, met_str,
                    transform=ax.transAxes,
                    fontsize=8, color="lime", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))

    plt.tight_layout()
    return fig


# ── メイン処理 ─────────────────────────────────────────────────────────────────

def run_demo(
    input_image_path: Optional[str] = None,
    gt_mask_path: Optional[str] = None,
    use_synthetic: bool = False,
    use_real_sd: bool = False,
    use_img2img: bool = False,
    use_inpaint: bool = False,
    use_ddpm: bool = False,
    ddpm_path: str = "./ddpm_model",
    use_monai: bool = False,
    monai_path: str = "./monai_ddpm_model",
    use_flow: bool = False,
    flow_path: str = "./flow_matching_model",
    t_noise: int = 400,
    t_start: float = 0.4,
    strength: float = 0.55,
    lora_path: Optional[str] = None,
    output_dir: str = "./demo_output",
    image_size: int = 512,
    atlas_path: str = None,
    atlas_strength: float = 0.0,
    normal_ref: Optional[str] = None,
    normal_ref_percentile: float = 95.0,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Counterfactual MRI Demo")
    print("=" * 60)

    # ── 入力画像 ──────────────────────────────────────────────
    if use_synthetic or input_image_path is None:
        print("\n[1/4] 合成脳MRI画像を生成中...")
        tumor_image, gt_mask = create_synthetic_brain_mri(size=image_size, add_tumor=True)
        print("      完了")
    else:
        print(f"\n[1/4] 画像をロード: {input_image_path}")
        tumor_image = Image.open(input_image_path).convert("RGB").resize((image_size, image_size))
        gt_mask = (
            Image.open(gt_mask_path).convert("L").resize((image_size, image_size))
            if gt_mask_path else None
        )

    # ── 反実仮想生成 ──────────────────────────────────────────
    if use_flow:
        if torch is None:
            print("\n[ERROR] --use_flow には torch が必要です。")
            sys.exit(1)
        print(f"\n[2/4] Flow Matching でCounterfactual生成中 (t_start={t_start})...")
        from counterfactual import FlowMatchingCounterfactual
        fm_cf = FlowMatchingCounterfactual(flow_path=flow_path, image_size=128)
        counterfactual_image, diff_map = fm_cf.generate(
            tumor_image, t_start=t_start,
            atlas_path=atlas_path, atlas_strength=atlas_strength,
        )
        counterfactual_image = counterfactual_image.resize((image_size, image_size), Image.LANCZOS)
        diff_map = diff_map.resize((image_size, image_size), Image.LANCZOS)
        print("      完了")
    elif use_monai:
        if torch is None:
            print("\n[ERROR] --use_monai には torch が必要です。")
            sys.exit(1)
        print(f"\n[2/4] MONAI DiffusionModelUNet でCounterfactual生成中 (t_noise={t_noise})...")
        from counterfactual import MONAICounterfactual
        monai_cf = MONAICounterfactual(monai_path=monai_path, image_size=128)
        counterfactual_image, diff_map = monai_cf.generate(
            tumor_image, t_noise=t_noise,
            atlas_path=atlas_path, atlas_strength=atlas_strength,
        )
        counterfactual_image = counterfactual_image.resize((image_size, image_size), Image.LANCZOS)
        diff_map = diff_map.resize((image_size, image_size), Image.LANCZOS)
        print("      完了")
    elif use_ddpm:
        if torch is None:
            print("\n[ERROR] --use_ddpm には torch が必要です。")
            sys.exit(1)
        print(f"\n[2/4] 正常MRI学習済みDDPMでCounterfactual生成中 (t_noise={t_noise})...")
        from counterfactual import DDPMCounterfactual
        ddpm = DDPMCounterfactual(ddpm_path=ddpm_path, image_size=256)
        counterfactual_image, diff_map = ddpm.generate(tumor_image, t_noise=t_noise)
        # 可視化のため元画像サイズに揃える
        counterfactual_image = counterfactual_image.resize((image_size, image_size), Image.LANCZOS)
        diff_map = diff_map.resize((image_size, image_size), Image.LANCZOS)
        print("      完了")
    elif use_inpaint:
        print("\n[2/4] インペインティング方式でCounterfactual生成中...")
        print("      (腫瘍領域を周囲の正常組織で補完 / SDなし・MRI外観保持)")
        from counterfactual import InpaintingCounterfactual
        inpainter = InpaintingCounterfactual()
        ext_mask = np.array(gt_mask.convert("L")) if gt_mask is not None else None
        counterfactual_image, diff_map = inpainter.generate(
            tumor_image, mask=ext_mask
        )
        print("      完了")
    elif use_real_sd:
        if torch is None:
            print("\n[ERROR] --use_real_sd には torch が必要です。")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            sys.exit(1)
        from counterfactual import CounterfactualPipeline
        pipeline = CounterfactualPipeline(image_size=image_size)

        # LoRA重みがあれば読み込む
        if lora_path and Path(lora_path).exists():
            print(f"  LoRAを読み込み: {lora_path}")
            lora_file = Path(lora_path)
            if lora_file.is_dir():
                pipeline.pipeline.load_lora_weights(str(lora_file))
            else:
                pipeline.pipeline.load_lora_weights(str(lora_file.parent),
                                                     weight_name=lora_file.name)
            print("  LoRA適用済み")

        if use_img2img:
            print(f"\n[2/4] img2img方式でCounterfactual生成中 (strength={strength})...")
            print("      元画像にノイズを加えて正常脳プロンプトでデノイズします")
            counterfactual_image, diff_map = pipeline.generate_counterfactual_img2img(
                tumor_image, strength=strength
            )
        else:
            print("\n[2/4] DDIM Inversion方式でCounterfactual生成中 (約2-3分)...")
            counterfactual_image, diff_map = pipeline.generate_counterfactual(tumor_image)
    else:
        print("\n[2/4] シミュレーション版でCounterfactual生成中...")
        counterfactual_image = simulate_counterfactual(tumor_image)
        diff_map = compute_difference_map(tumor_image, counterfactual_image)
        print("      完了")

    # ── 正常参照差分の計算（--normal_ref 指定時・可視化より前に実行）──
    normal_diff_map = None
    if normal_ref and (use_monai or use_flow or use_ddpm):
        print(f"\n[2.5/4] 正常参照画像を処理中: {normal_ref}")
        normal_image = Image.open(normal_ref).convert("RGB").resize((image_size, image_size))
        if use_flow:
            _, normal_diff_map = fm_cf.generate(
                normal_image, t_start=t_start,
                atlas_path=atlas_path, atlas_strength=atlas_strength,
            )
        elif use_monai:
            _, normal_diff_map = monai_cf.generate(
                normal_image, t_noise=t_noise,
                atlas_path=atlas_path, atlas_strength=atlas_strength,
            )
        elif use_ddpm:
            _, normal_diff_map = ddpm.generate(normal_image, t_noise=t_noise)
        normal_diff_map = normal_diff_map.resize((image_size, image_size), Image.LANCZOS)
        print("      正常差分を計算しました")

    # ── 可視化 ────────────────────────────────────────────────
    print("\n[3/4] 可視化中...")
    # 使用したパラメータを画像に記録
    if use_flow:
        vis_params = {"model": "Flow-Matching", "t_start": t_start}
    elif use_monai:
        vis_params = {"model": "MONAI-DDPM", "t_noise": t_noise}
    elif use_ddpm:
        vis_params = {"model": "diffusers-DDPM", "t_noise": t_noise}
    elif use_img2img:
        vis_params = {"model": "SD1.5-img2img", "strength": strength}
    elif use_real_sd:
        vis_params = {"model": "SD1.5-DDIM"}
    else:
        vis_params = {"model": "simulation"}

    # ── 評価（図生成前に計算してメトリクスを画像に埋め込む）──────
    eval_metrics = None
    if gt_mask is not None:
        print("\n[3.5/4] 評価中...")
        gt_arr    = np.array(gt_mask.convert("L"))
        gt_binary = gt_arr > 127
        if normal_diff_map is not None:
            pred_mask = threshold_by_normal_ref(diff_map, normal_diff_map, normal_ref_percentile)
            thresh_method = f"normal_ref(p={normal_ref_percentile})"
        else:
            pred_mask = threshold_otsu(diff_map)
            thresh_method = "otsu"
        eval_metrics = compute_metrics(pred_mask, gt_binary)
        eval_metrics["thresh"] = thresh_method
        print(f"\n  評価結果 (vs GT mask, threshold={thresh_method}):")
        for k, v in eval_metrics.items():
            if isinstance(v, float):
                bar = "#" * int(v * 20)
                print(f"    {k:12s}: {v:.4f}  {bar}")
            else:
                print(f"    {k:12s}: {v}")

    # タイムスタンプ付きサブフォルダを作成（上書きしない）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = vis_params.get("model", "unknown")
    run_name = f"{ts}_{model_tag}"
    if "t_noise" in vis_params:
        run_name += f"_t{vis_params['t_noise']}"
    elif "strength" in vis_params:
        run_name += f"_s{vis_params['strength']}"
    run_dir = output_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fig = create_analysis_figure(
        tumor_image, counterfactual_image, diff_map,
        gt_mask=gt_mask,
        title="Counterfactual Visual Attribution - Brain Tumor MRI",
        params=vis_params,
        metrics=eval_metrics,
    )
    result_path = run_dir / "result_analysis.png"
    fig.savefig(str(result_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    tumor_image.save(run_dir / "01_original.png")
    counterfactual_image.save(run_dir / "02_counterfactual.png")
    diff_map.save(run_dir / "03_difference_map.png")
    if normal_diff_map is not None:
        normal_diff_map.save(run_dir / "00_normal_ref_diff.png")
    print(f"      保存: {run_dir}/")

    # ── 評価（表示のみ・すでに計算済み）──────────────────────────
    if eval_metrics is not None:
        print("\n[4/4] 評価完了（GTパネルに埋め込み済み）")

    print(f"\n{'='*60}")
    print(f"  完了! -> {run_dir}\\result_analysis.png")
    print(f"{'='*60}")


# ── エントリーポイント ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Counterfactual MRI デモ")
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--gt_mask",     type=str, default=None)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="合成画像でテスト実行")
    parser.add_argument("--use_real_sd",   action="store_true",
                        help="Stable Diffusion本番版（GPU必要）")
    parser.add_argument("--use_img2img",   action="store_true",
                        help="img2img方式（ノイズ追加→デノイズ）。DDIM inversionより構造が保たれやすい")
    parser.add_argument("--use_monai",     action="store_true",
                        help="MONAI DiffusionModelUNet（医療画像専用・推奨）")
    parser.add_argument("--monai_path",   type=str, default="./monai_ddpm_model",
                        help="train_monai_ddpm.py で学習したモデルのディレクトリ")
    parser.add_argument("--use_flow",      action="store_true",
                        help="Flow Matching（Conditional Flow Matching・DDPMと比較用）")
    parser.add_argument("--flow_path",    type=str, default="./flow_matching_model",
                        help="train_flow_matching.py で学習したモデルのディレクトリ")
    parser.add_argument("--t_start",      type=float, default=0.4,
                        help="FM用: 腐敗の程度 (0.0〜1.0)。DDPMのt_noise/1000に相当 (推奨: 0.3〜0.5)")
    parser.add_argument("--use_ddpm",      action="store_true",
                        help="diffusers UNet2DModel による DDPM")
    parser.add_argument("--ddpm_path",    type=str, default="./ddpm_model",
                        help="train_ddpm.py で学習したモデルのディレクトリ")
    parser.add_argument("--t_noise",      type=int, default=400,
                        help="ノイズ追加タイムステップ (300〜500推奨)")
    parser.add_argument("--use_inpaint",   action="store_true",
                        help="インペインティング方式（SDなし・MRI外観を完全保持）")
    parser.add_argument("--strength",      type=float, default=0.55,
                        help="img2img strength: 0.0=変化なし〜1.0=完全再生成 (推奨: 0.4〜0.65)")
    parser.add_argument("--lora_path",   type=str, default=None,
                        help="LoRA重みのパス（train_lora.pyで学習したもの）")
    parser.add_argument("--output_dir",      type=str, default="./demo_output")
    parser.add_argument("--image_size",      type=int, default=512)
    parser.add_argument("--atlas_path",      type=str, default=None,
                        help="正常脳アトラス画像のパス (build_atlas.py で作成)")
    parser.add_argument("--atlas_strength",  type=float, default=0.0,
                        help="アトラスへの引き寄せ強度 0.0〜1.0 (推奨: 0.2〜0.4)")
    parser.add_argument("--normal_ref",      type=str, default=None,
                        help="正常脳スライス画像のパス。このスライスの差分を閾値の基準にする")
    parser.add_argument("--normal_ref_percentile", type=float, default=95.0,
                        help="正常差分の何パーセンタイルを閾値とするか (デフォルト: 95)")
    args = parser.parse_args()

    run_demo(
        input_image_path=args.input_image,
        gt_mask_path=args.gt_mask,
        use_synthetic=args.use_synthetic or args.input_image is None,
        use_real_sd=args.use_real_sd,
        use_img2img=args.use_img2img,
        use_inpaint=args.use_inpaint,
        use_ddpm=args.use_ddpm,
        ddpm_path=args.ddpm_path,
        use_monai=args.use_monai,
        monai_path=args.monai_path,
        use_flow=args.use_flow,
        flow_path=args.flow_path,
        t_noise=args.t_noise,
        t_start=args.t_start,
        strength=args.strength,
        lora_path=args.lora_path,
        atlas_path=args.atlas_path,
        atlas_strength=args.atlas_strength,
        normal_ref=args.normal_ref,
        normal_ref_percentile=args.normal_ref_percentile,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
