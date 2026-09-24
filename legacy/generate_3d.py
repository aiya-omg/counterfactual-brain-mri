"""
generate_3d.py
1患者の全スライスにMONAI / Flow Matchingモデルを適用し、
差分マップを3Dボリュームとして再構成してNIfTI保存＋可視化を行う。

使い方:
  # MONAI-DDPM
  python generate_3d.py \
    --patient_dir ./data/BraTS2023/.../BraTS-GLI-00000-000 \
    --monai_path ./monai_ddpm_model_v3 \
    --t_noise 400 \
    --output_dir ./output_3d

  # Flow Matching（正常参照閾値あり）
  python generate_3d.py \
    --patient_dir ./data/BraTS2023/.../BraTS-GLI-00000-000 \
    --use_flow --flow_path ./flow_matching_model \
    --t_start 0.4 \
    --normal_ref ./data/slices/normal/BraTS-GLI-00000-000_slice030.png \
    --output_dir ./output_3d_fm
"""

import argparse
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from counterfactual import MONAICounterfactual, FlowMatchingCounterfactual


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    """パーセンタイルクリッピングで [0,1] に正規化"""
    p1, p99 = np.percentile(arr[arr > 0], [1, 99]) if arr.max() > 0 else (0, 1)
    arr = np.clip(arr, p1, p99)
    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    return arr.astype(np.float32)


def slice_to_pil(arr2d: np.ndarray, image_size: int = 128) -> Image.Image:
    """2D numpy スライス → PIL Image (グレースケール, image_size x image_size)"""
    norm = normalize_slice(arr2d)
    img = Image.fromarray((norm * 255).astype(np.uint8)).convert("L")
    return img.resize((image_size, image_size), Image.LANCZOS)


def run_3d(
    patient_dir: str,
    monai_path: str = None,
    t_noise: int = 400,
    use_flow: bool = False,
    flow_path: str = None,
    t_start: float = 0.4,
    normal_ref: str = None,
    normal_ref_percentile: float = 95.0,
    modality: str = "t2f",
    output_dir: str = "./output_3d",
    image_size: int = 128,
):
    patient_dir = Path(patient_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    patient_id = patient_dir.name
    model_tag  = f"FM_t{t_start}" if use_flow else f"DDPM_t{t_noise}"
    print(f"患者: {patient_id}")
    print(f"モデル: {'Flow Matching' if use_flow else 'MONAI-DDPM'}  {model_tag}")

    # ── NIfTI ロード ─────────────────────────────────────────────────────
    img_path = patient_dir / f"{patient_id}-{modality}.nii.gz"
    seg_path = patient_dir / f"{patient_id}-seg.nii.gz"
    if not img_path.exists():
        img_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
        seg_path = patient_dir / f"{patient_id}_seg.nii.gz"

    print(f"MRI : {img_path.name}")
    img_nii = nib.load(str(img_path))
    affine  = img_nii.affine
    img_vol = img_nii.get_fdata().astype(np.float32)   # (H, W, D)
    seg_vol = nib.load(str(seg_path)).get_fdata().astype(np.float32) if seg_path.exists() else None

    H, W, D = img_vol.shape
    print(f"ボリューム形状: {H}x{W}x{D}")

    # ── モデルロード ─────────────────────────────────────────────────────
    print("モデルロード中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_flow:
        cf_model = FlowMatchingCounterfactual(flow_path=flow_path, device=device, image_size=image_size)
    else:
        cf_model = MONAICounterfactual(monai_path=monai_path, device=device, image_size=image_size)

    # ── 正常参照差分の事前計算（閾値用）────────────────────────────────
    normal_diff_arr = None
    if normal_ref:
        print(f"正常参照画像を処理中: {normal_ref}")
        normal_pil = Image.open(normal_ref).convert("RGB")
        with torch.no_grad():
            if use_flow:
                _, normal_diff_pil = cf_model.generate(normal_pil, t_start=t_start)
            else:
                _, normal_diff_pil = cf_model.generate(normal_pil, t_noise=t_noise)
        normal_diff_arr = np.array(normal_diff_pil.convert("L")).astype(np.float32) / 255.0
        normal_threshold = np.percentile(normal_diff_arr, normal_ref_percentile)
        print(f"  正常差分 閾値(p={normal_ref_percentile}): {normal_threshold:.4f}")

    # ── 全スライス処理 ───────────────────────────────────────────────────
    diff_vol        = np.zeros((image_size, image_size, D), dtype=np.float32)
    diff_thresh_vol = np.zeros((image_size, image_size, D), dtype=np.float32)
    cf_vol          = np.zeros((image_size, image_size, D), dtype=np.float32)
    orig_vol        = np.zeros((image_size, image_size, D), dtype=np.float32)
    valid_mask      = np.zeros(D, dtype=bool)

    for z in tqdm(range(D), desc="スライス処理中"):
        sl = img_vol[:, :, z]
        if sl.max() == 0:
            continue

        img_pil = slice_to_pil(sl, image_size)

        with torch.no_grad():
            if use_flow:
                cf_img, diff_map = cf_model.generate(img_pil, t_start=t_start)
            else:
                cf_img, diff_map = cf_model.generate(img_pil, t_noise=t_noise)

        cf_np   = np.array(cf_img.convert("L")).astype(np.float32) / 255.0
        diff_np = np.array(diff_map.convert("L")).astype(np.float32) / 255.0
        orig_np = np.array(img_pil).astype(np.float32) / 255.0

        cf_vol[:, :, z]   = cf_np
        diff_vol[:, :, z] = diff_np
        orig_vol[:, :, z] = orig_np

        # 正常参照閾値バージョン（指定時のみ）
        if normal_diff_arr is not None:
            binary = (diff_np > normal_threshold).astype(np.float32)
            diff_thresh_vol[:, :, z] = binary

        valid_mask[z] = True

    print(f"処理スライス数: {valid_mask.sum()} / {D}")

    # ── NIfTI 保存 ───────────────────────────────────────────────────────
    scale = image_size / H
    affine_scaled = affine.copy()
    affine_scaled[:3, :3] *= (1.0 / scale)

    def save_nifti(vol, name):
        nii = nib.Nifti1Image(vol, affine_scaled)
        path = output_path / name
        nib.save(nii, str(path))
        print(f"  保存: {path}")

    save_nifti(orig_vol,  f"{patient_id}_{model_tag}_original.nii.gz")
    save_nifti(cf_vol,    f"{patient_id}_{model_tag}_counterfactual.nii.gz")
    save_nifti(diff_vol,  f"{patient_id}_{model_tag}_diffmap.nii.gz")
    if normal_diff_arr is not None:
        save_nifti(diff_thresh_vol, f"{patient_id}_{model_tag}_diffmap_thresh.nii.gz")
    if seg_vol is not None:
        seg_small = np.zeros((image_size, image_size, D), dtype=np.float32)
        for z in range(D):
            sl = (seg_vol[:, :, z] > 0).astype(np.uint8) * 255
            seg_pil = Image.fromarray(sl).resize((image_size, image_size), Image.NEAREST)
            seg_small[:, :, z] = np.array(seg_pil) / 255.0
        save_nifti(seg_small, f"{patient_id}_{model_tag}_seg.nii.gz")

    # ── 可視化 ───────────────────────────────────────────────────────────
    print("可視化生成中...")
    thresh_vol_for_vis = diff_thresh_vol if normal_diff_arr is not None else None
    _visualize(orig_vol, cf_vol, diff_vol, seg_vol, patient_id, model_tag,
               output_path, D, image_size, thresh_vol_for_vis)

    print(f"\n完了 -> {output_path}/")
    print("NIfTIファイルは ITK-SNAP / 3D Slicer で開いて3D確認できます。")


def _visualize(orig_vol, cf_vol, diff_vol, seg_vol, patient_id, model_tag,
               output_path, D, image_size, thresh_vol=None):
    """3断面（Axial / Coronal / Sagittal）比較図を出力"""

    # 差分が最大のスライスを自動選択
    diff_sum = diff_vol.sum(axis=(0, 1))
    best_z   = int(np.argmax(diff_sum))

    n_rows      = 4 if thresh_vol is not None else 3
    rows_labels = ["Original", "Counterfactual", "Diff Map"]
    vols        = [orig_vol, cf_vol, diff_vol]
    cmaps       = ["gray", "gray", "hot"]
    if thresh_vol is not None:
        rows_labels.append("Thresholded")
        vols.append(thresh_vol)
        cmaps.append("hot")

    fig = plt.figure(figsize=(18, 4 * n_rows), facecolor="#1a1a2e")
    fig.suptitle(
        f"3D Counterfactual Analysis  |  {patient_id}  |  {model_tag}",
        color="white", fontsize=14, y=0.99
    )

    # ── 3断面 ────────────────────────────────────────────────────────────
    # Axial: xy @ best_z, Coronal: xz @ mid_y, Sagittal: yz @ mid_x
    mid_x = image_size // 2
    mid_y = image_size // 2

    col_titles = [
        f"Axial (z={best_z}, 最大差分)",
        f"Coronal (y={mid_y})",
        f"Sagittal (x={mid_x})",
    ]

    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.08, wspace=0.05,
                           left=0.06, right=0.97, top=0.96, bottom=0.02)

    for row, (vol, rl, cm) in enumerate(zip(vols, rows_labels, cmaps)):
        slices = [
            vol[:, :, best_z],          # Axial
            vol[:, mid_y, :],           # Coronal
            vol[mid_x, :, :],           # Sagittal
        ]
        for col, sl in enumerate(slices):
            ax = fig.add_subplot(gs[row, col])
            vmax = 1.0 if cm == "gray" else None
            ax.imshow(sl.T, cmap=cm, vmin=0, vmax=vmax, origin="lower", aspect="auto")
            if row == 0:
                ax.set_title(col_titles[col], color="white", fontsize=10, pad=4)
            if col == 0:
                ax.set_ylabel(rl, color="white", fontsize=10, labelpad=6)
            ax.axis("off")

            # GTマスクのオーバーレイ（差分マップ行・閾値行のみ）
            if row >= 2 and seg_vol is not None:
                from PIL import Image as PILImage
                seg_small_slices = [None, None, None]
                seg_np = (seg_vol > 0).astype(np.float32)
                # Axial
                sl_seg = seg_np[:, :, best_z]
                seg_rs = np.array(PILImage.fromarray((sl_seg * 255).astype(np.uint8)).resize(
                    (image_size, image_size), PILImage.NEAREST)) / 255.0
                seg_small_slices[0] = seg_rs
                # Coronal
                sl_seg = seg_np[:, mid_y, :]
                seg_rs = np.array(PILImage.fromarray((sl_seg * 255).astype(np.uint8)).resize(
                    (image_size, D), PILImage.NEAREST)) / 255.0
                seg_small_slices[1] = seg_rs
                # Sagittal
                sl_seg = seg_np[mid_x, :, :]
                seg_rs = np.array(PILImage.fromarray((sl_seg * 255).astype(np.uint8)).resize(
                    (image_size, D), PILImage.NEAREST)) / 255.0
                seg_small_slices[2] = seg_rs

                mask = seg_small_slices[col]
                if mask is not None and mask.max() > 0:
                    rgba = np.zeros((*mask.T.shape, 4))
                    rgba[..., 1] = 0.8   # green
                    rgba[..., 3] = mask.T * 0.5
                    ax.imshow(rgba, origin="lower", aspect="auto")
                    if col == 0 and row == 2:
                        ax.text(0.02, 0.97, "GT mask (green)", transform=ax.transAxes,
                                color="lime", fontsize=7, va="top")

    out_fig = output_path / f"{patient_id}_{model_tag}_3d_analysis.png"
    plt.savefig(str(out_fig), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  可視化: {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Counterfactual MRI生成")
    parser.add_argument("--patient_dir", type=str, required=True,
                        help="患者ディレクトリのパス")
    # MONAI-DDPM 設定
    parser.add_argument("--monai_path",  type=str, default="./monai_ddpm_model_v2")
    parser.add_argument("--t_noise",     type=int, default=400)
    # Flow Matching 設定
    parser.add_argument("--use_flow",    action="store_true",
                        help="Flow Matchingモデルを使用する")
    parser.add_argument("--flow_path",   type=str, default="./flow_matching_model")
    parser.add_argument("--t_start",     type=float, default=0.4,
                        help="Flow Matching の部分破壊開始時刻 (0〜1)")
    # 正常参照閾値
    parser.add_argument("--normal_ref",  type=str, default=None,
                        help="正常画像パス（指定時は正常差分を閾値として使用）")
    parser.add_argument("--normal_ref_percentile", type=float, default=95.0)
    # 共通設定
    parser.add_argument("--modality",    type=str, default="t2f")
    parser.add_argument("--output_dir",  type=str, default="./output_3d")
    parser.add_argument("--image_size",  type=int, default=128)
    args = parser.parse_args()

    run_3d(
        patient_dir=args.patient_dir,
        monai_path=args.monai_path,
        t_noise=args.t_noise,
        use_flow=args.use_flow,
        flow_path=args.flow_path,
        t_start=args.t_start,
        normal_ref=args.normal_ref,
        normal_ref_percentile=args.normal_ref_percentile,
        modality=args.modality,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )
