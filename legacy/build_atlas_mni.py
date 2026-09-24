"""
build_atlas_mni.py
MNI152標準脳テンプレートから、BraTSスライスに対応するアトラス画像を取り出す。

BraTSはMNI152空間に位置合わせ済みなので、スライス番号を比率変換するだけで
解剖学的に対応するスライスが得られる。
  z_mni = round(z_brats / 155 * 182)

使い方:
  conda activate cfmri
  pip install nilearn --break-system-packages   # 初回のみ

  # 単一スライス → atlas_slice065.png
  python build_atlas_mni.py --slice_z 65 --output ./atlas_mni_slice065.png

  # BraTSの全スライス分をまとめて作る（0〜154）
  python build_atlas_mni.py --all --output_dir ./atlas_mni/

  # 入力画像のファイル名からスライス番号を自動検出
  python build_atlas_mni.py --from_image ./data/slices_test/tumor/BraTS-GLI-00241-000_slice065.png --output ./atlas_mni_slice065.png
"""

import argparse
import re
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom, shift as ndshift


def load_mni152():
    """nilearn経由でMNI152 T1テンプレートをロード"""
    try:
        import nilearn.datasets as nlds
        import nibabel as nib
        print("MNI152テンプレートをロード中...")
        mni_img = nlds.load_mni152_template(resolution=1)  # 1mm等方性
        vol = mni_img.get_fdata()  # shape: (182, 218, 182) x,y,z
        print(f"  MNI152 shape: {vol.shape}")
        return vol
    except ImportError:
        raise ImportError("nilearn が必要です: pip install nilearn --break-system-packages")


def brats_z_to_mni_z(z_brats: int, brats_slices: int = 155, mni_slices: int = 182) -> int:
    """BraTSのスライス番号をMNI152のz座標に変換"""
    return round(z_brats / brats_slices * mni_slices)


def extract_mni_slice(
    vol: np.ndarray,
    z_mni: int,
    image_size: int = 128,
    flip_lr: bool = True,
    flip_ud: bool = False,
) -> Image.Image:
    """MNI152ボリュームからz_mniのaxialスライスを取り出してリサイズ"""
    z_mni = max(0, min(z_mni, vol.shape[2] - 1))
    sl = vol[:, :, z_mni]  # (182, 218) x,y

    # BraTSはLPS・MNI152はRASなので左右反転で合わせる
    if flip_lr:
        sl = np.fliplr(sl)
    if flip_ud:
        sl = np.flipud(sl)

    # パーセンタイルノーマライズ
    mask = sl > 0
    if mask.sum() > 0:
        p1, p99 = np.percentile(sl[mask], [1, 99])
        sl = np.clip(sl, p1, p99)
        sl = (sl - p1) / (p99 - p1 + 1e-8)
    else:
        sl = np.zeros_like(sl)

    img = Image.fromarray((sl * 255).astype(np.uint8)).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    return img


def get_brain_bbox(arr: np.ndarray, threshold: float = 0.05):
    """脳領域のバウンディングボックスと重心を返す"""
    mask = arr > (arr.max() * threshold)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cy = (rmin + rmax) / 2.0
    cx = (cmin + cmax) / 2.0
    h = rmax - rmin
    w = cmax - cmin
    return {"rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax,
            "cy": cy, "cx": cx, "h": h, "w": w}


def align_atlas_to_image(
    atlas_arr: np.ndarray,
    target_arr: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    アトラスの脳輪郭をターゲット画像の脳輪郭に合わせる。
    スケール（拡縮）＋平行移動のアフィン変換を適用。

    Args:
        atlas_arr  : アトラス画像 (H, W) 0〜1 float
        target_arr : 患者MRI画像 (H, W) 0〜1 float
        threshold  : 脳マスクの閾値（最大値に対する割合）

    Returns:
        aligned_atlas : アトラスをターゲットの脳サイズ・位置に合わせた画像 (H, W)
    """
    try:
        atlas_bb  = get_brain_bbox(atlas_arr,  threshold)
        target_bb = get_brain_bbox(target_arr, threshold)
    except (IndexError, ValueError):
        # 脳領域が検出できなければそのまま返す
        return atlas_arr

    # スケールファクター（高さ・幅の平均比率）
    scale_h = target_bb["h"] / (atlas_bb["h"] + 1e-8)
    scale_w = target_bb["w"] / (atlas_bb["w"] + 1e-8)
    scale   = (scale_h + scale_w) / 2.0

    # ズーム（拡縮）
    zoomed = zoom(atlas_arr, scale, order=1)

    # キャンバスサイズに合わせてクロップ or パディング
    H, W = atlas_arr.shape
    zh, zw = zoomed.shape
    canvas = np.zeros((H, W), dtype=np.float32)

    # ズーム後画像の中心をキャンバス中心に配置
    y0 = (H - zh) // 2
    x0 = (W - zw) // 2
    y1 = y0 + zh
    x1 = x0 + zw
    # クロップして貼り付け
    src_y0 = max(0, -y0);  dst_y0 = max(0, y0)
    src_x0 = max(0, -x0);  dst_x0 = max(0, x0)
    src_y1 = zh - max(0, y1 - H);  dst_y1 = min(H, y1)
    src_x1 = zw - max(0, x1 - W);  dst_x1 = min(W, x1)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = zoomed[src_y0:src_y1, src_x0:src_x1]

    # 平行移動：アトラス重心 → ターゲット重心
    # ズーム後のアトラス重心はキャンバス中心（H/2, W/2）
    dy = target_bb["cy"] - H / 2.0
    dx = target_bb["cx"] - W / 2.0
    aligned = ndshift(canvas, shift=(dy, dx), order=1, mode="constant", cval=0)

    return np.clip(aligned, 0, 1)


def get_slice_z_from_filename(image_path: str) -> int:
    """ファイル名からスライス番号を自動検出 (例: ..._slice065.png → 65)"""
    m = re.search(r"slice(\d+)", Path(image_path).name)
    if not m:
        raise ValueError(f"スライス番号をファイル名から検出できません: {image_path}")
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser(description="MNI152アトラススライス抽出")
    parser.add_argument("--slice_z",     type=int, default=None,
                        help="BraTSのスライス番号 (0〜154)")
    parser.add_argument("--from_image",  type=str, default=None,
                        help="推論対象画像のパス（ファイル名からスライス番号を自動検出）")
    parser.add_argument("--output",      type=str, default="./atlas_mni.png",
                        help="出力アトラス画像のパス")
    parser.add_argument("--all",         action="store_true",
                        help="BraTS全スライス分（0〜154）をまとめて生成")
    parser.add_argument("--output_dir",  type=str, default="./atlas_mni",
                        help="--all 使用時の出力ディレクトリ")
    parser.add_argument("--image_size",   type=int, default=128)
    parser.add_argument("--brats_slices", type=int, default=155)
    parser.add_argument("--no_flip_lr",   action="store_true",
                        help="左右反転を無効にする（デフォルト: 有効）")
    parser.add_argument("--flip_ud",      action="store_true",
                        help="上下反転を有効にする")
    args = parser.parse_args()

    vol = load_mni152()
    flip_lr = not args.no_flip_lr

    if args.all:
        # 全スライス生成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for z_brats in range(args.brats_slices):
            z_mni = brats_z_to_mni_z(z_brats, args.brats_slices)
            img = extract_mni_slice(vol, z_mni, args.image_size, flip_lr=flip_lr, flip_ud=args.flip_ud)
            out = output_dir / f"atlas_mni_slice{z_brats:03d}.png"
            img.save(str(out))
        print(f"全スライス保存完了: {output_dir}/  ({args.brats_slices}枚)")

    else:
        # 単一スライス
        if args.from_image:
            z_brats = get_slice_z_from_filename(args.from_image)
            print(f"ファイル名からスライス番号検出: z={z_brats}")
        elif args.slice_z is not None:
            z_brats = args.slice_z
        else:
            raise ValueError("--slice_z か --from_image を指定してください")

        z_mni = brats_z_to_mni_z(z_brats, args.brats_slices)
        print(f"BraTS z={z_brats} → MNI152 z={z_mni}  (flip_lr={flip_lr}, flip_ud={args.flip_ud})")
        img = extract_mni_slice(vol, z_mni, args.image_size, flip_lr=flip_lr, flip_ud=args.flip_ud)
        img.save(args.output)
        print(f"アトラス保存: {args.output}  ({args.image_size}x{args.image_size}px)")


if __name__ == "__main__":
    main()
