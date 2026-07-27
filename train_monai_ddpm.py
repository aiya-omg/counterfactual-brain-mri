"""
train_monai_ddpm.py
MONAI DiffusionModelUNet で正常脳MRIスライスを学習する。

diffusers の UNet2DModel との違い:
  - 医療画像専用設計（1ch グレースケールネイティブ）
  - ResNet + Cross-Attention ブロックが医療用チューニング済み
  - MONAI の前処理パイプラインと相性が良い
  - VAE なし・ピクセル空間で直接拡散（SD と異なる）

使い方:
  conda activate cfmri
  pip install monai-generative   # 初回のみ
  python train_monai_ddpm.py --normal_dir ./data/slices/normal --output_dir ./monai_ddpm_model

推論:
  python demo.py --input_image ./data/slices/tumor/xxx.png \\
                 --gt_mask ./data/slices/masks/xxx.png \\
                 --use_monai --monai_path ./monai_ddpm_model
"""

import argparse
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# MONAI UNet（医療画像専用）
try:
    from generative.networks.nets import DiffusionModelUNet
    MONAI_GEN = "generative"
except ImportError:
    try:
        from monai.networks.nets import DiffusionModelUNet
        MONAI_GEN = "monai"
    except ImportError:
        raise ImportError("MONAI Generative が見つかりません。pip install monai-generative")

# スケジューラは diffusers を使う（MONAI版より高速・安定）
from diffusers import DDPMScheduler

print(f"MONAI UNet ({MONAI_GEN}) + diffusers DDPMScheduler")


# ── データセット ─────────────────────────────────────────────────────────────────

class NormalMRIDataset(Dataset):
    """
    正常脳MRIスライスのデータセット。
    グレースケール1ch、[-1, 1] に正規化して返す。
    augment=True のとき水平flip・小回転・輝度ジッタを適用。
    """

    def __init__(self, image_dir: str, image_size: int = 128, augment: bool = True):
        self.paths = sorted(Path(image_dir).glob("*.png"))
        self.image_size = image_size
        self.augment = augment
        if len(self.paths) == 0:
            raise FileNotFoundError(f"PNGが見つかりません: {image_dir}")
        print(f"  正常スライス: {len(self.paths)} 枚  (augment={augment})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")  # グレースケール
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        if self.augment:
            # 水平flip（左右対称な脳に有効）
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # 小回転 ±8度（スキャナの傾き差を模倣）
            angle = random.uniform(-8, 8)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            # 輝度ジッタ ±12%（スキャナ間の輝度差を模倣）
            factor = random.uniform(0.88, 1.12)
            img = ImageEnhance.Brightness(img).enhance(factor)

        arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        return torch.from_numpy(arr).unsqueeze(0)              # (1, H, W)


# ── モデル定義 ────────────────────────────────────────────────────────────────────

def build_model(image_size: int = 128) -> DiffusionModelUNet:
    """
    MONAI DiffusionModelUNet を構築。
    RTX 4070 (12GB) に収まる軽量設定。
      - 画像サイズ: 128x128
      - チャンネル: (32, 64, 128, 128) — 小さめ
      - アテンション: 最深層のみ
      - バッチサイズ: 4
    """
    import inspect
    sig = inspect.signature(DiffusionModelUNet.__init__)
    params = sig.parameters.keys()

    ch_key = "num_channels" if "num_channels" in params else "channels"
    head_kwargs = {"num_head_channels": 16} if "num_head_channels" in params else {"num_heads": 4}

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        **{ch_key: (32, 64, 128, 128)},     # 軽量化
        attention_levels=(False, False, False, True),  # 最深層のみ
        num_res_blocks=1,                    # 各ブロック1層
        with_conditioning=False,
        **head_kwargs,
    )
    return model


# ── 学習 ─────────────────────────────────────────────────────────────────────────

def train(
    normal_dir: str,
    output_dir: str,
    image_size: int = 128,
    num_steps: int = 800000,
    batch_size: int = 4,
    learning_rate: float = 2.5e-5,
    save_every: int = 50000,
    num_train_timesteps: int = 1000,
    augment: bool = True,
    resume_from: str = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MONAI DDPM Training - 正常脳MRI専用モデル")
    print("=" * 60)
    print(f"  デバイス    : {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"  VRAM        : {vram}GB")
    print(f"  画像サイズ  : {image_size}x{image_size} (1ch グレースケール)")
    print(f"  学習ステップ: {num_steps:,}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  データ拡張  : {augment}")
    eta_h = num_steps / (25 * 3600)
    print(f"  推定時間    : {eta_h:.1f}時間 (25 it/s 想定)")
    print()

    # ── モデル & スケジューラ ────────────────────────────────────────────
    print("[1/4] モデル初期化中...")
    model = build_model(image_size).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  パラメータ数: {total_params:.1f}M")

    if resume_from:
        ckpt = Path(resume_from)
        if not ckpt.exists():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {resume_from}")
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        print(f"  チェックポイントをロード: {ckpt}")

    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    # ── データセット ─────────────────────────────────────────────────────
    print("[2/4] データセット準備中...")
    dataset = NormalMRIDataset(normal_dir, image_size=image_size, augment=augment)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda")
    )

    # ── オプティマイザ + コサインLRスケジューラ ────────────────────────────
    print("[3/4] オプティマイザ設定中...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 後半にかけてLRをゆっくり下げる（過学習抑制・収束安定化）
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=learning_rate * 0.1)
    scaler = torch.amp.GradScaler("cuda")

    # ── 学習ループ ────────────────────────────────────────────────────────
    print("[4/4] 学習開始...\n")
    model.train()
    step = 0
    losses = []
    pbar = tqdm(total=num_steps, desc="MONAI DDPM Training")

    while step < num_steps:
        for images in dataloader:
            if step >= num_steps:
                break

            images = images.to(device)  # (B, 1, H, W)

            # ランダムタイムステップ & ノイズ付加
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, num_train_timesteps,
                (images.shape[0],), device=device
            ).long()
            # diffusers DDPMScheduler の add_noise
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # MONAI UNet でノイズ予測（新API autocast）
            with torch.amp.autocast("cuda"):
                noise_pred = model(x=noisy_images, timesteps=timesteps)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            losses.append(loss.item())
            step += 1
            pbar.update(1)
            current_lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            # 中間保存
            if step % save_every == 0:
                _save(model, scheduler, output_path, step, image_size)
                avg = sum(losses[-save_every:]) / save_every
                print(f"\n  [{step:,}/{num_steps:,}] avg_loss={avg:.4f}  lr={current_lr:.2e}")

    pbar.close()
    _save(model, scheduler, output_path, step, image_size, final=True)

    print()
    print("=" * 60)
    print(f"  学習完了! -> {output_dir}/")
    print("  推論コマンド:")
    print(f"    python demo.py --input_image <画像> --use_monai --monai_path {output_dir}")
    print("=" * 60)


def _save(model, scheduler, output_path: Path, step: int, image_size: int, final: bool = False):
    """モデルを保存する"""
    tag = "final" if final else f"step{step}"
    ckpt_dir = output_path / tag
    ckpt_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "unet.pt")

    # スケジューラ設定を JSON で保存
    sched_cfg = {
        "num_train_timesteps": scheduler.num_train_timesteps,
        "beta_start": float(scheduler.betas[0]),
        "beta_end": float(scheduler.betas[-1]),
        "beta_schedule": "linear",
    }
    with open(ckpt_dir / "scheduler_config.json", "w") as f:
        json.dump(sched_cfg, f, indent=2)

    # モデル設定も保存
    model_cfg = {"image_size": image_size}
    with open(ckpt_dir / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    # output_path 直下にも "latest" として上書き保存
    torch.save(model.state_dict(), output_path / "unet_latest.pt")
    with open(output_path / "scheduler_config.json", "w") as f:
        json.dump(sched_cfg, f, indent=2)
    with open(output_path / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    print(f"  保存: {ckpt_dir}/")


# ── エントリーポイント ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MONAI DDPM 正常脳MRI学習")
    parser.add_argument("--normal_dir",  type=str, default="./data/slices/normal",
                        help="正常MRIスライスのディレクトリ")
    parser.add_argument("--output_dir",  type=str, default="./monai_ddpm_model",
                        help="モデル保存先")
    parser.add_argument("--image_size",  type=int, default=128)
    parser.add_argument("--num_steps",   type=int, default=800000,
                        help="学習ステップ数（10時間目安: 800000）")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=2.5e-5)
    parser.add_argument("--save_every",  type=int, default=50000)
    parser.add_argument("--no_augment",   action="store_true",
                        help="データ拡張を無効にする")
    parser.add_argument("--resume_from",  type=str, default=None,
                        help="継続学習用チェックポイント (例: ./monai_ddpm_model_v2/unet_latest.pt)")
    args = parser.parse_args()

    train(
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_every=args.save_every,
        augment=not args.no_augment,
        resume_from=args.resume_from,
    )
