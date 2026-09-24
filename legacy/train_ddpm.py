"""
train_ddpm.py
正常脳MRIスライスで軽量DDPMを学習する。

学習したモデルを counterfactual 生成に使う:
  腫瘍MRI → 部分ノイズ追加(t) → 正常モデルでデノイズ → 差分=腫瘍領域

使い方:
  conda activate cfmri
  python train_ddpm.py --normal_dir ./data/slices/normal --output_dir ./ddpm_model
  (RTX 4070で約30〜60分 / 5000steps)

推論:
  python demo.py --input_image ./data/slices/tumor/xxx.png --use_ddpm --ddpm_path ./ddpm_model
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler


# ── データセット ─────────────────────────────────────────────────────────────────

class NormalMRIDataset(Dataset):
    """正常脳MRIスライスのみで構成するデータセット"""

    def __init__(self, image_dir: str, image_size: int = 256):
        self.paths = sorted(Path(image_dir).glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),                      # MRIはグレースケール
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),          # [-1, 1]
        ])
        print(f"正常スライス数: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img)


# ── 学習 ─────────────────────────────────────────────────────────────────────────

def train(
    normal_dir: str,
    output_dir: str,
    image_size: int = 256,
    num_steps: int = 5000,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    save_every: int = 1000,
    num_train_timesteps: int = 1000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DDPM Training - 正常脳MRI専用モデル")
    print("=" * 60)
    print(f"  デバイス: {device}")
    print(f"  画像サイズ: {image_size}x{image_size} (グレースケール)")
    print(f"  学習ステップ: {num_steps}")
    print()

    # ── モデル定義 ─────────────────────────────────────────────
    # UNet2DModel: diffusers の軽量UNet（テキスト条件なし）
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,          # グレースケール
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),   # RTX 4070でVRAM収まるサイズ
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  モデルパラメータ: {total_params:.1f}M")

    # ── スケジューラ ────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
    )

    # ── データ ─────────────────────────────────────────────────
    dataset = NormalMRIDataset(normal_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # ── オプティマイザ ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=num_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ── 学習ループ ──────────────────────────────────────────────
    print("[Training]")
    model.train()
    step = 0
    losses = []

    pbar = tqdm(total=num_steps)

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            images = batch.to(device)  # (B, 1, H, W)

            # ランダムタイムステップでノイズを加える（拡散過程の前向きステップ）
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, num_train_timesteps, (images.shape[0],), device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # UNetでノイズ予測
            with torch.cuda.amp.autocast():
                noise_pred = model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

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
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.1e}",
            })

            # 中間保存
            if step % save_every == 0:
                ckpt_path = output_path / f"unet_step{step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                avg = sum(losses[-save_every:]) / save_every
                print(f"\n  [{step}/{num_steps}] 保存: {ckpt_path.name}  avg_loss={avg:.4f}")

    pbar.close()

    # 最終保存（diffusers形式）
    model.save_pretrained(output_path / "unet")
    noise_scheduler.save_pretrained(output_path / "scheduler")
    print()
    print("=" * 60)
    print(f"  完了! -> {output_dir}")
    print("  推論コマンド:")
    print(f"    python demo.py --input_image <画像> --use_ddpm --ddpm_path {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_dir",  type=str, default="./data/slices/normal")
    parser.add_argument("--output_dir",  type=str, default="./ddpm_model")
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--num_steps",   type=int, default=5000)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--save_every",  type=int, default=1000)
    args = parser.parse_args()

    train(
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_every=args.save_every,
    )
