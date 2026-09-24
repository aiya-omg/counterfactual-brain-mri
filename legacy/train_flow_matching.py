"""
train_flow_matching.py
Conditional Flow Matching (CFM) で正常脳MRIスライスを学習する。

DDPMとの違い:
  - ノイズ予測 (ε) ではなく速度場 (ベクトル場) を予測
  - t ~ Uniform(0, 1)  →  x_t = (1-t)・noise + t・data
  - 損失: ||v_θ(x_t, t) - (data - noise)||²
  - 推論: Euler ODE（DDPM比で少ないステップで高品質な生成が期待できる）

DDPMとの対応関係:
  - DDPMの t_noise=400 (0〜1000) ≈ FM の t_start=0.4 (0〜1)
  - DDPMのデノイズステップ数 ≈ FM の ODE 積分ステップ数

アーキテクチャ: train_monai_ddpm.py と同じ MONAI DiffusionModelUNet を流用。
              タイムステップ入力を [0,1] float → int(t * 999) に変換して渡す。

使い方:
  conda activate cfmri
  python train_flow_matching.py \\
    --normal_dir ./data/slices/normal \\
    --output_dir ./flow_matching_model

  # チェックポイントから再開
  python train_flow_matching.py \\
    --normal_dir ./data/slices/normal \\
    --output_dir ./flow_matching_model \\
    --resume_from ./flow_matching_model/unet_latest.pt

比較推論:
  python demo.py --input_image ./data/slices/tumor/xxx.png \\
                 --gt_mask ./data/slices/masks/xxx.png \\
                 --use_flow --flow_path ./flow_matching_model --t_start 0.4

  python demo.py --input_image ./data/slices/tumor/xxx.png \\
                 --gt_mask ./data/slices/masks/xxx.png \\
                 --use_monai --monai_path ./monai_ddpm_model_v3 --t_noise 400
"""

import argparse
import json
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# MONAI UNet（DDPMと同じアーキテクチャを流用）
try:
    from generative.networks.nets import DiffusionModelUNet
    MONAI_GEN = "generative"
except ImportError:
    try:
        from monai.networks.nets import DiffusionModelUNet
        MONAI_GEN = "monai"
    except ImportError:
        raise ImportError("MONAI Generative が見つかりません。pip install monai-generative")

print(f"MONAI UNet ({MONAI_GEN}) + Flow Matching")


# ── データセット ─────────────────────────────────────────────────────────────────
# train_monai_ddpm.py と同一。DDPM 側と同条件で比較するため変更なし。

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
        img = Image.open(self.paths[idx]).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angle = random.uniform(-8, 8)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            factor = random.uniform(0.88, 1.12)
            img = ImageEnhance.Brightness(img).enhance(factor)

        arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


# ── モデル構築 ───────────────────────────────────────────────────────────────────

def build_unet(image_size: int = 128) -> DiffusionModelUNet:
    """DDPMと同じアーキテクチャ（直接比較のため）"""
    import inspect
    sig = inspect.signature(DiffusionModelUNet.__init__)
    p = sig.parameters.keys()
    ch_key = "num_channels" if "num_channels" in p else "channels"
    head_kwargs = ({"num_head_channels": 16} if "num_head_channels" in p
                   else {"num_heads": 4})

    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        **{ch_key: (32, 64, 128, 128)},
        attention_levels=(False, False, False, True),
        num_res_blocks=1,
        with_conditioning=False,
        **head_kwargs,
    )


# ── 学習ループ ───────────────────────────────────────────────────────────────────

def train(
    normal_dir: str,
    output_dir: str,
    num_steps: int = 800_000,
    batch_size: int = 4,
    lr: float = 1e-4,
    image_size: int = 128,
    save_every: int = 50_000,
    device: str = "cuda",
    resume_from: str = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── データロード ──────────────────────────────────────────────────────────
    dataset = NormalMRIDataset(normal_dir, image_size=image_size, augment=True)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)
    loader_iter = iter(loader)

    # ── モデル ────────────────────────────────────────────────────────────────
    model = build_unet(image_size).to(device)
    if resume_from:
        ckpt = Path(resume_from)
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        print(f"  チェックポイントをロード: {ckpt}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)

    # ── 設定を保存 ────────────────────────────────────────────────────────────
    config = {
        "method": "flow_matching",
        "image_size": image_size,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "lr": lr,
        "note": "CFM with linear interpolation. Timestep: t in [0,1] -> int(t*999)"
    }
    with open(output_path / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── 学習 ──────────────────────────────────────────────────────────────────
    model.train()
    loss_ema = None
    log_interval = 500
    print(f"\n  Flow Matching 学習開始 (steps={num_steps}, device={device})")
    print(f"  DDPMとの比較: 同一アーキテクチャ / 異なる学習目標")

    for step in range(1, num_steps + 1):
        # バッチ取得（無限ループ）
        try:
            x1 = next(loader_iter).to(device)
        except StopIteration:
            loader_iter = iter(loader)
            x1 = next(loader_iter).to(device)

        B = x1.shape[0]

        # ── Conditional Flow Matching ────────────────────────────────────────
        # x_0: ノイズ（t=0 側）、x_1: データ（t=1 側）
        x0 = torch.randn_like(x1)

        # t ~ U(0, 1)  ← DDPMのdiscrete timestepsに相当するもの
        t = torch.rand(B, device=device)                   # (B,)
        t_expand = t.view(B, 1, 1, 1)                      # ブロードキャスト用

        # 線形補間: x_t = (1-t)・x_0 + t・x_1
        x_t = (1.0 - t_expand) * x0 + t_expand * x1

        # 目標速度場（線形CFMでは定数）: u_t = x_1 - x_0
        u_t = x1 - x0

        # タイムステップをMONAI UNet用の整数に変換（0〜999）
        t_int = (t * 999).long()

        # 速度場を予測
        v_pred = model(x=x_t, timesteps=t_int)

        # MSE損失
        loss = F.mse_loss(v_pred, u_t)

        # 最適化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # EMA損失
        l = loss.item()
        loss_ema = l if loss_ema is None else loss_ema * 0.99 + l * 0.01

        if step % log_interval == 0:
            print(f"  step {step:7d}/{num_steps}  loss={loss_ema:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if step % save_every == 0 or step == num_steps:
            ckpt = output_path / f"unet_step{step:07d}.pt"
            torch.save(model.state_dict(), str(ckpt))
            torch.save(model.state_dict(), str(output_path / "unet_latest.pt"))
            print(f"  保存: {ckpt.name}  (loss={loss_ema:.5f})")

    print(f"\n  ✅ 学習完了!  最終 avg loss: {loss_ema:.5f}")
    print(f"  保存先: {output_path}/")


# ── エントリーポイント ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional Flow Matching 学習")
    parser.add_argument("--normal_dir",   type=str, required=True,
                        help="正常脳スライス画像のディレクトリ")
    parser.add_argument("--output_dir",   type=str, default="./flow_matching_model",
                        help="モデル保存先")
    parser.add_argument("--num_steps",    type=int, default=800_000,
                        help="学習ステップ数（DDPM版と同じ800kで比較）")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--image_size",   type=int, default=128)
    parser.add_argument("--save_every",   type=int, default=50_000)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_from",  type=str, default=None,
                        help="チェックポイントパス（例: ./flow_matching_model/unet_latest.pt）")
    args = parser.parse_args()

    train(
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        save_every=args.save_every,
        device=args.device,
        resume_from=args.resume_from,
    )
