"""
train_lora.py
正常脳MRIスライスでSD 1.5をLoRAファインチューニング

RTX 4070 (12GB VRAM) 向け最適化済み
学習時間目安: 約1〜2時間 (1000steps)

使い方:
  conda activate cfmri
  python train_lora.py --normal_dir ./data/slices/normal --output_dir ./lora_model

学習後のデモ実行:
  python demo.py --input_image ./data/slices/tumor/xxx.png --use_real_sd --lora_path ./lora_model
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── データセット ────────────────────────────────────────────────────────────────

class MRIDataset(Dataset):
    """正常脳MRIスライスのデータセット"""

    PROMPT = "healthy normal brain MRI scan, FLAIR, no tumor, no lesion, normal white matter"

    def __init__(self, image_dir: str, image_size: int = 512):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]に正規化
        ])

        print(f"データセット: {len(self.image_paths)} 枚")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "prompt": self.PROMPT,
        }


# ── LoRAレイヤー ────────────────────────────────────────────────────────────────

class LoRALinearLayer(torch.nn.Module):
    """LoRA差分行列 (低ランク分解)"""

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.down = torch.nn.Linear(in_features, rank, bias=False)
        self.up   = torch.nn.Linear(rank, out_features, bias=False)
        self.scale = 1.0

        torch.nn.init.kaiming_uniform_(self.down.weight)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x)) * self.scale


def inject_lora(unet, rank: int = 4):
    """UNetのattentionレイヤーにLoRAを注入"""
    lora_layers = {}

    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(key in name for key in ["to_q", "to_k", "to_v", "to_out.0"]):
                lora = LoRALinearLayer(
                    module.in_features,
                    module.out_features,
                    rank=rank
                ).to(module.weight.device, dtype=module.weight.dtype)
                lora_layers[name] = lora

    return lora_layers


def apply_lora(unet, lora_layers):
    """Forward hook でLoRA差分を加算"""
    hooks = []
    for name, module in unet.named_modules():
        if name in lora_layers:
            lora = lora_layers[name]

            def make_hook(lora_layer):
                def hook(module, input, output):
                    return output + lora_layer(input[0])
                return hook

            hooks.append(module.register_forward_hook(make_hook(lora)))
    return hooks


# ── 学習ループ ──────────────────────────────────────────────────────────────────

def train(
    normal_dir: str,
    output_dir: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    image_size: int = 512,
    rank: int = 4,
    batch_size: int = 1,
    num_steps: int = 1000,
    learning_rate: float = 1e-4,
    save_every: int = 200,
    mixed_precision: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if mixed_precision else torch.float32

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LoRA Fine-tuning - 正常脳MRI特化")
    print("=" * 60)
    print(f"  デバイス: {device}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    print(f"  学習ステップ数: {num_steps}")
    print(f"  LoRAランク: {rank}")
    print()

    # ── モデルロード ───────────────────────────────────────────
    print("[1/4] モデルロード中...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.enable_attention_slicing()

    unet       = pipe.unet
    vae        = pipe.vae
    tokenizer  = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler  = pipe.scheduler

    # UNetのベース重みを固定（LoRAのみ学習）
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ── LoRA注入 ────────────────────────────────────────────────
    print("[2/4] LoRAレイヤーを注入中...")
    lora_layers = inject_lora(unet, rank=rank)
    hooks = apply_lora(unet, lora_layers)

    # LoRAパラメータをfp32で学習（NaN防止）
    lora_params = []
    for lora in lora_layers.values():
        lora.float()  # fp32に変換
        lora_params.extend(lora.parameters())

    total_params = sum(p.numel() for p in lora_params)
    print(f"  学習パラメータ数: {total_params:,}")

    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_steps,
    )

    # GradScaler（混合精度学習の安定化）
    scaler = torch.cuda.amp.GradScaler()

    # ── データセット ────────────────────────────────────────────
    print("[3/4] データセット準備中...")
    dataset    = MRIDataset(normal_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # プロンプトの埋め込みを事前計算
    text_inputs = tokenizer(
        [MRIDataset.PROMPT],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

    # ── 学習ループ ──────────────────────────────────────────────
    print("[4/4] 学習開始...")
    print()

    unet.train()
    step = 0
    losses = []

    pbar = tqdm(total=num_steps, desc="Training")

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=dtype)

            # 画像を潜在空間にエンコード
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # ランダムノイズとタイムステップ
            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (batch_size,), device=device).long()

            # ノイズを加えた潜在変数
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # UNetでノイズ予測（amp使用）
            embeds = text_embeds.expand(batch_size, -1, -1)
            with torch.cuda.amp.autocast():
                noise_pred = unet(noisy_latents, timesteps, embeds).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            losses.append(loss.item())
            step += 1

            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

            # 中間保存
            if step % save_every == 0:
                save_lora(lora_layers, output_path / f"lora_step{step}.pt")
                print(f"\n  [{step}/{num_steps}] 保存完了 | 平均Loss: {sum(losses[-save_every:])/save_every:.4f}")

    pbar.close()

    # 最終保存
    save_lora(lora_layers, output_path / "lora_final.pt")

    # hookを削除
    for h in hooks:
        h.remove()

    print()
    print("=" * 60)
    print(f"  学習完了! -> {output_dir}/lora_final.pt")
    print("=" * 60)
    print()
    print("次のコマンドで推論:")
    print(f"  python demo.py --input_image <画像パス> --use_real_sd --lora_path {output_dir}/lora_final.pt")


def save_lora(lora_layers, path):
    """LoRA重みを保存"""
    state_dict = {}
    for name, lora in lora_layers.items():
        state_dict[f"{name}.lora_down"] = lora.down.weight.cpu()
        state_dict[f"{name}.lora_up"]   = lora.up.weight.cpu()
    torch.save(state_dict, path)


# ── エントリーポイント ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Brain MRI")
    parser.add_argument("--normal_dir",   type=str, default="./data/slices/normal",
                        help="正常MRIスライスのディレクトリ")
    parser.add_argument("--output_dir",   type=str, default="./lora_model",
                        help="LoRA重みの保存先")
    parser.add_argument("--model_id",     type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--image_size",   type=int, default=512)
    parser.add_argument("--rank",         type=int, default=4,
                        help="LoRAランク (4〜16、大きいほど表現力↑・VRAM↑)")
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--num_steps",    type=int, default=1000)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--save_every",   type=int, default=200)
    args = parser.parse_args()

    train(
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        image_size=args.image_size,
        rank=args.rank,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        save_every=args.save_every,
    )
