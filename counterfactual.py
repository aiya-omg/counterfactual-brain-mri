"""
counterfactual.py
Counterfactual Visual Attribution パイプライン

アルゴリズム概要:
  1. 腫瘍MRI画像 → VAEエンコードで潜在変数 z_0 を取得
  2. DDIM Inversion: z_0 → z_T（ノイズ空間へ逆行）
  3. DDIMデノイズ: z_T を「健康な脳MRI」プロンプトで再生成
  4. 生成された反実仮想画像と元画像の差分 → 病変マップ

使い方:
  python counterfactual.py \
    --input_dir ./data/slices/tumor \
    --output_dir ./results \
    --num_images 10
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Tuple

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
)
from diffusers.utils import load_image


# ─── DDIM Inversion ────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_inversion(
    pipeline: StableDiffusionPipeline,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
) -> List[torch.Tensor]:
    """
    DDIM Inversion: 実画像の潜在変数からノイズ軌跡を逆算する

    Returns:
        all_latents: [z_0, z_1, ..., z_T] のリスト（逆方向のノイズ軌跡）
    """
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.flip(0)  # 逆順（0→T方向）

    all_latents = [latents.clone()]
    current_latents = latents.clone()

    for t in tqdm(timesteps, desc="DDIM Inversion", leave=False):
        # UNetでノイズ予測
        latent_input = torch.cat([current_latents] * 2)
        noise_pred = pipeline.unet(
            latent_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample

        # CFGなし（inversion時はguidance_scale=1）
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # 次のステップの潜在変数を計算（逆方向）
        current_latents = ddim_step_forward(
            scheduler, noise_pred, t, current_latents
        )
        all_latents.append(current_latents.clone())

    return all_latents


def ddim_step_forward(
    scheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    """DDIM 逆方向ステップ（x_t → x_{t+1}）"""
    prev_timestep = (
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 予測x_0
    pred_original_sample = (
        sample - beta_prod_t**0.5 * model_output
    ) / alpha_prod_t**0.5

    # 次のサンプル方向
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output

    prev_sample = (
        alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
    )
    return prev_sample


# ─── メインパイプライン ─────────────────────────────────────────────────────────

class CounterfactualPipeline:
    """
    医療画像の反実仮想（Counterfactual）生成パイプライン

    RTX 4070 (12GB VRAM) 向け最適化:
    - fp16精度
    - xformers memory efficient attention
    - gradient checkpointing不要（推論のみ）
    """

    # テキストプロンプト
    ABNORMAL_PROMPT = (
        "brain MRI scan showing tumor, glioma, lesion, abnormal tissue, "
        "high signal intensity, mass effect"
    )
    NORMAL_PROMPT = (
        "healthy normal brain MRI scan, no tumor, no lesion, "
        "normal white matter, normal gray matter, symmetric"
    )
    NEGATIVE_PROMPT = (
        "artifacts, noise, low quality, blurry, distorted, "
        "tumor, lesion, abnormal"
    )

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_size: int = 512,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size

        print(f"🔄 モデルをロード中: {model_id}")
        self.pipeline = self._load_pipeline(model_id)
        print("✅ モデルロード完了")

    def _load_pipeline(self, model_id: str) -> StableDiffusionPipeline:
        """パイプラインをVRAM効率化設定でロード"""
        # DDIMスケジューラに差し替え（Inversion用）
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            safety_checker=None,  # 医療画像はNSFW検出不要
            requires_safety_checker=False,
        ).to(self.device)

        # xformers（VRAM節約・高速化）
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  xformers: 有効")
        except Exception:
            print("  xformers: 無効（インストール推奨）")

        pipe.enable_attention_slicing()

        return pipe

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """PIL画像 → 潜在変数"""
        image = image.resize((self.image_size, self.image_size))
        image_tensor = (
            torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        )
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=torch.float16
        )
        latents = self.pipeline.vae.encode(image_tensor).latent_dist.mean
        latents = latents * self.pipeline.vae.config.scaling_factor
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """潜在変数 → PIL画像"""
        latents = latents / self.pipeline.vae.config.scaling_factor
        with torch.no_grad():
            image = self.pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((image * 255).astype(np.uint8))

    def _get_prompt_embeds(
        self, prompt: str, negative_prompt: str
    ) -> torch.Tensor:
        """テキストプロンプト → 埋め込みベクトル"""
        text_inputs = self.pipeline.tokenizer(
            [negative_prompt, prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeds = self.pipeline.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        return embeds

    @torch.no_grad()
    def generate_counterfactual(
        self,
        tumor_image: Image.Image,
        source_prompt: Optional[str] = None,
        target_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        腫瘍MRI画像から反実仮想（正常）画像を生成

        Args:
            tumor_image: 腫瘍を含む脳MRI画像
            source_prompt: 元画像の説明（デフォルト: 腫瘍プロンプト）
            target_prompt: 生成目標の説明（デフォルト: 正常脳プロンプト）
            negative_prompt: 除外したい要素のプロンプト

        Returns:
            (counterfactual_image, difference_map): 反実仮想画像と差分マップのタプル
        """
        source_prompt = source_prompt or self.ABNORMAL_PROMPT
        target_prompt = target_prompt or self.NORMAL_PROMPT
        negative_prompt = negative_prompt or self.NEGATIVE_PROMPT

        # 1. 画像を潜在空間にエンコード
        latents = self._encode_image(tumor_image)

        # 2. プロンプト埋め込みを取得
        source_embeds = self._get_prompt_embeds(source_prompt, negative_prompt)
        target_embeds = self._get_prompt_embeds(target_prompt, negative_prompt)

        # 3. DDIM Inversion（実画像 → ノイズ空間）
        inverted_latents = ddim_inversion(
            self.pipeline,
            latents,
            source_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=1.0,
        )
        noisy_latents = inverted_latents[-1]  # z_T（最終ノイズ）

        # 4. 正常プロンプトでデノイズ（ノイズ → 反実仮想）
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)

        current_latents = noisy_latents
        for t in tqdm(
            self.pipeline.scheduler.timesteps,
            desc="反実仮想生成",
            leave=False,
        ):
            latent_input = torch.cat([current_latents] * 2)
            noise_pred = self.pipeline.unet(
                latent_input,
                t,
                encoder_hidden_states=target_embeds,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            current_latents = self.pipeline.scheduler.step(
                noise_pred, t, current_latents
            ).prev_sample

        # 5. 潜在変数 → 画像にデコード
        counterfactual_image = self._decode_latents(current_latents)

        # 6. 差分マップ生成
        difference_map = compute_difference_map(
            tumor_image.resize((self.image_size, self.image_size)),
            counterfactual_image,
        )

        return counterfactual_image, difference_map

    @torch.no_grad()
    def generate_counterfactual_img2img(
        self,
        tumor_image: Image.Image,
        strength: float = 0.55,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        target_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        img2img方式で反実仮想を生成する。

        元画像にpartialなノイズを加えて（strength で制御）、
        「正常脳MRI」プロンプトでデノイズする。
        DDIM inversionより構造が保たれやすい。

        Args:
            tumor_image    : 腫瘍を含む脳MRI画像
            strength       : 0.0=変化なし〜1.0=完全再生成。0.4〜0.65が推奨。
            guidance_scale : CFGスケール（高いほどプロンプトに忠実）
            num_inference_steps: デノイズステップ数
            target_prompt  : 生成目標プロンプト
            negative_prompt: ネガティブプロンプト

        Returns:
            (counterfactual_image, difference_map)
        """
        target_prompt   = target_prompt   or self.NORMAL_PROMPT
        negative_prompt = negative_prompt or self.NEGATIVE_PROMPT

        # img2imgパイプラインを遅延ロード（VRAMを節約するため必要時のみ）
        if not hasattr(self, "_img2img_pipeline"):
            print("  img2imgパイプラインを初期化中...")
            self._img2img_pipeline = StableDiffusionImg2ImgPipeline(
                vae=self.pipeline.vae,
                text_encoder=self.pipeline.text_encoder,
                tokenizer=self.pipeline.tokenizer,
                unet=self.pipeline.unet,
                scheduler=self.pipeline.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            self._img2img_pipeline.to(self.device)

        # リサイズ
        input_image = tumor_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        ).convert("RGB")

        # img2img実行（内部でノイズ追加→デノイズを行う）
        result = self._img2img_pipeline(
            prompt=target_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        # 差分マップ生成
        difference_map = compute_difference_map(input_image, result)

        return result, difference_map


# ─── 差分マップ計算 ─────────────────────────────────────────────────────────────

def compute_difference_map(
    original: Image.Image,
    counterfactual: Image.Image,
    smooth_sigma: float = 2.0,
) -> Image.Image:
    """
    元画像と反実仮想画像の差分から病変マップを生成

    Args:
        original: 元の腫瘍MRI画像
        counterfactual: 生成された正常画像
        smooth_sigma: ガウシアンスムージングのシグマ

    Returns:
        差分マップ画像（グレースケール、明るい部分＝病変候補）
    """
    from scipy.ndimage import gaussian_filter

    orig_arr = np.array(original).astype(float)
    cf_arr = np.array(counterfactual).astype(float)

    # チャンネル方向の絶対差分
    diff = np.abs(orig_arr - cf_arr).mean(axis=2)

    # スムージングで小さなノイズを除去
    diff_smooth = gaussian_filter(diff, sigma=smooth_sigma)

    # 0-255にノーマライズ
    if diff_smooth.max() > 0:
        diff_norm = (diff_smooth / diff_smooth.max() * 255).astype(np.uint8)
    else:
        diff_norm = np.zeros_like(diff_smooth, dtype=np.uint8)

    return Image.fromarray(diff_norm)


# ─── 正常MRI学習済みDDPMによる反実仮想 ────────────────────────────────────────────

class DDPMCounterfactual:
    """
    正常脳MRIで学習したDDPMを使う反実仮想生成。

    アルゴリズム:
      1. 腫瘍MRI を潜在空間ではなくピクセル空間でノイズ追加 (t_noise step)
      2. 正常脳のみで学習したUNetでデノイズ
         → モデルは正常脳しか知らないので、腫瘍部分が「正常」に補正される
      3. 差分マップ = 腫瘍領域

    t_noise の調整:
      低い(200) → 元画像に近い。腫瘍が残りやすい
      高い(600) → より正常化される。全体的な構造が変わりすぎる危険
      推奨: 300〜500
    """

    def __init__(self, ddpm_path: str, device: str = "cuda", image_size: int = 256):
        self.device = device
        self.image_size = image_size

        print(f"  DDPMモデルをロード中: {ddpm_path}")
        from diffusers import UNet2DModel, DDPMScheduler

        self.unet = UNet2DModel.from_pretrained(
            f"{ddpm_path}/unet"
        ).to(device)
        self.unet.eval()

        self.scheduler = DDPMScheduler.from_pretrained(
            f"{ddpm_path}/scheduler"
        )
        print("  ロード完了")

    @torch.no_grad()
    def generate(
        self,
        tumor_image: Image.Image,
        t_noise: int = 400,
        num_inference_steps: int = 1000,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        腫瘍MRI → 部分ノイズ追加 → 正常モデルでデノイズ → 差分

        Args:
            tumor_image        : 腫瘍MRI (PIL)
            t_noise            : ノイズを加えるタイムステップ (300〜500推奨)
            num_inference_steps: デノイズステップ数

        Returns:
            (counterfactual_image, difference_map)
        """
        from scipy.ndimage import gaussian_filter

        # グレースケール変換 & テンソル化
        img_gray = tumor_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        ).convert("L")
        img_np = np.array(img_gray).astype(np.float32) / 127.5 - 1.0  # [-1,1]
        x0 = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # ── 前向き拡散: x0 に t_noise ステップ分のノイズを加える ──
        self.scheduler.set_timesteps(num_inference_steps)
        noise = torch.randn_like(x0)
        t_tensor = torch.tensor([t_noise], device=self.device, dtype=torch.long)
        x_t = self.scheduler.add_noise(x0, noise, t_tensor)

        # ── 逆向き拡散: t_noise → 0 を正常モデルでデノイズ ──
        # スケジューラのタイムステップから t_noise 以降のものを使う
        timesteps = [t for t in self.scheduler.timesteps if t <= t_noise]

        x = x_t
        for t in tqdm(timesteps, desc="DDPM denoising", leave=False):
            t_batch = torch.tensor([t], device=self.device, dtype=torch.long)
            noise_pred = self.unet(x, t_batch).sample
            x = self.scheduler.step(noise_pred, t, x).prev_sample

        # テンソル → PIL (グレースケール → RGB)
        cf_np = ((x.squeeze().cpu().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        cf_rgb = np.stack([cf_np] * 3, axis=2)
        counterfactual_image = Image.fromarray(cf_rgb)

        # 差分マップ
        orig_np = np.array(tumor_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        ).convert("L")).astype(float)
        diff = gaussian_filter(np.abs(orig_np - cf_np.astype(float)), sigma=2.0)
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        difference_map = Image.fromarray(diff)

        return counterfactual_image, difference_map


# ─── MONAI DiffusionModelUNet による反実仮想 ──────────────────────────────────────

class MONAICounterfactual:
    """
    MONAI DiffusionModelUNet で学習した正常脳MRI専用モデルによる反実仮想生成。

    diffusers の UNet2DModel との違い:
      - 医療画像専用アーキテクチャ（1ch グレースケールネイティブ）
      - ResNet + アテンションが医療用チューニング済み
      - ピクセル空間直接拡散（VAE なし → MRI 統計が保たれる）

    アルゴリズム:
      1. 腫瘍MRI を 1ch グレースケールに変換
      2. t_noise ステップ分の前向き拡散（ノイズ追加）
      3. 正常脳のみで学習した MONAI UNet でデノイズ
      4. 差分マップ = 腫瘍領域
    """

    def __init__(self, monai_path: str, device: str = "cuda", image_size: int = 256):
        import json

        self.device = device
        self.image_size = image_size

        print(f"  MONAIモデルをロード中: {monai_path}")

        try:
            from generative.networks.nets import DiffusionModelUNet
        except ImportError:
            from monai.networks.nets import DiffusionModelUNet
        from diffusers import DDPMScheduler as MonaiDDPMScheduler

        # モデル設定ロード
        cfg_path = Path(monai_path) / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            image_size = cfg.get("image_size", image_size)

        # UNet 構築 & 重みロード（バージョン差異を吸収）
        import inspect
        sig = inspect.signature(DiffusionModelUNet.__init__)
        p = sig.parameters.keys()
        ch_key = "num_channels" if "num_channels" in p else "channels"
        head_kwargs = {"num_head_channels": 16} if "num_head_channels" in p else {"num_heads": 4}

        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            **{ch_key: (32, 64, 128, 128)},
            attention_levels=(False, False, False, True),
            num_res_blocks=1,
            with_conditioning=False,
            **head_kwargs,
        ).to(device)

        unet_path = Path(monai_path) / "unet_latest.pt"
        self.unet.load_state_dict(
            torch.load(str(unet_path), map_location=device)
        )
        self.unet.eval()

        # スケジューラ設定ロード
        sched_path = Path(monai_path) / "scheduler_config.json"
        if sched_path.exists():
            with open(sched_path) as f:
                sched_cfg = json.load(f)
            self.scheduler = MonaiDDPMScheduler(
                num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000)
            )
        else:
            self.scheduler = MonaiDDPMScheduler(num_train_timesteps=1000)

        print("  ロード完了")

    @torch.no_grad()
    def generate(
        self,
        tumor_image: Image.Image,
        t_noise: int = 400,
        num_inference_steps: int = 1000,
        atlas_path: str = None,
        atlas_strength: float = 0.0,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        腫瘍MRI → 部分ノイズ追加 → MONAI正常モデルでデノイズ → 差分

        Args:
            tumor_image    : 腫瘍MRI (PIL)
            t_noise        : ノイズ追加ステップ (300〜500推奨)
            atlas_path     : 正常脳アトラス画像のパス (Noneで無効)
            atlas_strength : アトラスへの引き寄せ強度 (0.0〜1.0)
                             0.0 = アトラス無効（従来と同じ）
                             0.3 = 弱めのガイド（推奨）
                             1.0 = アトラスに完全に引き寄せ

        Returns:
            (counterfactual_image, difference_map)
        """
        from scipy.ndimage import gaussian_filter

        # グレースケール → テンソル [-1, 1]
        img_gray = tumor_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        ).convert("L")
        img_np = np.array(img_gray).astype(np.float32) / 127.5 - 1.0
        x0 = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # アトラスをテンソルに変換 [-1, 1]（脳輪郭を患者画像に自動アライメント）
        atlas_tensor = None
        if atlas_path and atlas_strength > 0:
            from build_atlas_mni import align_atlas_to_image
            atlas_img = Image.open(atlas_path).convert("L").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            atlas_arr  = np.array(atlas_img).astype(np.float32) / 255.0
            target_arr = (img_np + 1.0) / 2.0  # [-1,1] → [0,1]
            atlas_arr  = align_atlas_to_image(atlas_arr, target_arr)
            atlas_np_aligned = atlas_arr * 2.0 - 1.0  # [0,1] → [-1,1]
            atlas_tensor = torch.from_numpy(atlas_np_aligned).unsqueeze(0).unsqueeze(0).to(self.device)

        # 異常領域（高輝度）の自動検出
        img_01 = (img_np + 1.0) / 2.0  # [0,1]
        # 背景を確実に除外：画像最大値の10%以上を脳領域とする
        brain_mask_np = img_01 > (img_01.max() * 0.1)
        if brain_mask_np.sum() > 100:
            brain_vals = img_01[brain_mask_np]
            # 脳ピクセルの上位5%を異常候補とする（σベースはT2-FLAIRで破綻するため）
            threshold = np.percentile(brain_vals, 95)
            anomaly_np = (brain_mask_np & (img_01 > threshold)).astype(np.float32)
        else:
            anomaly_np = np.zeros_like(img_01)
        anomaly_mask = torch.from_numpy(anomaly_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # 前向き拡散
        self.scheduler.set_timesteps(num_inference_steps)
        noise = torch.randn_like(x0)
        t_tensor = torch.tensor([t_noise], device=self.device, dtype=torch.long)

        # 全体をt_noiseでノイズ追加
        x_t = self.scheduler.add_noise(x0, noise, t_tensor)

        # 異常領域だけ純粋なランダムノイズに差し替え（RePaintスタイル）
        # → UNetはt_noiseで一貫してデノイズできる。異常領域は周囲の文脈から正常脳として補完される
        x_t = x_t * (1 - anomaly_mask) + torch.randn_like(x0) * anomaly_mask

        # 逆向き拡散: t_noise → 0（タイムステップは一貫してt_noise基準）
        timesteps = [t for t in self.scheduler.timesteps if t <= t_noise]
        x = x_t
        for t in tqdm(timesteps, desc="MONAI denoising", leave=False):
            t_batch = torch.tensor([t], device=self.device, dtype=torch.long)
            noise_pred = self.unet(x=x, timesteps=t_batch)

            if atlas_tensor is not None:
                # アトラスガイド：各ステップでx̂_0予測にアトラスをブレンド
                # x̂_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
                alpha_bar = self.scheduler.alphas_cumprod[t].to(self.device)
                x0_pred = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
                # アトラスとブレンド
                x0_guided = (1 - atlas_strength) * x0_pred + atlas_strength * atlas_tensor
                # ブレンド後のx̂_0からノイズ予測を逆算
                noise_pred = (x - alpha_bar.sqrt() * x0_guided) / (1 - alpha_bar).sqrt()

            x = self.scheduler.step(noise_pred, t, x).prev_sample

        # テンソル → PIL (グレースケール → RGB)
        cf_np = ((x.squeeze().cpu().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        cf_rgb = np.stack([cf_np] * 3, axis=2)
        counterfactual_image = Image.fromarray(cf_rgb)

        # 差分マップ
        orig_np = np.array(img_gray).astype(float)
        diff = gaussian_filter(np.abs(orig_np - cf_np.astype(float)), sigma=2.0)
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        difference_map = Image.fromarray(diff.astype(np.uint8))

        return counterfactual_image, difference_map


# ─── Flow Matching による反実仮想 ──────────────────────────────────────────────

class FlowMatchingCounterfactual:
    """
    Conditional Flow Matching (CFM) で学習した正常脳MRI専用モデルによる反実仮想生成。

    DDPMとの違い:
      - 学習目標: ノイズ予測 (ε) → 速度場予測 (v = x_data - x_noise)
      - 推論:     DDPMのマルコフ連鎖 → 決定論的ODE (Euler法)
      - パラメータ: t_noise (0〜1000整数) → t_start (0.0〜1.0小数)

    CFMの時刻定義:
      t=0: 純粋ノイズ分布 N(0,I)
      t=1: 正常脳MRI データ分布

    反実仮想生成の手順:
      1. 腫瘍MRI を (1-t_start)·noise + t_start·tumor に補間（部分腐敗）
      2. 異常領域（高輝度）は純粋ノイズに差し替え（Anomaly-Aware版）
      3. Euler ODE: t_start → 1.0 方向に積分
         dx/dt = v_θ(x, t)
      4. 差分マップ = |tumor - counterfactual|

    t_start の目安:
      0.2〜0.3 → 元画像に近い。腫瘍が残りやすい
      0.4〜0.5 → バランス（推奨）
      0.6〜0.8 → より正常化される。解剖学的構造が変わりやすい
    """

    def __init__(self, flow_path: str, device: str = "cuda", image_size: int = 128):
        import json

        self.device = device
        self.image_size = image_size

        print(f"  Flow Matchingモデルをロード中: {flow_path}")

        try:
            from generative.networks.nets import DiffusionModelUNet
        except ImportError:
            from monai.networks.nets import DiffusionModelUNet

        # モデル設定ロード
        cfg_path = Path(flow_path) / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.image_size = cfg.get("image_size", image_size)

        # MONAI UNet（DDPMと同一アーキテクチャ）
        import inspect
        sig = inspect.signature(DiffusionModelUNet.__init__)
        p = sig.parameters.keys()
        ch_key = "num_channels" if "num_channels" in p else "channels"
        head_kwargs = ({"num_head_channels": 16} if "num_head_channels" in p
                       else {"num_heads": 4})

        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            **{ch_key: (32, 64, 128, 128)},
            attention_levels=(False, False, False, True),
            num_res_blocks=1,
            with_conditioning=False,
            **head_kwargs,
        ).to(device)

        unet_path = Path(flow_path) / "unet_latest.pt"
        self.unet.load_state_dict(
            torch.load(str(unet_path), map_location=device)
        )
        self.unet.eval()
        print("  ロード完了")

    @torch.no_grad()
    def generate(
        self,
        tumor_image: Image.Image,
        t_start: float = 0.4,
        num_steps: int = 100,
        atlas_path: str = None,
        atlas_strength: float = 0.0,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        腫瘍MRI → 部分腐敗 → Euler ODE で正常化 → 差分マップ

        Args:
            tumor_image   : 腫瘍MRI (PIL)
            t_start       : 腐敗の程度 (0.0〜1.0)。DDPMのt_noise/1000に相当。
                            0.4 ≈ DDPM t_noise=400 に相当
            num_steps     : Euler法の積分ステップ数（DDPMより少なくて済む）
            atlas_path    : MNI152アトラス画像のパス (Noneで無効)
            atlas_strength: アトラスへの引き寄せ強度 (0.0〜1.0)

        Returns:
            (counterfactual_image, difference_map)
        """
        from scipy.ndimage import gaussian_filter

        # グレースケール → テンソル [-1, 1]
        img_gray = tumor_image.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        ).convert("L")
        img_np = np.array(img_gray).astype(np.float32) / 127.5 - 1.0
        x1_tumor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # アトラスをテンソルに変換（DDPMと同じアライメント処理）
        atlas_tensor = None
        if atlas_path and atlas_strength > 0:
            from build_atlas_mni import align_atlas_to_image
            atlas_img = Image.open(atlas_path).convert("L").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            atlas_arr  = np.array(atlas_img).astype(np.float32) / 255.0
            target_arr = (img_np + 1.0) / 2.0
            atlas_arr  = align_atlas_to_image(atlas_arr, target_arr)
            atlas_tensor = torch.from_numpy(atlas_arr * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)

        # 異常領域の検出（DDPMと同じパーセンタイル法）
        img_01 = (img_np + 1.0) / 2.0
        brain_mask_np = img_01 > (img_01.max() * 0.1)
        if brain_mask_np.sum() > 100:
            brain_vals = img_01[brain_mask_np]
            threshold  = np.percentile(brain_vals, 95)
            anomaly_np = (brain_mask_np & (img_01 > threshold)).astype(np.float32)
        else:
            anomaly_np = np.zeros_like(img_01)
        anomaly_mask = torch.from_numpy(anomaly_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # ── 前向き補間: x_t = (1-t_start)·noise + t_start·tumor ────────────
        x0_noise = torch.randn_like(x1_tumor)
        x_t = (1.0 - t_start) * x0_noise + t_start * x1_tumor

        # 異常領域を純粋ノイズに差し替え（Anomaly-Aware Noising）
        # → ODE積分中に周囲の正常文脈から補完される
        x_t = x_t * (1.0 - anomaly_mask) + x0_noise * anomaly_mask

        # ── Euler ODE: t_start → 1.0 ─────────────────────────────────────────
        # dx/dt = v_θ(x, t)
        # アトラスガイド版では各ステップでx̂_1予測にアトラスをブレンド
        dt = (1.0 - t_start) / num_steps
        x = x_t
        t = t_start

        for _ in tqdm(range(num_steps), desc="Flow Matching ODE", leave=False):
            t_int = torch.tensor([int(t * 999)], device=self.device, dtype=torch.long)
            v_pred = self.unet(x=x, timesteps=t_int)  # 速度場予測

            if atlas_tensor is not None:
                # x̂_1 = x + v·(1-t): 現時点から最終データ到達先を予測
                # （線形フローの場合、残り時間 (1-t) だけ速度を積分するとx1に着く）
                x1_pred = x + v_pred * (1.0 - t)
                # アトラスとブレンド
                x1_guided = (1.0 - atlas_strength) * x1_pred + atlas_strength * atlas_tensor
                # ガイド後のx1から速度場を逆算
                v_pred = (x1_guided - x) / max(1.0 - t, 1e-5)

            x = x + v_pred * dt
            t = t + dt

        # テンソル → PIL (グレースケール → RGB)
        cf_np = ((x.squeeze().cpu().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        cf_rgb = np.stack([cf_np] * 3, axis=2)
        counterfactual_image = Image.fromarray(cf_rgb)

        # 差分マップ
        orig_np = np.array(img_gray).astype(float)
        diff = gaussian_filter(np.abs(orig_np - cf_np.astype(float)), sigma=2.0)
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        difference_map = Image.fromarray(diff.astype(np.uint8))

        return counterfactual_image, difference_map


# ─── インペインティングベースの反実仮想（SDなし・MRI外観を保つ）─────────────────

class InpaintingCounterfactual:
    """
    OpenCV インペインティングによる反実仮想生成。

    SDを使わず、腫瘍領域を周囲の正常組織ピクセルで補完するので
    見た目が本物のMRI画像のまま保たれる。

    アルゴリズム:
      1. 腫瘍領域を自動検出（高輝度閾値 or 外部マスク）
      2. マスクを少し拡張（境界を自然にする）
      3. cv2.inpaint で周囲ピクセルから補完
      4. マスク境界をポアソンブレンディングでなじませる
    """

    def __init__(self, inpaint_radius: int = 15):
        self.inpaint_radius = inpaint_radius

    def detect_tumor_mask(
        self,
        image: Image.Image,
        percentile: float = 92.0,
        dilate_px: int = 12,
    ) -> np.ndarray:
        """
        高輝度領域から腫瘍マスクを自動生成する。

        Args:
            image      : RGB PIL画像
            percentile : この値より明るいピクセルを腫瘍候補とする
            dilate_px  : マスクの拡張ピクセル数（境界をなじませるため）

        Returns:
            uint8 マスク (0=正常, 255=腫瘍候補)
        """
        import cv2
        from scipy.ndimage import binary_dilation

        gray = np.array(image.convert("L")).astype(float)

        # 脳内領域のみで閾値を計算（背景の黒を除外）
        brain_pixels = gray[gray > 10]
        if len(brain_pixels) == 0:
            return np.zeros(gray.shape, dtype=np.uint8)

        thresh = np.percentile(brain_pixels, percentile)
        mask = (gray >= thresh).astype(bool)

        # 小さすぎる孤立領域を除去
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

        # マスクを膨張させて境界を含める
        struct = np.ones((dilate_px, dilate_px), dtype=bool)
        mask_dilated = binary_dilation(mask_u8 > 0, structure=struct)

        return mask_dilated.astype(np.uint8) * 255

    def generate(
        self,
        tumor_image: Image.Image,
        mask: Optional[np.ndarray] = None,
        percentile: float = 92.0,
        dilate_px: int = 12,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        インペインティングで反実仮想を生成する。

        Args:
            tumor_image : 腫瘍を含む脳MRI画像
            mask        : 腫瘍マスク (uint8 numpy配列, 255=腫瘍)。
                          Noneの場合は自動検出。
            percentile  : 自動マスク生成時の閾値パーセンタイル
            dilate_px   : 自動マスクの膨張サイズ

        Returns:
            (counterfactual_image, difference_map)
        """
        import cv2
        from scipy.ndimage import gaussian_filter

        img_arr = np.array(tumor_image.convert("RGB"))

        # マスク生成（外部指定 or 自動）
        if mask is None:
            mask = self.detect_tumor_mask(tumor_image, percentile, dilate_px)
        else:
            mask = (mask > 127).astype(np.uint8) * 255

        # OpenCV インペインティング（Telea法: 境界から伝播して補完）
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(
            img_bgr, mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

        # マスク境界をガウスブレンドでなじませる
        # （マスクのソフトエッジを作りクロスフェード）
        mask_float = mask.astype(float) / 255.0
        blur_sigma = max(dilate_px * 0.6, 4)
        mask_soft = gaussian_filter(mask_float, sigma=blur_sigma)
        mask_soft = np.clip(mask_soft, 0, 1)[:, :, np.newaxis]

        orig_f  = img_arr.astype(float)
        inp_f   = inpainted_rgb.astype(float)
        blended = orig_f * (1 - mask_soft) + inp_f * mask_soft
        result  = np.clip(blended, 0, 255).astype(np.uint8)

        counterfactual_image = Image.fromarray(result)

        # 差分マップ
        diff = gaussian_filter(
            np.abs(orig_f - blended).mean(axis=2), sigma=2.0
        )
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        difference_map = Image.fromarray(diff.astype(np.uint8))

        return counterfactual_image, difference_map


# ─── バッチ処理 ─────────────────────────────────────────────────────────────────

def run_batch(
    input_dir: str,
    output_dir: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    num_images: int = 10,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    image_size: int = 512,
):
    """複数画像のバッチ処理"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    (output_path / "counterfactual").mkdir(parents=True, exist_ok=True)
    (output_path / "difference").mkdir(parents=True, exist_ok=True)
    (output_path / "comparison").mkdir(parents=True, exist_ok=True)

    # 入力画像リスト
    image_files = sorted(input_path.glob("*.png"))[:num_images]
    print(f"処理対象: {len(image_files)} 枚")

    # パイプライン初期化
    cf_pipeline = CounterfactualPipeline(
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_size=image_size,
    )

    for img_path in tqdm(image_files, desc="バッチ処理"):
        original = Image.open(img_path).convert("RGB")

        # 反実仮想生成
        counterfactual, diff_map = cf_pipeline.generate_counterfactual(original)

        stem = img_path.stem

        # 個別保存
        counterfactual.save(output_path / "counterfactual" / f"{stem}_cf.png")
        diff_map.save(output_path / "difference" / f"{stem}_diff.png")

        # 比較画像（横並び: 元画像 | 反実仮想 | 差分マップ）
        comparison = make_comparison_image(
            original.resize((image_size, image_size)),
            counterfactual,
            diff_map,
        )
        comparison.save(output_path / "comparison" / f"{stem}_compare.png")

    print(f"\n✅ 処理完了! 結果: {output_dir}")


def make_comparison_image(
    original: Image.Image,
    counterfactual: Image.Image,
    diff_map: Image.Image,
) -> Image.Image:
    """3枚を横並びにした比較画像を作成"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("black")

    titles = ["元画像（腫瘍MRI）", "反実仮想（正常脳）", "差分マップ（病変領域）"]
    images = [original, counterfactual, diff_map]
    cmaps = [None, None, "hot"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")

    plt.tight_layout(pad=0.5)

    # matplotlibからPIL画像へ変換
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(buf)


# ─── エントリーポイント ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual MRI生成パイプライン")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="腫瘍スライス画像のディレクトリ")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="出力先ディレクトリ")
    parser.add_argument("--model_id", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Faceモデル ID")
    parser.add_argument("--num_images", type=int, default=10,
                        help="処理する画像数")
    parser.add_argument("--steps", type=int, default=50,
                        help="推論ステップ数")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFGスケール")
    parser.add_argument("--image_size", type=int, default=512,
                        help="処理画像サイズ")
    args = parser.parse_args()

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        num_images=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
    )
