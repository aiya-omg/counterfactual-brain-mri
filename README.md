# 🧠 Counterfactual Visual Attribution for Brain MRI

Stable Diffusion + DDIM Inversion を使って「もしこの患者が健康だったら」という反実仮想画像を生成し、
実画像との差分から脳腫瘍領域を可視化する研究プロトタイプです。

## 環境要件

- GPU: RTX 4070 以上（VRAM 12GB）
- Python 3.10+
- CUDA 12.x

## セットアップ

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# xformers（VRAM節約に必須）
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

## クイックスタート（合成データで動作確認）

```bash
# BraTSデータ不要でまず動かす
python demo.py --use_synthetic --output_dir ./demo_output
```

## 本番実行フロー

### Step 1: BraTSデータの取得
https://www.synapse.org/#!Synapse:syn27046444/wiki/ からBraTS2021データをダウンロード

### Step 2: 前処理（NIfTI → PNG スライス）

```bash
python preprocess.py \
  --data_dir /path/to/BraTS2021_Training_Data \
  --output_dir ./data/slices \
  --modality t2 \
  --max_patients 20        # まず少数でテスト
```

### Step 3: Counterfactual生成

```bash
python counterfactual.py \
  --input_dir ./data/slices/tumor \
  --output_dir ./results \
  --num_images 10 \
  --steps 50 \
  --guidance_scale 7.5
```

### Step 4: 可視化・評価

```bash
python visualize.py \
  --original_dir ./data/slices/tumor \
  --cf_dir ./results/counterfactual \
  --diff_dir ./results/difference \
  --mask_dir ./data/slices/masks \    # GTマスクで定量評価
  --output_dir ./results/report
```

### Step 5: デモ（実SDモデルで単枚確認）

```bash
python demo.py \
  --input_image ./data/slices/tumor/BraTS2021_00000_slice060.png \
  --gt_mask ./data/slices/masks/BraTS2021_00000_slice060.png \
  --use_real_sd \
  --output_dir ./demo_output
```

## ファイル構成

```
counterfactual_mri/
├── requirements.txt      # 依存ライブラリ
├── preprocess.py         # BraTS NIfTI → PNG変換
├── counterfactual.py     # メインパイプライン（DDIM Inversion + SD1.5）
├── visualize.py          # 差分マップ可視化・評価
├── demo.py               # クイックデモ（合成/実データ両対応）
└── README.md
```

## アルゴリズム

```
腫瘍MRI画像
    ↓ VAE encode
潜在変数 z_0
    ↓ DDIM Inversion（逆拡散）
ノイズ z_T
    ↓ DDIM Denoise（"healthy brain MRI"プロンプトで条件付け）
反実仮想画像（正常脳の推定）
    ↓ |元画像 - 反実仮想|
差分マップ → 病変領域の可視化
```

## RTX 4070 での VRAM使用量目安

| 処理 | VRAM |
|------|------|
| SD 1.5 fp16 + xformers (推論) | ~5 GB |
| SD 1.5 fp16 + xformers (DDIM Inv.) | ~7 GB |
| 512×512 処理 | ~8 GB |

## 評価指標

GTマスクがある場合、以下の指標を自動計算します：
- **Dice係数**: 病変領域の重複度（1.0が最高）
- **IoU**: 予測とGTの交差割合
- **Precision / Recall**: 精度と再現率

## 参考論文

- Singla et al. (2023) "Explaining the Black-box Smoothly" - Nature Machine Intelligence
- Latent Drifting (CVPR 2025)
- DDIM: Song et al. (2020) "Denoising Diffusion Implicit Models"
