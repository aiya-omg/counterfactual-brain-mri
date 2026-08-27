# 🧠 Brain MRI Research

## 現在の主軸: SENORA-MRI 外部検証

サハラ以南アフリカの実地臨床脳MRI（SENORA-MRI）に対して、脳卒中病変セグメンテーションの
外部検証を行う研究に主軸を移しました。設計書は [docs/senora-study-design.md](docs/senora-study-design.md) を参照。

新規手法は提案せず、既製の nnU-Net を用いて以下を明らかにします。

1. 研究グレードデータで学習したモデルの、実地臨床データでの性能低下の定量化
2. 人工劣化対照群による、低下要因の分解（撮像条件 vs 集団・臨床要因）
3. 患者の社会経済状況・教育水準と性能の関連解析

### 進捗（2026-08-27）

- 段階0〜0.6完了。主アームは FLAIR（16例）、学習元は ISLES 2022 FLAIR
- 段階1: nnU-Net fold0 を1000 epoch 完走。ただし**検収を通らなかった**
  （症例ごとの Dice 中位 0.132）
- 段階2: SENORA 16例へ適用したが Dice 中位 0.000、8例は予測が空
- 原因は撮像分解能の不一致。断面外のボクセル間隔が学習元 0.71 mm に対し
  SENORA は 6.8 mm で約10倍
- 対応: ISLES を SENORA の撮像幾何（5 mm 厚 / 6.8 mm 間隔）へ落として Dataset502 を作成。
  病変体積は中位97.4%が残り、fold0 を再学習中（約24時間）
- 詳細は [docs/worklog-2026-08-27.md](docs/worklog-2026-08-27.md) と設計書 8.5
- 主要スクリプトは `senora/scripts/`（取得・正規化・登録・配置・検収・推論）
- nnU-Net の作業ディレクトリは Windows の日本語パス回避のため `%USERPROFILE%\senora_nnunet`

---

## 旧: Counterfactual Visual Attribution for Brain MRI

> 以下は初期の反実仮想生成プロトタイプの記録です。BraTS 上で Dice 中央値 0.104 に留まり、
> 公開SOTA（0.699）との差が大きく、かつ手法的な新規性も確保できないと判断して主軸から外しました。
> コードは教師なし異常検知の資産として SENORA 側で再利用します。

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
