# SENORA-MRI 外部検証（アームC）

研究用 ISLES 2022 FLAIR で学習した nnU-Net を、ナイジェリアの実地臨床
MRI（SENORA）16例へ当て、性能低下を測る。新規手法は提案しない。

## いまの数字（fold 0）

| | Dataset501（等方 0.71 mm） | Dataset502（5 mm / 6.8 mm） |
|---|---|---|
| ソース内 Dice 中位 | 0.132 | 0.288 |
| SENORA 16例 Dice 中位 | 0.000 | 0.000 |

分解能を揃えても SENORA は床。残る制約は DWI 由来ラベルと病期・装置。
詳細は [../docs/worklog-2026-08-27.md](../docs/worklog-2026-08-27.md) と
設計書 8.5。

## 作業場所

日本語パスを SimpleITK / nnU-Net が読めないため、学習・推論の実体は
`%USERPROFILE%\senora_nnunet` に置く。このディレクトリ配下の
`data/` と `results/` は git 対象外。

```
%USERPROFILE%\senora_nnunet\
  nnUNet_raw\Dataset501_ISLES22FLAIR\
  nnUNet_raw\Dataset502_ISLES22FLAIRTHICK\
  nnUNet_preprocessed\
  nnUNet_results\
  armc_senora\            # Dataset501 向けの頭蓋除去済み FLAIR
  armc_senora_d502\       # Dataset502 向けの推論出力
```

## スクリプト

| 段階 | ファイル | すること |
|---|---|---|
| 0 | `fetch_senora.py` | Zenodo から SENORA を取る |
| 0 | `inventory_senora.py` | シーケンスとマスクの集計 |
| 0.5 | `normalize_senora.py` | 重複 run を解消して正規化ツリーを作る |
| 0.6 | `characterize_masks.py` | マスクの体積・信号・幾何を測る |
| 1 | `fetch_isles2022.py` | ISLES 2022 を取る |
| 1 | `register_isles_flair.py` | DWI マスクを FLAIR 空間へ剛体登録 |
| 1 | `prepare_nnunet_armc.py` | nnU-Net Dataset 形式に並べる（501 / 502） |
| 1 | `degrade_isles_resolution.py` | 案E。ISLES を SENORA の幾何へ落とす |
| 1 | `plot_nnunet_dice.py` | 学習曲線（複数ログを束ねる） |
| 1 | `evaluate_source_baseline.py` | ソース内を症例ごとの Dice で検収する |
| 2 | `predict_senora_armc.py` | 頭蓋除去（HD-BET）・推論・評価 |

## 再開

```
set nnUNet_raw=%USERPROFILE%\senora_nnunet\nnUNet_raw
set nnUNet_preprocessed=%USERPROFILE%\senora_nnunet\nnUNet_preprocessed
set nnUNet_results=%USERPROFILE%\senora_nnunet\nnUNet_results
nnUNetv2_train 502 3d_fullres 0 --c
```

再開フラグは `--c`。`-c` ではない。
