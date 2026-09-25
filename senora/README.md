# SENORA-MRI 外部検証

研究用 ISLES 2022 で学習した nnU-Net を、ナイジェリアの実地臨床 MRI（SENORA）へ当て、
性能低下とその内訳を測る。新規手法は提案しない。

## いまの数字

| | Dataset501 | Dataset502 | Dataset503 |
|---|---|---|---|
| 入力 | FLAIR（等方 0.71 mm） | FLAIR（5 mm / 6.8 mm） | DWI + ADC（5.5 mm / 7.15 mm） |
| 対象 | アームC 16例 | アームC 16例 | アームA 7例 |
| ソース内 Dice 中位 | 0.132 | 0.288 | 0.827（fold 0、最終重み） |
| SENORA Dice 中位 | 0.000 | 0.000 | 0.153（fold 0、最終重み） |
| 急性コアの再現率 中位 | — | — | 0.978（コアあり4例） |

アームCは全例が慢性期か不明で、病期が学習元と合わない。アームAの低下の大半は
参照マスクの定義（読影医は新旧を問わず描く）と病期のずれ。

23本のマスクを DWI の ADC で測ると、急性コアを含むのは4本だけ（すべてアームA）。
アームCの検証済み15例に急性コアはなく、9例は空洞化した陳旧性梗塞（設計書 8.6.7）。

詳細は設計書 8.6 と [../docs/worklog-2026-09-25.md](../docs/worklog-2026-09-25.md)。

## 作業場所

日本語パスを SimpleITK / nnU-Net が読めないため、学習・推論の実体は
`%USERPROFILE%\senora_nnunet` に置く。このディレクトリ配下の
`data/` と `results/` は git 対象外。

```
%USERPROFILE%\senora_nnunet\
  nnUNet_raw\Dataset501_ISLES22FLAIR\
  nnUNet_raw\Dataset502_ISLES22FLAIRTHICK\
  nnUNet_raw\Dataset503_ISLES22DWIThick\
  nnUNet_preprocessed\
  nnUNet_results\
  armc_senora\            # Dataset501 向けの頭蓋除去済み FLAIR
  armc_senora_d502\       # Dataset502 向けの推論出力
  arma_senora\            # アームA: b0 / b1000 / ADC、脳マスク、predictions_<重み>_f<fold>
  senora_all\             # 23例: 同じ構成。アームCのマスクは DWI 格子へ写したもの
```

学習中に HD-BET の CLI を並行して走らせないこと。子プロセスを12個起こして
コミットメモリを使い切り、学習ごと落ちる（2026-09-26 に fold 1 が落ちた）。
`stage_senora_lesions.py` は同じプロセス内で1例ずつ推論する。

## スクリプト

| 段階 | ファイル | すること |
|---|---|---|
| 0 | `fetch_senora.py` | Zenodo から SENORA を取る |
| 0 | `inventory_senora.py` | シーケンスとマスクの集計 |
| 0.5 | `normalize_senora.py` | 重複 run を解消して正規化ツリーを作る |
| 0.6 | `characterize_masks.py` | マスクの体積・信号・幾何を測る（信号は FLAIR のみ。DWI は `stage_senora_lesions.py`） |
| 1 | `fetch_isles2022.py` | ISLES 2022 を取る |
| 1 | `register_isles_flair.py` | DWI マスクを FLAIR 空間へ剛体登録（round-trip Dice は登録の正しさを測らない。設計書 8.6.5） |
| 1 | `prepare_nnunet_armc.py` | FLAIR を nnU-Net Dataset 形式に並べる（501 / 502） |
| 1 | `degrade_isles_resolution.py` | 案E。ISLES FLAIR を SENORA の幾何へ落とす |
| 1 | `prepare_nnunet_arma.py` | ISLES DWI + ADC を SENORA の DWI 幾何へ落として並べる（503）。`--qc-only` で QC だけ |
| 1 | `plot_nnunet_dice.py` | 学習曲線（複数ログを束ねる） |
| 1 | `evaluate_source_baseline.py` | FLAIR 版のソース内を症例ごとの Dice で検収する |
| 1 | `summarize_source_cv.py` | 503 のソース内を学習済みの全 fold から集める（最終重みが主） |
| 2 | `predict_senora_armc.py` | アームC: 頭蓋除去（HD-BET）・推論・評価 |
| 2 | `predict_senora_arma.py` | アームA: b1000 / ADC の staging、b0 での頭蓋除去、推論・評価 |
| 2 | `analyze_arma_diffusion.py` | 病期別の評価（急性コアとそれ以外、病変 F1、断面図）。`--work` で23例にも使う |
| 2 | `make_review_sheets.py` | 臨床医判定の資料。予測を載せず、アームA 7例を乱数順に並べる（`results/review/`） |
| 2 | `stage_senora_lesions.py` | 23例を DWI 格子に揃える（FLAIR → DWI の剛体登録と独立 QC）、画像上の病期を測る |

## アームAの再現

```
set nnUNet_raw=%USERPROFILE%\senora_nnunet\nnUNet_raw
set nnUNet_preprocessed=%USERPROFILE%\senora_nnunet\nnUNet_preprocessed
set nnUNet_results=%USERPROFILE%\senora_nnunet\nnUNet_results
python senora\scripts\prepare_nnunet_arma.py
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity -c 3d_fullres
nnUNetv2_train 503 3d_fullres 0        # 1〜4 も同様
python senora\scripts\summarize_source_cv.py
python senora\scripts\predict_senora_arma.py --all --folds 0,1,2,3,4
python senora\scripts\analyze_arma_diffusion.py --tag final_f01234

rem 23例（先に predict_senora_armc.py --stage と predict_senora_arma.py --stage）
python senora\scripts\stage_senora_lesions.py
python senora\scripts\predict_senora_arma.py --work %USERPROFILE%\senora_nnunet\senora_all --predict --evaluate --folds 0,1,2,3,4 --out senora\results\staging\dwi_model
python senora\scripts\analyze_arma_diffusion.py --work %USERPROFILE%\senora_nnunet\senora_all --tag final_f01234 --out senora\results\staging\dwi_model
```

中断した学習の再開フラグは `--c`。`-c` ではない。
