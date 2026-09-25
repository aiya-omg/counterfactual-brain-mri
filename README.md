# SENORA-MRI 外部検証

サハラ以南アフリカの実地臨床脳MRI（SENORA-MRI）に対して、研究グレード
データで学習した脳卒中病変セグメンテーションがどれだけ性能を落とすかを
測る。新規手法は提案せず、既製の nnU-Net を使う。

設計書: [docs/senora-study-design.md](docs/senora-study-design.md)
作業ログ: [docs/worklog-2026-09-25.md](docs/worklog-2026-09-25.md)（最新） /
[docs/worklog-2026-08-27.md](docs/worklog-2026-08-27.md) /
[docs/worklog-2026-07-28.md](docs/worklog-2026-07-28.md)
スクリプト: [senora/README.md](senora/README.md)

## 進捗（2026-09-26）

主アームを FLAIR（アームC、16例）から DWI + ADC（アームA、7例）へ移した。
アームCは全例が慢性期か不明で、急性期の ISLES で学習したモデルとは病期が合わない。

| | Dataset501 | Dataset502 | Dataset503 |
|---|---|---|---|
| 入力 | FLAIR（等方 0.71 mm） | FLAIR（5 mm / 6.8 mm） | **DWI + ADC（5.5 mm / 7.15 mm）** |
| ソース内 Dice 中位（fold 0、最終重み） | 0.132 | 0.288 | **0.827** |
| SENORA Dice 中位 | 0.000（アームC 16例） | 0.000（同） | **0.153**（アームA 7例） |

アームAの低下は、大半が参照マスクの定義と病期のずれで説明できる。
読影医は新旧を問わず梗塞を描いているが、ISLES のラベルは急性の拡散制限域である。
マスク内の急性コア（ADC 620 未満）に限ると、再現率は中位 0.978（コアあり4例）。

アームCのマスクも DWI へ写して ADC で測ると、SENORA のマスク22本（位置合わせを検証できた分）の
うち急性コアを含むのは4本だけだった。大半は陳旧性梗塞である（設計書 8.6.7）。
fold 1〜4 を学習中で、主要な値は 5 fold アンサンブルで確定させる。DWI モデルを23例すべてに
当てたときの予測は、結果を見る前に設計書 8.6.8 に固定した。

## リポジトリ構成

```
docs/                      設計書と作業ログ
senora/scripts/            現行のパイプライン
senora/data/               生データ（gitignore）
senora/results/            測定値と図（gitignore。数字は作業ログへ転記）
legacy/                    旧・反実仮想生成プロトタイプ（主軸から外した）
requirements.txt           現行（nnU-Net / HD-BET）の依存
```

学習の実体は Windows の日本語パスを避けるため
`%USERPROFILE%\senora_nnunet` に置く。

## 環境

- GPU: RTX 4070 12GB
- Python 3.10+（conda 環境 `cfmri`）
- CUDA 12.x

```bash
pip install -r requirements.txt
# hd-bet は argparse バックポートが標準ライブラリを隠すため
pip install hd-bet --no-deps
```

## 旧プロトタイプ

[legacy/](legacy/) に残してある。BraTS 上の反実仮想生成で Dice 中央値 0.104。
公開 SOTA（0.699）との差と新規性の不足から主軸を外した。
