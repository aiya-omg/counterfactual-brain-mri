# SENORA-MRI 外部検証

サハラ以南アフリカの実地臨床脳MRI（SENORA-MRI）に対して、研究グレード
データで学習した脳卒中病変セグメンテーションがどれだけ性能を落とすかを
測る。新規手法は提案せず、既製の nnU-Net を使う。

設計書: [docs/senora-study-design.md](docs/senora-study-design.md)
作業ログ: [docs/worklog-2026-08-27.md](docs/worklog-2026-08-27.md)（最新） /
[docs/worklog-2026-07-28.md](docs/worklog-2026-07-28.md)
スクリプト: [senora/README.md](senora/README.md)

## 進捗（2026-08-28）

主アームは FLAIR（SENORA 16例）。学習元は ISLES 2022 FLAIR。

| | Dataset501（等方 0.71 mm） | Dataset502（SENORA と同じ 5 mm / 6.8 mm） |
|---|---|---|
| 学習 | 1000 epoch 完了 | 1000 epoch 完了 |
| ソース内 Dice 中位 | 0.132 | **0.288** |
| SENORA 16例 Dice 中位 | 0.000 | **0.000** |

分解能を揃えた効果はソース内にだけ出た。SENORA 側の断面外間隔は
学習時と一致しているので、空予測の原因はもう補間ではない。
次は学習元または課題設定の見直し（設計書 8.5.6 の案 A〜D）。

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
