# 旧: Counterfactual Visual Attribution

BraTS 上で「もしこの患者が健康だったら」という反実仮想画像を作り、
差分から病変を可視化するプロトタイプ。Dice 中央値 0.104 に留まり、
公開 SOTA（0.699）との差と手法的な新規性の不足から主軸を外した。

コードは残してある。教師なし異常検知として SENORA 側で再利用する
可能性があるため削除していない。依存は当時の `diffusers` / LoRA / DDPM
一式で、現在の主軸（nnU-Net）とは別物である。

入口は `demo.py` と `counterfactual.py`。手順はリポジトリ初期の README
にあったが、現行の手順は [../README.md](../README.md) の SENORA 側を使う。
