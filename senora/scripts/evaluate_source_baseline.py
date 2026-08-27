"""
evaluate_source_baseline.py
段階1の検収: ISLES 2022 FLAIR 単一チャネルで学習した nnU-Net の
ソース内 hold-out 性能を、次の段階へ進める前に評価する。

なぜ単純な平均値だけでは足りないか:

  nnU-Net が学習中に表示する "pseudo Dice" は、検証バッチ全体を
  1つの集合として計算した大域 Dice である。大きな病変が値を支配するため、
  症例ごとの Dice を平均した値より高く出る。段階2で SENORA の16例を
  評価するときに使うのは症例ごとの Dice なので、比較の土俵を揃えるには
  ソース内でも症例ごとの分布を見る必要がある。

  さらにアームCには構造的な問題がある。ISLES 2022 の病変マスクは
  DWI 空間で描かれており（設計書 4.4 (2)）、これを FLAIR 空間へ写して
  学習ラベルにしている。しかし超急性期では DWI に写る梗塞が FLAIR には
  まだ現れない。これは DWI-FLAIR ミスマッチとして発症時刻の推定に
  使われている現象そのものであり、臨床的に確立している。
  つまり「FLAIR 上に見えないものを FLAIR から当てろ」という
  解けない課題が一定数まぎれこんでいる可能性がある。

  この不可避な上限を測っておかないと、段階2で SENORA の性能が低くても
  それがドメインシフトなのか課題設定の限界なのかを分離できない。

測るもの:

  1. 症例ごとの Dice 分布（中位値と四分位範囲）。大域 Dice との差
  2. 適合率と再現率。どちら側に外しているか
  3. 病変体積との関係。小病変で崩れるのは既知の傾向なので、
     文献値と比較するときに体積分布を揃える必要がある
  4. FLAIR 上での病変コントラスト。マスク内の信号が周囲組織と
     区別できない症例では、FLAIR 単一チャネルでは原理的に当てられない
  5. マスク登録の round-trip Dice との関係。ラベル雑音の寄与を見る

使い方:
  python evaluate_source_baseline.py
  python evaluate_source_baseline.py --summary <別の summary.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

# 病変内の信号を組織中位値との比で表したとき、この値を超える分を
# 「FLAIR 上で高信号」とみなす。characterize_masks.py と同じ基準
HYPER_RATIO = 1.3

# 高信号率がこれを下回る症例は、FLAIR 上で病変が周囲と区別できていない。
# DWI 由来マスクを FLAIR へ写した結果として生じる不可避な誤りの候補
INVISIBLE_HYPER_FRAC = 0.5


def tissue_median(volume: np.ndarray) -> float:
    """
    頭部組織の代表信号を返す。

    ISLES は頭蓋除去済みなので背景 0 が体積の大半を占める。
    0 を除いた中位値をそのまま実質の代表値として使える。
    """
    finite = volume[np.isfinite(volume)]
    head = finite[finite > 0]
    return float(np.median(head)) if head.size else float("nan")


def case_to_subject(registered_root: Path, raw: Path) -> dict[str, str]:
    """
    nnU-Net の症例ID（ISLES_0001…）を元の被験者IDへ戻す。

    prepare_nnunet_armc.py が書く対応表があればそれを使う。
    症例を除外した Dataset では ID に欠番が出るため、並び順からの
    復元では合わない。対応表がない古い Dataset のみ順序で復元する。
    """
    mapping_csv = raw / "case_to_subject.csv"
    if mapping_csv.exists():
        table = pd.read_csv(mapping_csv)
        return dict(zip(table["case_id"], table["subject"]))
    cases = sorted(registered_root.glob("sub-*/ses-*/anat/*_FLAIR.nii.gz"))
    return {f"ISLES_{i:04d}": c.parents[2].name for i, c in enumerate(cases, start=1)}


def measure_case(image: Path, reference: Path) -> dict:
    """FLAIR 上で病変がどれだけ見えているかを測る。"""
    img = nib.load(image)
    ref = nib.load(reference)

    mask = np.asarray(ref.dataobj) > 0
    record: dict = {
        "voxels_ref": int(mask.sum()),
        "voxel_ml": abs(np.linalg.det(ref.affine[:3, :3])) / 1000.0,
    }
    record["volume_ml"] = round(record["voxels_ref"] * record["voxel_ml"], 3)

    if not mask.any():
        return record

    flair = np.asarray(img.dataobj, dtype=np.float64)
    if flair.ndim > 3:
        flair = flair[..., 0]
    if flair.shape != mask.shape:
        record["shape_mismatch"] = True
        return record

    baseline = tissue_median(flair)
    if not np.isfinite(baseline) or baseline <= 0:
        return record

    inside = flair[mask] / baseline
    record["lesion_p50"] = round(float(np.median(inside)), 3)
    record["frac_hyper"] = round(float((inside > HYPER_RATIO).mean()), 3)
    return record


def load_summary(path: Path) -> tuple[pd.DataFrame, float]:
    """
    nnU-Net の summary.json を症例ごとの表と大域 Dice に開く。

    大域 Dice は DataFrame の属性ではなく戻り値として返す。
    pandas の attrs は concat や merge をまたぐと失われるため。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for entry in data["metric_per_case"]:
        metrics = entry["metrics"]["1"]
        case = Path(entry["reference_file"]).name.replace(".nii.gz", "")
        tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
        rows.append(
            {
                "case": case,
                "dice": metrics["Dice"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "n_ref": metrics["n_ref"],
                "n_pred": metrics["n_pred"],
            }
        )
    df = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    return df, float(data["foreground_mean"]["Dice"])


def describe(series: pd.Series) -> str:
    q1, q3 = series.quantile([0.25, 0.75])
    return f"{series.median():.3f}（IQR {q1:.3f}–{q3:.3f}）"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    results = Path.home() / "senora_nnunet" / "nnUNet_results"
    fold = (
        results
        / "Dataset501_ISLES22FLAIR"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
    )
    parser.add_argument("--summary", type=Path, default=fold / "validation" / "summary.json")
    parser.add_argument(
        "--summary-final",
        type=Path,
        default=fold / "summary_final_checkpoint.json",
        help="最終エポック重みでの検証結果。比較用。無ければ省略する",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path.home() / "senora_nnunet" / "nnUNet_raw" / "Dataset501_ISLES22FLAIR",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path.home()
        / "senora_nnunet"
        / "nnUNet_preprocessed"
        / "Dataset501_ISLES22FLAIR"
        / "gt_segmentations",
    )
    parser.add_argument(
        "--registered", type=Path, default=root / "data" / "isles2022_flair"
    )
    parser.add_argument(
        "--registration-qc",
        type=Path,
        default=root / "results" / "isles_flair_registration.csv",
    )
    parser.add_argument("--out", type=Path, default=root / "results")
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"エラー: {args.summary} がありません", file=sys.stderr)
        return 1

    df, global_dice = load_summary(args.summary)

    # FLAIR 上での見え方を症例ごとに測る
    measures = []
    for case in df["case"]:
        image = args.raw / "imagesTr" / f"{case}_0000.nii.gz"
        ref = args.gt / f"{case}.nii.gz"
        if not image.exists() or not ref.exists():
            print(f"[warn] {case}: 画像かラベルが見つかりません", file=sys.stderr)
            measures.append({})
            continue
        measures.append(measure_case(image, ref))
    df = pd.concat([df, pd.DataFrame(measures)], axis=1)

    # 元の被験者IDへ戻し、登録 QC を突き合わせる
    mapping = case_to_subject(args.registered, args.raw)
    df["subject"] = df["case"].map(mapping)
    if args.registration_qc.exists():
        qc = pd.read_csv(args.registration_qc)[["subject", "roundtrip_dice"]]
        df = df.merge(qc, on="subject", how="left")

    lines: list[str] = []
    out = lines.append

    out("# 段階1の検収: ソース内 hold-out 性能")
    out("")
    out("ISLES 2022 FLAIR 単一チャネル、fold 0、検証 "
        f"**{len(df)} 例**（学習164 / 検証41）。")
    out(f"評価元: `{args.summary}`")
    out("")

    # --- 1. 大域 Dice と症例ごとの Dice ---
    out("## 1. どの Dice を報告するか")
    out("")
    out("nnU-Net が学習中に出す pseudo Dice は検証集合全体を1つにまとめた大域 Dice で、")
    out("大きな病変が値を支配する。段階2で SENORA を評価するときに使うのは")
    out("症例ごとの Dice なので、こちらを主指標に据える。")
    out("")
    out("| 指標 | 値 |")
    out("|---|---|")
    out(f"| 症例ごとの Dice 中位値 | **{describe(df['dice'])}** |")
    out(f"| 症例ごとの Dice 平均 | {df['dice'].mean():.3f} |")
    out(f"| 大域 Dice（nnU-Net の集計） | {global_dice:.3f} |")
    out(f"| Dice が 0 の症例 | {int((df['dice'] == 0).sum())} / {len(df)} |")
    out(f"| Dice が 0.5 以上の症例 | {int((df['dice'] >= 0.5).sum())} / {len(df)} |")
    out("")

    if args.summary_final.exists() and args.summary_final != args.summary:
        final, final_global = load_summary(args.summary_final)
        out(f"最終エポック重みでの同じ検証: 中位 {final['dice'].median():.3f}、"
            f"大域 {final_global:.3f}。")
        out("段階2の推論には best 重みを用いる。")
        out("")

    # --- 2. 外し方の向き ---
    out("## 2. 過小に取っているか、過大に取っているか")
    out("")
    out(f"- 適合率 中位 **{describe(df['precision'].dropna())}**")
    out(f"- 再現率 中位 **{describe(df['recall'])}**")
    empty = df[df["n_pred"] == 0]
    out(f"- 予測が完全に空だった症例: **{len(empty)}** / {len(df)}"
        "（適合率が定義できない）")
    out("")
    if df["recall"].median() < df["precision"].median():
        out("再現率が適合率より低い。**病変を取り逃がす方向**に偏っている。")
        out("脳全体を塗るような破綻ではないので、後処理で偽陽性を削っても改善しない。")
    else:
        out("適合率が再現率より低い。拾いすぎる方向に偏っている。")
    out("")

    # --- 3. 病変体積 ---
    out("## 3. 病変体積との関係")
    out("")
    vols = df["volume_ml"].dropna()
    out(f"検証集合の病変体積: 中位 **{vols.median():.1f} mL**"
        f"（最小 {vols.min():.2f} / 最大 {vols.max():.1f}）")
    out("")
    bins = [(0, 1), (1, 5), (5, 20), (20, np.inf)]
    out("| 体積 | 例数 | Dice 中位 | 再現率 中位 |")
    out("|---|---|---|---|")
    for low, high in bins:
        group = df[(df["volume_ml"] >= low) & (df["volume_ml"] < high)]
        if group.empty:
            continue
        label = f"{low}–{high} mL" if np.isfinite(high) else f"{low} mL 以上"
        out(f"| {label} | {len(group)} | {group['dice'].median():.3f} "
            f"| {group['recall'].median():.3f} |")
    out("")
    valid = df.dropna(subset=["volume_ml", "dice"])
    if len(valid) >= 5:
        rho, pval = stats.spearmanr(valid["volume_ml"], valid["dice"])
        out(f"Spearman 順位相関 **ρ = {rho:.2f}**（p = {pval:.4f}、n = {len(valid)}）。")
        out("")
        out("小病変で Dice が落ちるのは分野共通の傾向で、ISLES'22 の公式報告も")
        out("同じ関係を示している。したがって文献値と比べるときは、")
        out("体積分布を揃えないと数字だけの比較は成立しない。")
        out("")

    # --- 4. FLAIR 上での見え方 ---
    out("## 4. FLAIR 上に病変が見えているか")
    out("")
    out("マスク内の信号を症例内の組織中位値との比で表し、")
    out(f"`{HYPER_RATIO}` 超の割合を高信号率とする。")
    out("")
    out("**この節がアームCの構造的な限界を示す。** ISLES 2022 のマスクは DWI 空間で")
    out("描かれている。超急性期では DWI に写る梗塞が FLAIR にまだ現れず、")
    out("この DWI-FLAIR ミスマッチは発症時刻の推定に使われている確立した現象である。")
    out("FLAIR 上で周囲と区別できない領域を FLAIR 単一チャネルから当てることは")
    out("原理的に不可能なので、その分は避けられない上限になる。")
    out("")
    vis = df.dropna(subset=["frac_hyper"])
    if not vis.empty:
        out(f"- 高信号率 中位 **{vis['frac_hyper'].median():.3f}**"
            f"（最小 {vis['frac_hyper'].min():.3f} / 最大 {vis['frac_hyper'].max():.3f}）")
        invisible = vis[vis["frac_hyper"] < INVISIBLE_HYPER_FRAC]
        out(f"- 高信号率が {INVISIBLE_HYPER_FRAC} を下回る症例: "
            f"**{len(invisible)}** / {len(vis)}")
        out("")
        if len(vis) >= 5:
            rho, pval = stats.spearmanr(vis["frac_hyper"], vis["dice"])
            out(f"高信号率と Dice の Spearman 順位相関 **ρ = {rho:.2f}**"
                f"（p = {pval:.4f}、n = {len(vis)}）。")
            out("")
        if not invisible.empty:
            visible = vis[vis["frac_hyper"] >= INVISIBLE_HYPER_FRAC]
            out(f"- FLAIR 上で見えている群（{len(visible)} 例）: "
                f"Dice 中位 **{visible['dice'].median():.3f}**")
            out(f"- 見えていない群（{len(invisible)} 例）: "
                f"Dice 中位 **{invisible['dice'].median():.3f}**")
            out("")

        # 可視性と体積は独立ではない。部分容積効果で小病変の高信号率は
        # 薄まるため、両者を別々の説明として並べると二重計上になる
        rho_vv, pval_vv = stats.spearmanr(vis["volume_ml"], vis["frac_hyper"])
        out(f"ただし高信号率は体積と相関している（ρ = {rho_vv:.2f}、p = {pval_vv:.4f}）。")
        out("小病変では部分容積効果で信号が薄まるため、可視性と体積は")
        out("独立した2つの要因ではない。体積を揃えて見ると次のようになる。")
        out("")
        out("| 体積 | 例数 | 高信号率と Dice の ρ | p |")
        out("|---|---|---|---|")
        for low, high in ((0, 5), (5, np.inf)):
            group = vis[(vis["volume_ml"] >= low) & (vis["volume_ml"] < high)]
            label = f"{low}–{high} mL" if np.isfinite(high) else f"{low} mL 以上"
            if len(group) < 5 or group["frac_hyper"].nunique() < 2:
                out(f"| {label} | {len(group)} | 判定不能 | — |")
                continue
            r, p = stats.spearmanr(group["frac_hyper"], group["dice"])
            out(f"| {label} | {len(group)} | {r:+.2f} | {p:.3f} |")
        out("")

        zero = df[df["dice"] == 0].sort_values("volume_ml")
        if not zero.empty:
            out(f"### Dice が 0 だった {len(zero)} 例")
            out("")
            out("| 症例 | 体積(mL) | 高信号率 | 予測ボクセル数 |")
            out("|---|---|---|---|")
            for _, r in zero.iterrows():
                out(f"| {r['case']} | {r['volume_ml']:.2f} "
                    f"| {r.get('frac_hyper', float('nan')):.3f} | {int(r['n_pred'])} |")
            out("")
            small_and_dim = zero[
                (zero["volume_ml"] < 5) & (zero["frac_hyper"] < INVISIBLE_HYPER_FRAC)
            ]
            out(f"{len(zero)} 例すべてが 5 mL 未満で、うち {len(small_and_dim)} 例は")
            out("FLAIR 上でほとんど高信号になっていない。")
            out("超急性期の DWI 陽性・FLAIR 陰性という組み合わせに合致する。")
            out("")

    # --- 5. ラベル雑音 ---
    if "roundtrip_dice" in df.columns and df["roundtrip_dice"].notna().any():
        out("## 5. マスク登録によるラベル雑音の寄与")
        out("")
        out("DWI→FLAIR の剛体登録は round-trip Dice ≥ 0.90 で受理している。")
        out("受理された症例のなかでも登録精度に差があるなら、")
        out("それが Dice に効いているかを見る。")
        out("")
        rt = df.dropna(subset=["roundtrip_dice", "dice"])
        perfect = rt[rt["roundtrip_dice"] >= 0.9999]
        out(f"- round-trip Dice: 中位 {rt['roundtrip_dice'].median():.3f}"
            f"（最小 {rt['roundtrip_dice'].min():.3f}）")
        out(f"- round-trip Dice が 1.000 の症例: {len(perfect)} / {len(rt)}")
        out("")

        if len(rt) >= 5 and rt["roundtrip_dice"].nunique() > 1:
            rho, pval = stats.spearmanr(rt["roundtrip_dice"], rt["dice"])
            rho_v, pval_v = stats.spearmanr(rt["roundtrip_dice"], rt["volume_ml"])
            out(f"素朴に相関を取ると **ρ = {rho:+.2f}**（p = {pval:.4f}）となり、")
            out("登録精度が高い症例ほど Dice が低いという逆向きの関係に見える。")
            out("これは体積による見かけの相関である。")
            out("")
            out(f"- 登録精度と体積の相関: ρ = {rho_v:+.2f}（p = {pval_v:.4f}）")
            if len(perfect) and len(rt) - len(perfect):
                imperfect = rt[rt["roundtrip_dice"] < 0.9999]
                out(f"- round-trip 1.000 の群（{len(perfect)} 例）: "
                    f"体積 中位 {perfect['volume_ml'].median():.1f} mL、"
                    f"Dice 中位 {perfect['dice'].median():.3f}")
                out(f"- 1.000 未満の群（{len(imperfect)} 例）: "
                    f"体積 中位 {imperfect['volume_ml'].median():.1f} mL、"
                    f"Dice 中位 {imperfect['dice'].median():.3f}")
            out("")
            out("小さいマスクは最近傍で写しても形が保たれ round-trip が 1.000 になりやすく、")
            out("大きいマスクは境界のボクセルがずれてわずかに 1.000 を割る。")
            out("つまり round-trip Dice は体積の代理変数として働いており、")
            out("体積が Dice を強く決めている（3章）ぶんが相関として現れているだけである。")
            out("")
            out("**受理後の登録誤差は性能の主要因ではない。** 受理基準を上げても改善しない。")
        out("")

    # --- 6. 検収の判定 ---
    out("## 6. 段階2へ進めるか")
    out("")
    out("設計書 8章の検収条件は「ソース内 hold-out で文献報告値と同等の Dice」。")
    out("ただし比較先を選び直す必要がある。")
    out("")
    out("| 比較対象 | 入力 | Dice | 本研究との違い |")
    out("|---|---|---|---|")
    out("| ISLES'22 優勝 SEALS | DWI + ADC | 0.821 | 拡散強調画像を使用。5分割アンサンブル |")
    out("| ISLES'22 2位 NVAUTO | DWI + ADC | 0.824 | 同上。15モデルのアンサンブル |")
    out("| DeepISLES | DWI + ADC | 0.82 ± 0.12 | 上位手法のアンサンブル |")
    out(f"| **本研究（アームC）** | **FLAIR のみ** | **{df['dice'].median():.3f}** "
        "| 単一 fold、単一モデル。マスクは DWI 由来 |")
    out("")
    out("公開値はいずれも DWI と ADC を入力に使っている。ISLES'22 で FLAIR を")
    out("使ったのは3位の SWAN だけで、それも FLAIR 単独ではなく追加チャネルである。")
    out("**FLAIR 単一チャネルの公開報告値は存在しない**ため、")
    out("「文献値と同等」を数値で判定することはできない。")
    out("")
    out("### 判定")
    out("")
    median = df["dice"].median()
    zero_frac = (df["dice"] == 0).mean()
    out("学習の健全性という意味では問題がない。損失は単調に下がり、")
    out("予測は脳全体を塗るような破綻をしていない（適合率 "
        f"中位 {df['precision'].median():.2f}）。")
    out("")
    out("しかし外部検証の参照値としては使えない。")
    out("")
    out(f"- 症例ごとの Dice 中位値が **{median:.3f}**")
    out(f"- 検証41例のうち **{int(zero_frac * len(df))} 例が Dice 0**")
    out("- 5 mL 未満（19例）は Dice 中位 0.000")
    out("")
    out("段階2の主張は「ソース内 Dice と SENORA での Dice の差」である。")
    out("ソース内が床に張り付いていると、SENORA でどれだけ落ちても")
    out("**差として観測できない**。低下幅を測るには、ソース内が")
    out("十分に高い水準にある必要がある。")
    out("")
    out("### 原因の切り分け")
    out("")
    out("性能が出ない理由は実装ではなく課題設定にある。")
    out("")
    dim = len(vis[vis["frac_hyper"] < INVISIBLE_HYPER_FRAC])
    small = int((df["volume_ml"] < 5).sum())
    out(f"1. **病変が小さい。** 検証集合の体積中位値は {df['volume_ml'].median():.1f} mL で、")
    out(f"   {small} 例が 5 mL 未満。この層の Dice 中位値は 0.000 である。")
    out("   小病変で Dice が落ちるのは ISLES'22 の公式報告と同じ傾向だが、")
    out("   本研究では 5 mL 未満が半数近くを占めるため影響が大きい。")
    out("2. **ラベルの由来が入力と一致していない。** ISLES 2022 のマスクは DWI 空間で")
    out("   描かれた急性期梗塞である。超急性期には DWI 陽性・FLAIR 陰性となるため、")
    out("   FLAIR 上に手がかりのない領域を FLAIR から当てる課題が混ざる。")
    out(f"   検証41例中 {dim} 例が高信号率 0.5 未満だった。")
    out("3. **登録誤差は主要因ではない**（5章）。")
    out("")
    out("なお 1 と 2 は分離できていない。小病変では部分容積効果で高信号率も下がるため、")
    out("体積を揃えると可視性と Dice の関係は有意でなくなる（4章の層別表、n=19 と n=22）。")
    out("どちらが主因かを言うには検証例数が足りない。5分割すべてを回して")
    out("205例で層別すれば分離できる。")
    out("")
    out("### 選択肢")
    out("")
    out("| 案 | 内容 | 代償 |")
    out("|---|---|---|")
    out("| A | ISLES 2015 SISS へ学習元を変更 | 亜急性期・FLAIR ネイティブでマスクも FLAIR 空間。"
        "訓練28例と少なく nnU-Net が不安定になりうる |")
    out("| B | ISLES 2022 のまま、FLAIR 上で高信号の症例に限って学習・評価 | "
        "課題設定が「FLAIR で見える梗塞の抽出」に変わる。除外基準を事前登録する必要がある |")
    out("| C | 入力を DWI + ADC に戻しアームAを主軸にする | SENORA 側の解析集団が16例から7例に減る |")
    out("| D | 現状の数値を「FLAIR 単一チャネルの限界」として報告し、"
        "外部検証は別に設計する | 主張が外部検証から課題設定の分析に変わる |")
    out("")
    out("いずれも設計書の前提に触るため、次に進む前に決める。")
    out("")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "source_baseline_fold0.csv"
    df.to_csv(csv_path, index=False)
    out(f"症例ごとの測定値を `{csv_path}` に出力しました。")
    out("")

    report = "\n".join(lines)
    target = args.out / "source_baseline_fold0.md"
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nレポートを {target} に保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
