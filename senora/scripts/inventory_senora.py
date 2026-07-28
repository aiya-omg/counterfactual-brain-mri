"""
inventory_senora.py
研究設計書 4.3 のチェックリストを実データから自動集計する。

段階0の判断材料を出すのが目的。以下を確認する。

  1. participants.tsv の実際の列構成（記載どおりか）
  2. 症例ごとに実在するシーケンス（8種そろっているか）
  3. 病変マスクの所在と件数
  4. アームA（急性期・DWI）とアームB（慢性期・T1w）に使える症例数
  5. 品質フラグの分布と、FAIL 除外後の残数
  6. 社会経済状況・教育水準の分布（群間比較に足るか）

列名や値の表記はデータセット記述からの推測を含むため、完全一致ではなく
キーワードによる曖昧照合を行い、照合結果を必ず表示する。
想定と違った場合は --show-columns で実際の列を確認し、必要なら本スクリプトの
COLUMN_HINTS / VALUE_HINTS を修正する。

使い方:
  # メタデータのみ取得した段階でも動く（画像スキャンは自動でスキップ）
  python inventory_senora.py

  # 実際の列名を確認したいとき
  python inventory_senora.py --show-columns
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# 設計書の打ち切り基準。各アームがこれを下回ったら計画を見直す
CUTOFF = 10

# participants.tsv の列を概念名にひもづけるためのキーワード
COLUMN_HINTS = {
    "subject": ["participant_id", "subject", "sub_id", "id"],
    "age": ["age"],
    "sex": ["sex", "gender"],
    "bmi": ["bmi"],
    "subtype": ["subtype", "stroke_type", "stroke", "diagnosis"],
    "presentation": ["presentation", "onset", "acuity"],
    "ses": ["socioeconomic", "ses", "economic", "income"],
    "education": ["education", "edu"],
    "qc": ["qc", "quality", "flag"],
    "handedness": ["handedness", "hand"],
    "state": ["state", "origin", "region"],
    "mask": ["has_lesion_mask", "lesion_mask", "has_mask", "mask"],
    "mask_type": ["mask_type"],
}

# マスクありと解釈する値
MASK_TRUE = {"yes", "y", "true", "1"}

# 値の表記ゆれを吸収するためのキーワード
VALUE_HINTS = {
    "ischemic": ["ischemic", "ischaemic", "ischemia", "infarct"],
    "hemorrhagic": ["hemorrhagic", "haemorrhagic", "hemorrhage", "bleed"],
    "acute": ["acute"],
    "chronic": ["chronic"],
    "fail": ["fail"],
    "partial": ["partial"],
    "pass": ["pass"],
}

# シーケンス判定に使う BIDS suffix
SEQUENCES = ["T1w", "T2w", "FLAIR", "dwi", "T2starw", "T1wCE", "T2wCOR", "T2wSAG"]


def find_participants_tsv(root: Path) -> Path | None:
    hits = sorted(root.rglob("participants.tsv"))
    return hits[0] if hits else None


def match_column(columns: list[str], concept: str) -> str | None:
    """キーワード照合で概念に対応する列名を1つ返す。"""
    hints = COLUMN_HINTS.get(concept, [])
    lowered = {c.lower(): c for c in columns}
    # 完全一致を優先
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    # 部分一致
    for hint in hints:
        for low, original in lowered.items():
            if hint in low:
                return original
    return None


def value_is(value: object, concept: str) -> bool:
    """値が概念に該当するかを曖昧判定する。"""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return any(hint in text for hint in VALUE_HINTS.get(concept, []))


def scan_sequences(bids_root: Path) -> dict[str, set[str]]:
    """
    BIDS ツリーを走査し、被験者ごとに実在するシーケンスの集合を返す。
    ファイル名の suffix（_T1w.nii.gz など）で判定する。
    """
    available: dict[str, set[str]] = {}
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        found = set()
        for nii in sub_dir.rglob("*.nii*"):
            stem = nii.name.replace(".nii.gz", "").replace(".nii", "")
            for seq in SEQUENCES:
                if stem.endswith(f"_{seq}") or f"_{seq}." in nii.name:
                    found.add(seq)
        available[sub_dir.name] = found
    return available


def scan_masks(root: Path) -> set[str]:
    """
    病変マスクを持つ被験者IDの集合を返す。
    derivatives 配下、または lesion / mask / seg を含むファイル名を対象にする。
    """
    subjects: set[str] = set()
    keywords = ("lesion", "mask", "roi", "seg")
    for nii in root.rglob("*.nii*"):
        name = nii.name.lower()
        in_derivatives = "derivatives" in {p.lower() for p in nii.parts}
        if not (in_derivatives or any(k in name for k in keywords)):
            continue
        for part in nii.parts:
            if part.startswith("sub-"):
                subjects.add(part)
                break
    return subjects


def distribution(series: pd.Series) -> Counter:
    return Counter(
        "（未記録）" if pd.isna(v) else str(v).strip() for v in series
    )


def render_counter(counter: Counter, indent: str = "    ") -> list[str]:
    total = sum(counter.values())
    lines = []
    for key, count in counter.most_common():
        pct = 100 * count / total if total else 0
        lines.append(f"{indent}{key:<28} {count:>4}  ({pct:4.1f}%)")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="senora/data のパス",
    )
    parser.add_argument(
        "--show-columns",
        action="store_true",
        help="participants.tsv の実際の列名と先頭数行を表示して終了する",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "inventory.md",
        help="レポートの出力先",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"エラー: {args.data} がありません。先に fetch_senora.py を実行してください",
              file=sys.stderr)
        return 1

    tsv = find_participants_tsv(args.data)
    if tsv is None:
        print(f"エラー: participants.tsv が {args.data} 以下に見つかりません",
              file=sys.stderr)
        print("ヒント: python fetch_senora.py --metadata-only", file=sys.stderr)
        return 1

    df = pd.read_csv(tsv, sep="\t")

    if args.show_columns:
        print(f"participants.tsv: {tsv}")
        print(f"\n列 ({len(df.columns)}):")
        for c in df.columns:
            print(f"  {c}")
        print(f"\n先頭5行:\n{df.head().to_string()}")
        return 0

    lines: list[str] = []
    out = lines.append

    out("# SENORA-MRI インベントリ")
    out("")
    out(f"- participants.tsv: `{tsv}`")
    out(f"- 総症例数: **{len(df)}**")
    out("")

    # --- 1. 列の照合結果 ---
    out("## 1. 列の照合")
    out("")
    resolved: dict[str, str | None] = {}
    out("| 概念 | 実際の列 |")
    out("|---|---|")
    for concept in COLUMN_HINTS:
        col = match_column(list(df.columns), concept)
        resolved[concept] = col
        out(f"| {concept} | {col if col else '**見つからず**'} |")
    out("")
    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        out(f"> 照合できなかった概念: {', '.join(missing)}")
        out("> `--show-columns` で実際の列名を確認し、COLUMN_HINTS を修正してください。")
        out("")

    # --- 2. シーケンスとマスク ---
    out("## 2. 画像の実在状況")
    out("")
    bids_roots = [p.parent for p in args.data.rglob("dataset_description.json")]
    bids_root = bids_roots[0] if bids_roots else None
    seq_map: dict[str, set[str]] = {}
    mask_subjects: set[str] = set()

    if bids_root is None or not any(args.data.rglob("sub-*")):
        out("画像データが未取得のため、シーケンス走査をスキップしました。")
        out("画像を含めて確認するには `python fetch_senora.py` を実行してください。")
        out("")
    else:
        seq_map = scan_sequences(bids_root)
        mask_subjects = scan_masks(args.data)
        out(f"- BIDS ルート: `{bids_root}`")
        out(f"- 画像ディレクトリを持つ被験者: **{len(seq_map)}**")
        out(f"- 病変マスクを持つ被験者: **{len(mask_subjects)}**")
        out("")
        seq_counts = Counter()
        for found in seq_map.values():
            seq_counts.update(found)
        out("シーケンス別の保有症例数:")
        out("")
        out("| シーケンス | 症例数 |")
        out("|---|---|")
        for seq in SEQUENCES:
            out(f"| {seq} | {seq_counts.get(seq, 0)} |")
        out("")
        complete = sum(1 for s in seq_map.values() if len(s) >= len(SEQUENCES))
        out(f"8シーケンスすべてを持つ症例: **{complete}** / {len(seq_map)}")
        out("")

    # --- 3. 臨床属性の分布 ---
    out("## 3. 臨床属性の分布")
    out("")
    for concept, label in [
        ("subtype", "脳卒中サブタイプ"),
        ("presentation", "発症状態"),
        ("qc", "品質フラグ"),
        ("mask", "病変マスクの有無"),
        ("mask_type", "マスク種別"),
        ("ses", "社会経済状況"),
        ("education", "教育水準"),
        ("sex", "性別"),
    ]:
        col = resolved.get(concept)
        if col is None:
            continue
        out(f"### {label}（列: `{col}`）")
        out("")
        out("```")
        lines.extend(render_counter(distribution(df[col])))
        out("```")
        out("")

    # --- 4. アーム別の使用可能症例数 ---
    out("## 4. アーム別の使用可能症例数")
    out("")

    subject_col = resolved.get("subject")
    subtype_col = resolved.get("subtype")
    presentation_col = resolved.get("presentation")
    qc_col = resolved.get("qc")
    mask_col = resolved.get("mask")

    if subject_col is None or subtype_col is None:
        out("被験者ID列またはサブタイプ列を特定できず、アーム集計を実行できません。")
        out("`--show-columns` で確認のうえ COLUMN_HINTS を修正してください。")
    else:
        def has_sequence(subject: str, seq: str) -> bool:
            # 画像未取得なら条件から外す（過小評価を避ける）
            if not seq_map:
                return True
            return seq in seq_map.get(str(subject), set())

        def has_mask(subject: str, row: pd.Series) -> bool:
            # participants.tsv に有無が書かれていればそれを優先する。
            # 画像を落とす前でもマスク条件を適用できる
            if mask_col is not None:
                return str(row[mask_col]).strip().lower() in MASK_TRUE
            if not mask_subjects:
                return True
            return str(subject) in mask_subjects

        rows = []
        for _, row in df.iterrows():
            subject = row[subject_col]
            ischemic = value_is(row[subtype_col], "ischemic")
            acute = (
                value_is(row[presentation_col], "acute")
                if presentation_col else False
            )
            chronic = (
                value_is(row[presentation_col], "chronic")
                if presentation_col else False
            )
            qc_fail = value_is(row[qc_col], "fail") if qc_col else False

            rows.append({
                "subject": subject,
                "ischemic": ischemic,
                "acute": acute,
                "chronic": chronic,
                "qc_fail": qc_fail,
                "has_dwi": has_sequence(subject, "dwi"),
                "has_t1w": has_sequence(subject, "T1w"),
                "has_mask": has_mask(subject, row),
            })

        elig = pd.DataFrame(rows)
        arm_a = elig[
            elig.ischemic & elig.acute & elig.has_dwi
            & elig.has_mask & ~elig.qc_fail
        ]
        arm_b = elig[
            elig.ischemic & elig.chronic & elig.has_t1w
            & elig.has_mask & ~elig.qc_fail
        ]

        skipped = []
        if not seq_map:
            skipped.append("シーケンス実在")
        if mask_col is None and not mask_subjects:
            skipped.append("マスク有無")
        if qc_col is None:
            skipped.append("品質フラグ")
        out(f"打ち切り基準: 各アーム **{CUTOFF} 例**")
        if skipped:
            out("")
            out(f"> 未適用の条件: {', '.join(skipped)}。以下は該当数の上限値です。")
        out("")
        out("| アーム | 条件 | 該当数 | 判定 |")
        out("|---|---|---|---|")
        for name, subset, cond in [
            ("A（急性期・DWI）", arm_a, "虚血性 かつ Acute かつ DWI かつ マスク かつ QC≠FAIL"),
            ("B（慢性期・T1w）", arm_b, "虚血性 かつ Chronic かつ T1w かつ マスク かつ QC≠FAIL"),
        ]:
            verdict = "続行可" if len(subset) >= CUTOFF else "**基準未満**"
            out(f"| {name} | {cond} | {len(subset)} | {verdict} |")
        out("")

        out("内訳（段階的に条件を足したときの残数）:")
        out("")
        out("```")
        out(f"    全症例                          {len(elig):>4}")
        out(f"    虚血性                          {int(elig.ischemic.sum()):>4}")
        out(f"    虚血性 かつ QC≠FAIL             {int((elig.ischemic & ~elig.qc_fail).sum()):>4}")
        out(f"      うち Acute                    {int((elig.ischemic & ~elig.qc_fail & elig.acute).sum()):>4}")
        out(f"      うち Chronic                  {int((elig.ischemic & ~elig.qc_fail & elig.chronic).sum()):>4}")
        out(f"      うち マスクあり               {int((elig.ischemic & ~elig.qc_fail & elig.has_mask).sum()):>4}")
        out("```")
        out("")

        if max(len(arm_a), len(arm_b)) < CUTOFF:
            out("> **両アームとも基準未満です。** 設計書 4.3 の打ち切り基準に従い、")
            out("> アームを分けず全症例をまとめる設計に切り替えるか、")
            out("> Crouzon-PUACT への変更を検討してください。")
            out("")

        csv_path = args.out.parent / "eligibility.csv"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        elig.to_csv(csv_path, index=False)
        out(f"症例ごとの判定を `{csv_path}` に出力しました。")
        out("")

    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nレポートを {args.out} に保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
