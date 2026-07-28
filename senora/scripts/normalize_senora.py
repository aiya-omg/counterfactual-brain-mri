"""
normalize_senora.py
段階0.5: SENORA-MRI のツリーを正規化し、シーケンスごとに正しい1本を選ぶ。

背景（段階0で判明した問題）:

  1. ディレクトリが多重ネストしている（sub-001/anat/anat/, sub-002/dwi/dwi/dwi/）
  2. 同一 suffix に `_run-NN` 付きの複数ファイルがあり、これは繰り返し撮像ではなく
     **別系列**である。localizer（スカウト）が混ざっている
  3. run 無しの「正規」ファイルが localizer であることがある。
     例: sub-001_T2star.nii.gz の SeriesDescription は "localizer"、3スライスのみ
  4. dcm2niix が BidsGuess に "discard" と記録した系列がそのまま採用されている

したがって単純なハッシュ重複排除では正しい画像を選べない。
JSON サイドカーの SeriesDescription / ProtocolName / スライス数で系列を判定する。

使い方:
  # まず何が入っているかを調べる。選別規則を決めるための材料
  python normalize_senora.py --audit

  # 正規化を実行（元データは読むだけ。出力は senora/data/senora_normalized）
  python normalize_senora.py
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory_senora import SEQUENCES, sequence_of  # noqa: E402

# これらが SeriesDescription / ProtocolName に含まれる系列は撮像位置決め用、
# または最大値投影（MIP）であり、解析には使えない
SCOUT_KEYWORDS = ("localizer", "scout", "survey", "aahead", "mip", "myelo")

# 解析に使える最低スライス数。localizer は3枚程度、MIP は1枚
MIN_SLICES = 8

# 各 suffix で採るべき撮像面。suffix 名に面が入っているものはそれに従う
TARGET_ORIENTATION = {
    "T1w": "tra",
    "T2w": "tra",
    "FLAIR": "tra",
    "dwi": "tra",
    "T2star": "tra",
    "T1wCE": "tra",
    "T2wCOR": "cor",
    "T2wSAG": "sag",
    "MRA": "",
}


def load_sidecar(nii_path: Path) -> dict:
    """画像に対応する JSON サイドカーを読む。無ければ空辞書。"""
    name = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
    sidecar = nii_path.parent / f"{name}.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def orientation_of(meta: dict, series: str) -> str:
    """撮像面を tra / cor / sag のいずれかで返す。判定できなければ空文字。"""
    text = str(meta.get("ImageOrientationText", "")).strip()
    if text:
        head = re.split(r"[>(]", text)[0].strip().lower()
        for prefix in ("tra", "cor", "sag"):
            if head.startswith(prefix):
                return prefix
    low = series.lower()
    for key, value in (
        ("_tra", "tra"), ("_ax", "tra"), ("_cor", "cor"), ("_sag", "sag"),
    ):
        if key in low:
            return value
    return ""


def output_name(suffix: str, series: str) -> str:
    """
    出力時の suffix を決める。

    dwi には拡散強調（TRACEW）と ADC マップが同じ名前で混在しているため分離する。
    T2* は BIDS 標準の T2starw に直す。
    """
    if suffix == "dwi":
        return "ADC" if "adc" in series.lower() else "dwi"
    return {"T2star": "T2starw"}.get(suffix, suffix)


def describe(nii_path: Path, meta: dict) -> dict:
    """1ファイルの素性をまとめる。"""
    series = str(meta.get("SeriesDescription", "")).strip()
    protocol = str(meta.get("ProtocolName", "")).strip()
    guess = meta.get("BidsGuess", [])
    discard = bool(guess) and str(guess[0]).lower() == "discard"

    try:
        shape = nib.load(nii_path).shape
    except Exception:
        shape = ()
    slices = shape[2] if len(shape) >= 3 else 0

    haystack = f"{series} {protocol}".lower()
    is_scout = any(k in haystack for k in SCOUT_KEYWORDS)

    suffix = sequence_of(nii_path.name)
    orientation = orientation_of(meta, series)
    target = TARGET_ORIENTATION.get(suffix, "")
    # 撮像面が判定できない場合は棄却しない。判定できて食い違う場合のみ落とす
    orientation_ok = (not target) or (not orientation) or orientation == target

    return {
        "path": nii_path,
        "suffix": suffix,
        "out_suffix": output_name(suffix, series),
        "series": series or "（記載なし）",
        "protocol": protocol,
        "discard_flag": discard,
        "shape": shape,
        "slices": slices,
        "tr": meta.get("RepetitionTime"),
        "te": meta.get("EchoTime"),
        "thickness": meta.get("SliceThickness"),
        "series_number": meta.get("SeriesNumber", 0),
        "orientation": orientation or "?",
        "orientation_ok": orientation_ok,
        "is_scout": is_scout,
        "usable": (
            (not is_scout) and slices >= MIN_SLICES and orientation_ok
        ),
        "canonical_name": "_run-" not in nii_path.name,
    }


def collect(bids_root: Path) -> list[dict]:
    """全被験者の画像を走査して素性一覧を作る。"""
    records = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        for nii in sorted(sub_dir.rglob("*.nii*")):
            if "lesion_mask" in nii.name.lower():
                continue
            if sequence_of(nii.name) is None:
                continue
            record = describe(nii, load_sidecar(nii))
            record["subject"] = sub_dir.name
            records.append(record)
    return records


def audit(records: list[dict], out: callable) -> None:
    """シーケンスごとにどんな系列が混ざっているかを報告する。"""
    out("# SENORA-MRI サイドカー監査")
    out("")
    out(f"対象: {len(records)} ファイル / "
        f"{len({r['subject'] for r in records})} 被験者")
    out("")

    out("## 1. suffix ごとの系列の内訳")
    out("")
    out("`SeriesDescription` が実際に撮られたものを示す。")
    out("suffix はデータセット作成者が付けた名前であり、中身と一致するとは限らない。")
    out("")

    by_suffix: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_suffix[r["suffix"]].append(r)

    for suffix in SEQUENCES:
        group = by_suffix.get(suffix, [])
        if not group:
            continue
        out(f"### {suffix}（{len(group)} ファイル）")
        out("")
        target = TARGET_ORIENTATION.get(suffix, "")
        if target:
            out(f"採るべき撮像面: `{target}`")
            out("")
        out("| SeriesDescription | 本数 | 面 | スライス数 | TR (s) | TE (s) | 厚み | 出力名 | 判定 |")
        out("|---|---|---|---|---|---|---|---|---|")
        combos: dict[str, list[dict]] = defaultdict(list)
        for r in group:
            combos[r["series"]].append(r)
        for series, items in sorted(
            combos.items(), key=lambda kv: -len(kv[1])
        ):
            head = items[0]
            slice_counts = sorted({i["slices"] for i in items})
            slice_text = (
                str(slice_counts[0]) if len(slice_counts) == 1
                else f"{slice_counts[0]}〜{slice_counts[-1]}"
            )
            if head["is_scout"]:
                verdict = "**スカウト/MIP**"
            elif head["slices"] < MIN_SLICES:
                verdict = "**スライス不足**"
            elif not head["orientation_ok"]:
                verdict = f"**面が違う（{head['orientation']}）**"
            else:
                verdict = "採用候補"
            discard = " (dcm2niix: discard)" if head["discard_flag"] else ""
            out(f"| {series}{discard} | {len(items)} | {head['orientation']} | "
                f"{slice_text} | {head['tr']} | {head['te']} | "
                f"{head['thickness']} | {head['out_suffix']} | {verdict} |")
        out("")

    out("## 2. run 無しの「正規」ファイルの実態")
    out("")
    out("`sub-XXX_<suffix>.nii.gz` を素直に読むと何が得られるか。")
    out("")
    out("| suffix | 正規ファイル数 | うちスカウト | うち使える |")
    out("|---|---|---|---|")
    trap_total = 0
    for suffix in SEQUENCES:
        canon = [r for r in by_suffix.get(suffix, []) if r["canonical_name"]]
        if not canon:
            continue
        scouts = sum(1 for r in canon if not r["usable"])
        trap_total += scouts
        flag = "**" if scouts else ""
        out(f"| {suffix} | {len(canon)} | {flag}{scouts}{flag} | "
            f"{len(canon) - scouts} |")
    out("")
    if trap_total:
        out(f"> **{trap_total} 本の「正規」ファイルが解析に使えない画像。**")
        out("> ファイル名だけを信じて読み込むと、スカウト画像をモデルに入れることになる。")
        out("")

    out("## 3. 正規化後に各シーケンスが残る症例数")
    out("")
    out("採用候補が1本以上ある症例を、出力名ごとに数える。")
    out("")
    out("| 出力名 | 使える症例数 | 全症例に対する割合 |")
    out("|---|---|---|")
    subjects = {r["subject"] for r in records}
    by_output: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r["usable"]:
            by_output[r["out_suffix"]].add(r["subject"])
    for name, subs in sorted(by_output.items(), key=lambda kv: -len(kv[1])):
        pct = 100 * len(subs) / len(subjects) if subjects else 0
        out(f"| {name} | {len(subs)} | {pct:.1f}% |")
    out("")


def choose(records: list[dict]) -> dict[tuple[str, str], dict]:
    """
    (被験者, 出力名) ごとに採用する1本を決める。

    スカウト・MIP・スライス不足・撮像面違いを除外したうえで、
    スライス数が最も多いものを選ぶ。同数なら SeriesNumber が小さい
    （= その検査で先に撮られた本来の系列）ものを採る。
    """
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        if not r["usable"]:
            continue
        key = (r["subject"], r["out_suffix"])
        current = best.get(key)
        if current is None or (r["slices"], -r["series_number"]) > (
            current["slices"], -current["series_number"]
        ):
            best[key] = r
    return best


def normalize(
    records: list[dict], bids_root: Path, dest: Path, out: callable
) -> None:
    """選ばれた1本だけを、正しい BIDS 配置でコピーする。"""
    import shutil

    # 出力名 -> (置き場所, ファイル名の末尾)
    layout = {
        "dwi": ("dwi", "dwi"),
        "ADC": ("dwi", "desc-ADC_dwi"),
    }

    chosen = choose(records)
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for (subject, out_suffix), record in sorted(chosen.items()):
        folder, stem = layout.get(out_suffix, ("anat", out_suffix))
        target_dir = dest / subject / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        base = f"{subject}_{stem}"

        shutil.copy2(record["path"], target_dir / f"{base}.nii.gz")
        copied += 1

        # サイドカー類も同じ名前で持っていく
        src_stem = record["path"].name.replace(".nii.gz", "").replace(".nii", "")
        for ext in (".json", ".bval", ".bvec"):
            src = record["path"].parent / f"{src_stem}{ext}"
            if src.exists():
                shutil.copy2(src, target_dir / f"{base}{ext}")

    # マスクは derivatives へ
    mask_count = 0
    for mask in sorted(bids_root.rglob("*lesion_mask*.nii*")):
        subject = next(p for p in mask.parts if p.startswith("sub-"))
        variant = mask.name.replace(".nii.gz", "").replace(f"{subject}_lesion_mask", "")
        variant = variant.lstrip("_")
        label = f"_desc-{variant}" if variant else ""
        target_dir = dest / "derivatives" / "manual_lesion" / subject / "anat"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            mask, target_dir / f"{subject}{label}_label-lesion_roi.nii.gz"
        )
        mask_count += 1

    # トップレベルのメタデータは、正規化ツリー単独で解析できるよう持っていく
    meta_copied = []
    for name in (
        "participants.tsv", "dataset_description.json", "README.md",
        "_segmentation_report.csv",
    ):
        src = bids_root / name
        if src.exists():
            shutil.copy2(src, dest / name)
            meta_copied.append(name)

    out("# 正規化の結果")
    out("")
    out(f"- 出力先: `{dest}`")
    out(f"- 画像: **{copied}** 本（{len({s for s, _ in chosen})} 被験者）")
    out(f"- マスク: **{mask_count}** 本を `derivatives/manual_lesion/` へ")
    out(f"- メタデータ: {', '.join(meta_copied) if meta_copied else 'なし'}")
    out("")
    dropped = len(records) - copied
    out(f"- 除外: **{dropped}** 本（スカウト、スライス不足、同一系列の重複）")
    out("")

    per_output = Counter(out_suffix for _, out_suffix in chosen)
    out("| 出力名 | 採用症例数 |")
    out("|---|---|")
    for name, count in per_output.most_common():
        out(f"| {name} | {count} |")
    out("")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "senora")
    parser.add_argument(
        "--dest", type=Path, default=root / "data" / "senora_normalized"
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="選別せず、何が入っているかだけを報告する",
    )
    parser.add_argument("--out", type=Path, default=root / "results")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"エラー: {args.data} がありません。先に fetch_senora.py を実行してください",
              file=sys.stderr)
        return 1

    print("サイドカーを読み込んでいます...")
    records = collect(args.data)
    if not records:
        print("エラー: 画像が見つかりません", file=sys.stderr)
        return 1

    lines: list[str] = []
    out = lines.append

    if args.audit:
        audit(records, out)
        target = args.out / "sidecar_audit.md"
    else:
        normalize(records, args.data, args.dest, out)
        target = args.out / "normalization.md"

    report = "\n".join(lines)
    args.out.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nレポートを {target} に保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
