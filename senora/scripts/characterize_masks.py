"""
characterize_masks.py
段階0.6: SENORA-MRI の病変マスク23本の実態を測り、学習元データの要件を確定する。

なぜこれが必要か:

  アームC（FLAIR上のマスク16例）の学習元を選ぶには、評価対象の病変が
  どう見えているかを知る必要がある。とくに慢性期の梗塞は空洞化し、
  FLAIR では中心が CSF 様の低信号、辺縁のグリオーシスが高信号になる。
  読影医が空洞まで含めて描いたのか、高信号の辺縁だけを描いたのかで、
  学習元に求めるものが変わる。

  さらに前提として、読影医がマスクを描いた系列と normalize_senora.py が
  選んだ系列が同一である保証がない。同一 suffix に複数系列がある
  データセットなので（4.3.3）、ここが食い違えばマスクは画像と揃わない。
  幾何（shape と affine）の一致を先に確かめる。

測るもの:

  1. マスクと参照画像の幾何一致（これが崩れていたら他の数字は無意味）
  2. 撮像幾何。スライス厚とボクセル間隔の差、すなわちスライスギャップ
  3. 病変体積、連結成分数、広がるスライス数
  4. マスク内の信号強度分布。組織中位値に対する比で表す。
     低信号成分の割合が空洞を含めたかどうかの指標になる

使い方:
  python characterize_masks.py
  python characterize_masks.py --raw senora/data/senora   # 不一致時の追跡に使う
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats

# 参照シーケンス名（_segmentation_report.csv の表記）から
# 正規化ツリー上のファイル名 suffix への対応
REFERENCE_SUFFIX = {"FLAIR": "FLAIR", "DWI": "dwi"}

# 組織中位値に対する比。この境界で低信号／等信号／高信号を分ける。
# 慢性期梗塞の空洞は CSF 様まで落ちるため 0.8 を下回る
HYPO_RATIO = 0.8
HYPER_RATIO = 1.3

# affine の一致判定。再標本化なしの同一系列なら誤差はこの程度に収まる
AFFINE_TOL = 1e-3


def load_mask(deriv_root: Path, subject: str) -> Path | None:
    """
    被験者のマスクを1本返す。読影医2名の症例は合意マスクを優先する。
    """
    anat = deriv_root / "manual_lesion" / subject / "anat"
    if not anat.is_dir():
        return None
    files = sorted(anat.glob("*_label-lesion_roi.nii.gz"))
    if not files:
        return None
    for f in files:
        if "desc-consensus" in f.name:
            return f
    # 単独読影の症例は desc なしの1本のみ
    plain = [f for f in files if "desc-" not in f.name]
    return plain[0] if plain else files[0]


def geometry(img: nib.Nifti1Image) -> tuple:
    return img.shape[:3], img.affine


def geometry_matches(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    shape_a, affine_a = geometry(a)
    shape_b, affine_b = geometry(b)
    return shape_a == shape_b and np.allclose(affine_a, affine_b, atol=AFFINE_TOL)


def voxel_volume_ml(affine: np.ndarray) -> float:
    """1ボクセルの体積を mL で返す。"""
    return abs(np.linalg.det(affine[:3, :3])) / 1000.0


def tissue_median(volume: np.ndarray) -> float:
    """
    頭部組織の代表的な信号強度を返す。

    SENORA は頭蓋除去されていないため頭皮・頭蓋が含まれる。ただし脳実質が
    体積の大半を占めるので、背景を落としたうえでの中位値は実質の代表値として
    使える。厳密な NAWM 推定ではなく、症例内で信号を比に直すための基準。
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return float("nan")
    # 背景（空気）は 0 付近に大きな峰を作る。全体の中位値で切って頭部を残す
    background_cut = np.percentile(finite, 50)
    head = finite[finite > background_cut]
    return float(np.median(head)) if head.size else float("nan")


def find_matching_raw(raw_root: Path, subject: str, mask: nib.Nifti1Image) -> list[dict]:
    """
    幾何が不一致だった場合、生ツリーのどのファイルがマスクと揃うかを探す。
    見つかった系列の SeriesDescription を返し、読影医が使った系列を特定する。
    """
    sub_dir = raw_root / subject
    if not sub_dir.is_dir():
        return []
    hits = []
    for nii in sorted(sub_dir.rglob("*.nii*")):
        if "lesion_mask" in nii.name.lower():
            continue
        try:
            img = nib.load(nii)
        except Exception:
            continue
        if not geometry_matches(mask, img):
            continue
        sidecar = nii.parent / (
            nii.name.replace(".nii.gz", "").replace(".nii", "") + ".json"
        )
        series = ""
        if sidecar.exists():
            try:
                series = str(
                    json.loads(sidecar.read_text(encoding="utf-8"))
                    .get("SeriesDescription", "")
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        hits.append({"file": nii.name, "series": series})
    return hits


def measure(mask_path: Path, ref_path: Path) -> dict:
    """マスク1本の幾何・形態・信号を測る。"""
    mask_img = nib.load(mask_path)
    ref_img = nib.load(ref_path)

    record: dict = {
        "mask_file": mask_path.name,
        "ref_file": ref_path.name,
        "mask_shape": "x".join(str(v) for v in mask_img.shape[:3]),
        "ref_shape": "x".join(str(v) for v in ref_img.shape[:3]),
        "aligned": geometry_matches(mask_img, ref_img),
    }

    # 撮像幾何。SliceThickness と SpacingBetweenSlices が違えばギャップがある。
    # ギャップのある積層は連続した3次元ボリュームではないため、体積計算と
    # 人工劣化の作り方の両方に影響する
    sidecar = ref_path.parent / (
        ref_path.name.replace(".nii.gz", "").replace(".nii", "") + ".json"
    )
    meta: dict = {}
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            meta = {}
    thickness = meta.get("SliceThickness")
    spacing = meta.get("SpacingBetweenSlices")
    record["protocol"] = meta.get("SeriesDescription", "")
    record["slice_thickness"] = thickness
    record["slice_spacing"] = spacing
    if isinstance(thickness, (int, float)) and isinstance(spacing, (int, float)):
        record["slice_gap"] = round(spacing - thickness, 3)
        record["imaged_fraction"] = round(thickness / spacing, 3)

    mask = np.asarray(mask_img.dataobj)
    # 4D の DWI 上に描かれたマスクは4次元で保存されている。
    # 全ボリュームで同一のはずだが、念のため和を取ってから2値化する
    record["mask_ndim"] = int(mask.ndim)
    if mask.ndim > 3:
        flat = mask.reshape(mask.shape[:3] + (-1,))
        per_volume = [int((flat[..., v] > 0).sum()) for v in range(flat.shape[3])]
        record["mask_volumes"] = len(per_volume)
        record["voxels_per_volume"] = ";".join(str(v) for v in per_volume)
        record["only_first_volume"] = bool(
            per_volume[0] > 0 and all(v == 0 for v in per_volume[1:])
        )
        mask = flat.max(axis=3)

    binary = mask > 0
    n_voxels = int(binary.sum())
    record["voxels"] = n_voxels
    # 慣例どおりボクセル間隔から求める。ギャップがある場合はこの値が
    # 実際に撮像された組織の体積を上回る
    record["volume_ml"] = round(n_voxels * voxel_volume_ml(mask_img.affine), 3)
    if record.get("imaged_fraction"):
        record["volume_imaged_ml"] = round(
            record["volume_ml"] * record["imaged_fraction"], 3
        )

    zooms = mask_img.header.get_zooms()[:3]
    record["voxel_mm"] = "x".join(f"{z:.2f}" for z in zooms)

    if n_voxels == 0:
        record["components"] = 0
        record["slices"] = 0
        return record

    labelled, n_comp = ndimage.label(binary)
    sizes = ndimage.sum(binary, labelled, range(1, n_comp + 1))
    record["components"] = int(n_comp)
    record["largest_frac"] = round(float(sizes.max()) / n_voxels, 3)
    record["slices"] = int(np.count_nonzero(binary.any(axis=(0, 1))))

    if not record["aligned"]:
        # 揃っていない画像から信号を読んでも意味がない
        return record

    ref = np.asarray(ref_img.dataobj, dtype=np.float64)
    if ref.ndim > 3:
        ref = ref[..., 0]
    baseline = tissue_median(ref)
    record["tissue_median"] = round(baseline, 1)
    if not np.isfinite(baseline) or baseline <= 0:
        return record

    inside = ref[binary] / baseline
    record["lesion_p10"] = round(float(np.percentile(inside, 10)), 3)
    record["lesion_p50"] = round(float(np.percentile(inside, 50)), 3)
    record["lesion_p90"] = round(float(np.percentile(inside, 90)), 3)
    record["frac_hypo"] = round(float((inside < HYPO_RATIO).mean()), 3)
    record["frac_hyper"] = round(float((inside > HYPER_RATIO).mean()), 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "senora_normalized")
    parser.add_argument(
        "--raw", type=Path, default=root / "data" / "senora",
        help="幾何が不一致だった場合に、読影医が使った系列を探す生ツリー",
    )
    parser.add_argument("--out", type=Path, default=root / "results")
    args = parser.parse_args()

    report_csv = args.data / "_segmentation_report.csv"
    if not report_csv.exists():
        print(f"エラー: {report_csv} がありません", file=sys.stderr)
        return 1
    seg = pd.read_csv(report_csv)

    participants = args.data / "participants.tsv"
    presentation: dict[str, str] = {}
    if participants.exists():
        pt = pd.read_csv(participants, sep="\t")
        presentation = {
            str(r["participant_id"]): str(r.get("presentation_status", "")).strip().lower()
            for _, r in pt.iterrows()
        }

    records = []
    for _, row in seg.iterrows():
        subject = str(row["subject_id"])
        reference = str(row["sequence"]).strip().upper()
        suffix = REFERENCE_SUFFIX.get(reference)

        mask_path = load_mask(args.data / "derivatives", subject)
        if mask_path is None:
            print(f"[warn] {subject}: マスクが見つかりません", file=sys.stderr)
            continue

        candidates = sorted(
            (args.data / subject).rglob(f"*_{suffix}.nii.gz")
        ) if suffix else []
        # desc-ADC は dwi suffix を持つが別物なので除く
        candidates = [c for c in candidates if "desc-" not in c.name]
        if not candidates:
            print(f"[warn] {subject}: 参照画像 {suffix} が見つかりません", file=sys.stderr)
            continue

        record = measure(mask_path, candidates[0])
        record["subject"] = subject
        record["reference"] = reference
        record["stage"] = presentation.get(subject, "")
        record["raters"] = (
            "2名" if str(row.get("radiologist_a")).strip() == "yes"
            and str(row.get("radiologist_b")).strip() == "yes" else "1名"
        )
        records.append(record)

    if not records:
        print("エラー: 測定できたマスクがありません", file=sys.stderr)
        return 1

    df = pd.DataFrame(records)

    lines: list[str] = []
    out = lines.append

    out("# 段階0.6: 病変マスクの実態")
    out("")
    out(f"対象: {len(df)} 例（`_segmentation_report.csv` の全マスク）")
    out(f"参照画像: `{args.data}` の正規化後シーケンス")
    out("")

    # --- 1. 幾何の一致 ---
    out("## 1. マスクと参照画像の幾何が一致するか")
    out("")
    out("読影医が描いた系列と `normalize_senora.py` が選んだ系列が同一かを見る。")
    out("ここが崩れていると、マスクを画像に重ねられず以降の数字も無意味になる。")
    out("")
    aligned = df[df.aligned]
    misaligned = df[~df.aligned]
    out(f"- 一致: **{len(aligned)}** / {len(df)}")
    out(f"- 不一致: **{len(misaligned)}**")
    out("")

    if len(misaligned):
        out("### 不一致の内訳")
        out("")
        out("| 症例 | 参照 | マスクの形 | 選んだ画像の形 |")
        out("|---|---|---|---|")
        for _, r in misaligned.iterrows():
            out(f"| {r['subject']} | {r['reference']} | {r['mask_shape']} | {r['ref_shape']} |")
        out("")
        if args.raw.exists():
            out("生ツリーの中でマスクと幾何が一致するファイルを探した結果:")
            out("")
            out("| 症例 | 一致したファイル | SeriesDescription |")
            out("|---|---|---|")
            for _, r in misaligned.iterrows():
                mask_img = nib.load(
                    load_mask(args.data / "derivatives", r["subject"])
                )
                hits = find_matching_raw(args.raw, r["subject"], mask_img)
                if not hits:
                    out(f"| {r['subject']} | **一致するものなし** | — |")
                for hit in hits:
                    out(f"| {r['subject']} | {hit['file']} | {hit['series'] or '（記載なし）'} |")
            out("")

    if aligned.empty:
        out("> 一致した症例がないため、以降の測定は行いません。")
        report = "\n".join(lines)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "mask_characterization.md").write_text(report, encoding="utf-8")
        print(report)
        return 1

    four_d = df[df["mask_ndim"] > 3]
    if len(four_d):
        out("### 4次元で保存されたマスク")
        out("")
        out(f"{len(four_d)} 例のマスクが4次元で保存されている。")
        out("参照が4次元の DWI（trace 3本）であるため、マスクも同じ次元を持つ。")
        out("素直に3次元として読むとエラーになる。")
        out("")
        out("| 症例 | 形 | ボリューム数 | ボリューム別のボクセル数 | 1本目のみ有効 |")
        out("|---|---|---|---|---|")
        for _, r in four_d.iterrows():
            out(f"| {r['subject']} | {r['mask_shape']} | {int(r['mask_volumes'])} "
                f"| {r['voxels_per_volume']} | {'はい' if r['only_first_volume'] else '**いいえ**'} |")
        out("")
        if bool(four_d["only_first_volume"].all()):
            out("いずれも1本目のボリュームにのみ内容があり、残りは空。")
            out("全ボリュームの最大値を取れば正しい3次元マスクが得られる。")
        else:
            out("> **1本目以外にも内容があるマスクがある。** 単純な次元削減では扱えない。")
        out("")

    # --- 1.5 撮像幾何 ---
    out("## 2. 参照画像の撮像幾何")
    out("")
    out("スライス厚とボクセル間隔が違う場合、その差はスライスギャップである。")
    out("ギャップのある積層は連続した3次元ボリュームではない。")
    out("撮像されていない組織があるため、体積計算と人工劣化の作り方の両方に効く。")
    out("")
    out("| 参照 | プロトコル | 厚み(mm) | 間隔(mm) | ギャップ(mm) | 撮像率 | 例数 |")
    out("|---|---|---|---|---|---|---|")
    geometry_combos = Counter(
        (
            r["reference"], r.get("protocol", ""), r.get("slice_thickness"),
            r.get("slice_spacing"), r.get("slice_gap"), r.get("imaged_fraction"),
        )
        for _, r in df.iterrows()
    )
    for (ref, proto, thick, space, gap, frac), count in sorted(
        geometry_combos.items(), key=lambda kv: (kv[0][0], -kv[1])
    ):
        out(f"| {ref} | {proto or '—'} | {thick} | {space} | "
            f"{gap if gap is not None else '—'} | "
            f"{f'{frac:.0%}' if frac else '—'} | {count} |")
    out("")

    gapped = df[df.get("slice_gap", pd.Series(dtype=float)).fillna(0) > 0]
    if len(gapped):
        worst = gapped["imaged_fraction"].min()
        out(f"> マスク付き {len(gapped)} / {len(df)} 例にスライスギャップがある。")
        out(f"> 最も粗い症例では組織の **{worst:.0%}** しか撮像されていない。")
        out("> データセット記述はスライス厚のみを載せ、ギャップに言及していない。")
        out("> 段階3の人工劣化（設計書 5.2 の C1）は、厚みとギャップの両方を")
        out("> 再現しなければ SENORA の撮像条件を模したことにならない。")
        out("")

    # --- 2. 病変の大きさ ---
    out("## 3. 病変の大きさと形")
    out("")
    out("幾何が一致した症例のみ。")
    out("体積はボクセル間隔から求めた慣例的な値。ギャップがあるため、")
    out("実際に撮像された組織の体積はこれより小さい（`volume_imaged_ml` 列）。")
    out("")
    for reference in ("FLAIR", "DWI"):
        group = aligned[aligned.reference == reference]
        if group.empty:
            continue
        vols = group["volume_ml"]
        out(f"### {reference} 上のマスク（{len(group)} 例）")
        out("")
        out(f"- 体積 中位 **{vols.median():.1f} mL**（"
            f"最小 {vols.min():.1f} / 最大 {vols.max():.1f}）")
        out(f"- 連結成分数 中位 {group['components'].median():.0f}"
            f"（最大 {group['components'].max():.0f}）")
        out(f"- 広がるスライス数 中位 {group['slices'].median():.0f}"
            f"（最小 {group['slices'].min():.0f} / 最大 {group['slices'].max():.0f}）")
        out("")

    # --- 3. 信号強度 ---
    out("## 4. マスク内の信号強度")
    out("")
    out("症例内の組織中位値を1としたときの比。")
    out(f"`{HYPO_RATIO}` 未満を低信号、`{HYPER_RATIO}` 超を高信号とする。")
    out("")
    out("**この節が学習元の選択を決める。** 慢性期の空洞化した梗塞は中心が CSF 様の")
    out("低信号になる。読影医が空洞を含めて描いていれば、学習元にも同じ見え方の")
    out("病変が含まれていないと、モデルは辺縁だけを拾って中心を落とす。")
    out("")
    out("| 症例 | 参照 | 発症状態 | 読影 | 体積(mL) | p10 | p50 | p90 | 低信号率 | 高信号率 |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in aligned.sort_values(["reference", "subject"]).iterrows():
        out(f"| {r['subject']} | {r['reference']} | {r['stage'] or '—'} | {r['raters']} "
            f"| {r['volume_ml']:.1f} | {r.get('lesion_p10', '—')} "
            f"| {r.get('lesion_p50', '—')} | {r.get('lesion_p90', '—')} "
            f"| {r.get('frac_hypo', '—')} | {r.get('frac_hyper', '—')} |")
    out("")

    for reference in ("FLAIR", "DWI"):
        group = aligned[(aligned.reference == reference) & aligned.frac_hypo.notna()]
        if group.empty:
            continue
        out(f"{reference} 上のマスク {len(group)} 例の要約:")
        out("")
        out(f"- 低信号率 中位 **{group['frac_hypo'].median():.3f}**"
            f"（最小 {group['frac_hypo'].min():.3f} / 最大 {group['frac_hypo'].max():.3f}）")
        out(f"- 高信号率 中位 **{group['frac_hyper'].median():.3f}**")
        out(f"- 低信号率が 0.2 を超える症例: **{int((group['frac_hypo'] > 0.2).sum())}** / {len(group)}")
        out("")

    # --- 3.5 体積と低信号率の関係 ---
    flair = aligned[(aligned.reference == "FLAIR") & aligned.frac_hypo.notna()]
    if len(flair) >= 5:
        rho, pval = stats.spearmanr(flair["volume_ml"], flair["frac_hypo"])
        out("### 体積と低信号率の関係")
        out("")
        out(f"Spearman 順位相関 **ρ = {rho:.2f}**（p = {pval:.3f}、n = {len(flair)}）。")
        out("")
        small = flair[flair["volume_ml"] < 5]
        large = flair[flair["volume_ml"] >= 5]
        out(f"- 5 mL 未満（{len(small)} 例）: 低信号率 中位 {small['frac_hypo'].median():.3f}")
        out(f"- 5 mL 以上（{len(large)} 例）: 低信号率 中位 {large['frac_hypo'].median():.3f}")
        out("")
        out("小さい病変は一様に高信号のまま、大きい病変は低信号成分を持つ。")
        out("陳旧性の領域梗塞が空洞化し、ラクナ梗塞やグリオーシスは高信号にとどまる、")
        out("という臨床像と一致する。学習元には両方の見え方が含まれている必要がある。")
        out("")

    # --- 4. 発症状態の内訳 ---
    out("## 5. 発症状態の内訳")
    out("")
    out("| 参照 | 発症状態 | 例数 |")
    out("|---|---|---|")
    for reference in ("FLAIR", "DWI"):
        group = aligned[aligned.reference == reference]
        for stage, count in Counter(
            g or "（未記録）" for g in group["stage"]
        ).most_common():
            out(f"| {reference} | {stage} | {count} |")
    out("")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "mask_characterization.csv"
    df.to_csv(csv_path, index=False)
    out(f"症例ごとの測定値を `{csv_path}` に出力しました。")
    out("")

    report = "\n".join(lines)
    target = args.out / "mask_characterization.md"
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nレポートを {target} に保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
