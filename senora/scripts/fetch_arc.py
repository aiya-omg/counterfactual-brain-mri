"""
fetch_arc.py
慢性期モデルの学習元として、Aphasia Recovery Cohort（ARC、OpenNeuro ds004884、CC0）から
FLAIR・病変マスク・マスクを描いた T2w を取ってくる。

ARC の病変マスクは T2w の上に描かれていて、FLAIR は別の日に撮られていることが多い。
セッション名は発症後の日数なので、マスクごとに日数が最も近い FLAIR を選び、
差が MAX_GAP_DAYS 以下の症例だけを使う。全例が発症後 6 か月以上の慢性期で、
数か月の差では病変の形は変わらないとみなす。

公開 S3 から認証なしで取る。取った後は prepare_nnunet_arc.py で T2w を FLAIR に剛体登録し、
マスクを FLAIR の格子へ写す。

使い方:
  python fetch_arc.py            # 対応表を作ってダウンロード
  python fetch_arc.py --list     # 対応表だけ
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds004884"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
MAX_GAP_DAYS = 90
KEY = re.compile(r"(sub-[^/]+)/ses-(\d+)/anat/([^/]+)$")


def list_keys(prefix: str):
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        root = ET.fromstring(urllib.request.urlopen(
            f"{BUCKET}?{urllib.parse.urlencode(query)}", timeout=60).read())
        for item in root.findall("s3:Contents", NS):
            yield item.find("s3:Key", NS).text
        if root.find("s3:IsTruncated", NS).text != "true":
            break
        token = root.find("s3:NextContinuationToken", NS).text


def pair_masks_with_flair(keys: list[str]) -> pd.DataFrame:
    masks, flairs = [], []
    for key in keys:
        m = KEY.search(key)
        if not m:
            continue
        subject, day, name = m.group(1), int(m.group(2)), m.group(3)
        if "/derivatives/lesion_masks/" in key and name.endswith("_desc-lesion_mask.nii.gz"):
            t2w = key.replace("/derivatives/lesion_masks/", "/").replace("_desc-lesion_mask", "")
            masks.append({"subject": subject, "mask_day": day, "mask_key": key, "t2w_key": t2w})
        elif "/derivatives/" not in key and name.endswith("_FLAIR.nii.gz"):
            acq = re.search(r"acq-([^_]+)", name)
            flairs.append({"subject": subject, "flair_day": day, "flair_key": key,
                           "flair_acq": acq.group(1) if acq else ""})
    pairs = pd.DataFrame(masks).merge(pd.DataFrame(flairs), on="subject")
    pairs["gap_days"] = (pairs["flair_day"] - pairs["mask_day"]).abs()
    best = (pairs.sort_values(["subject", "gap_days", "flair_day"])
            .groupby("subject", as_index=False).head(1).reset_index(drop=True))
    best["t2w_listed"] = best["t2w_key"].isin(set(keys))
    return best


def download(key: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}", timeout=120) as r:
        tmp.write_bytes(r.read())
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path.home() / "senora_nnunet" / "arc")
    parser.add_argument("--max-gap", type=int, default=MAX_GAP_DAYS)
    parser.add_argument("--list", action="store_true", help="対応表だけ作る")
    args = parser.parse_args()

    keys = list(list_keys(f"{DATASET}/"))
    best = pair_masks_with_flair(keys)
    best["used"] = (best["gap_days"] <= args.max_gap) & best["t2w_listed"]
    args.out.mkdir(parents=True, exist_ok=True)
    best.to_csv(args.out / "arc_pairs.csv", index=False)
    used = best[best["used"]]
    print(f"マスクと FLAIR の両方がある症例 {len(best)}、日数差 {args.max_gap} 日以下 {len(used)}")
    if args.list:
        return 0

    for index, row in enumerate(used.itertuples(), start=1):
        case = args.out / "raw" / row.subject
        download(row.flair_key, case / "flair.nii.gz")
        download(row.t2w_key, case / "t2w.nii.gz")
        download(row.mask_key, case / "mask_t2w.nii.gz")
        print(f"{index}/{len(used)} {row.subject}", flush=True)
    print(f"完了 → {args.out / 'raw'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
