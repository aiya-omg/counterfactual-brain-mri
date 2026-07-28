"""
fetch_isles2022.py
ISLES 2022 training set を Zenodo から取得して展開する。

DOI: 10.5281/zenodo.7153326（レコード 7153326、約1.7GB、CC BY 4.0）
申請不要。アームA（DWI）とアームC（FLAIR）の共通学習元。

使い方:
  python fetch_isles2022.py
  python fetch_isles2022.py --no-extract   # ダウンロードのみ
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_API = "https://zenodo.org/api"
RECORD_ID = "7153326"
ARCHIVE_NAME = "ISLES-2022.zip"
CHUNK = 1 << 20  # 1 MiB


def fetch_record() -> dict:
    resp = requests.get(f"{ZENODO_API}/records/{RECORD_ID}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict, dest_dir: Path) -> Path:
    name = entry["key"]
    expected_size = entry["size"]
    expected_md5 = entry["checksum"].removeprefix("md5:")
    url = entry["links"]["self"]
    dest = dest_dir / name

    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {name} は取得済み。チェックサムを確認します")
        if md5_of(dest) == expected_md5:
            print(f"[ok]   {name} チェックサム一致")
            return dest
        print(f"[warn] {name} チェックサム不一致。取り直します")
        dest.unlink()

    resume_from = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    mode = "ab" if resume_from else "wb"
    if resume_from:
        print(f"[resume] {name} を {resume_from:,} バイト目から再開します")

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, mode) as f, tqdm(
            total=expected_size,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                bar.update(len(chunk))

    actual_md5 = md5_of(dest)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"{name} のチェックサムが一致しません。\n"
            f"  期待値: {expected_md5}\n"
            f"  実際:   {actual_md5}\n"
            f"ファイルを削除して再実行してください。"
        )
    print(f"[ok]   {name} チェックサム一致")
    return dest


def extract(archive: Path, dest_dir: Path) -> None:
    print(f"[extract] {archive.name} -> {dest_dir}")
    with zipfile.ZipFile(archive) as zf:
        for member in tqdm(zf.namelist(), desc=f"展開 {archive.name}", unit="file"):
            zf.extract(member, dest_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "data",
        help="出力先（既定: senora/data）",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="ダウンロードのみ行い展開しない",
    )
    args = parser.parse_args()

    archive_dir = args.out / "_archives"
    bids_dir = args.out / "isles2022"
    archive_dir.mkdir(parents=True, exist_ok=True)
    bids_dir.mkdir(parents=True, exist_ok=True)

    print("Zenodo レコードを取得しています...")
    record = fetch_record()
    print(f"レコード {record['id']} / DOI {record['doi']}")

    files = [f for f in record["files"] if f["key"] == ARCHIVE_NAME]
    if not files:
        print(f"エラー: {ARCHIVE_NAME} が見つかりません", file=sys.stderr)
        return 1

    total = sum(f["size"] for f in files)
    print(f"取得対象 {len(files)} ファイル / 合計 {total / 1e9:.2f} GB")

    downloaded = [download(entry, archive_dir) for entry in files]

    if args.no_extract:
        print(f"\n展開をスキップしました。ZIP は {archive_dir} にあります")
        return 0

    for archive in downloaded:
        extract(archive, bids_dir)

    print(f"\n完了しました。BIDS ツリー: {bids_dir}")
    print("次: python register_isles_flair.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
