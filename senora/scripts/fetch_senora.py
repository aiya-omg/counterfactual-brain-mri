"""
fetch_senora.py
SENORA-MRI を Zenodo から取得して展開する。

Zenodo の concept DOI (10.5281/zenodo.20757772) から常に最新版を解決する。
初版レコード (20757773) にはメタデータプレビューしか含まれていないため、
バージョン番号を直接指定せず concept record 経由で解決している。

使い方:
  # まず participants.tsv だけ取得（5KB、数秒）。段階0の判断はこれで足りる
  python fetch_senora.py --metadata-only

  # 画像本体も取得（約5GB、2分割ZIP）
  python fetch_senora.py

  # 中断した場合は同じコマンドで再開できる（HTTP Range による続きからのダウンロード）
  python fetch_senora.py
"""

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_API = "https://zenodo.org/api"
CONCEPT_RECORD = "20757772"

METADATA_FILE = "SENORA-MRI_metadata_preview.zip"
CHUNK = 1 << 20  # 1 MiB


def resolve_latest_record() -> dict:
    """concept record から最新バージョンのレコードを解決する。"""
    url = f"{ZENODO_API}/records/{CONCEPT_RECORD}/versions/latest"
    resp = requests.get(url, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict, dest_dir: Path) -> Path:
    """
    1ファイルをダウンロードする。既存ファイルが完全ならスキップし、
    途中までなら Range ヘッダで続きから取得する。
    """
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
    """
    ZIP を展開する。2分割ZIPは同一フォルダに展開して BIDS ツリーを再構成する。
    """
    print(f"[extract] {archive.name} -> {dest_dir}")
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=f"展開 {archive.name}", unit="file"):
            zf.extract(member, dest_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="出力先（既定: senora/data）",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="participants.tsv を含むメタデータZIPのみ取得する（約5KB）",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="ダウンロードのみ行い展開しない",
    )
    args = parser.parse_args()

    archive_dir = args.out / "_archives"
    bids_dir = args.out / "senora"
    archive_dir.mkdir(parents=True, exist_ok=True)
    bids_dir.mkdir(parents=True, exist_ok=True)

    print("Zenodo から最新バージョンを解決しています...")
    record = resolve_latest_record()
    version = record["metadata"].get("version", "不明")
    print(f"レコード {record['id']} / バージョン {version} / DOI {record['doi']}")

    files = record["files"]
    if args.metadata_only:
        files = [f for f in files if f["key"] == METADATA_FILE]
        if not files:
            print(f"エラー: {METADATA_FILE} が見つかりません", file=sys.stderr)
            return 1
    else:
        total = sum(f["size"] for f in files)
        print(f"取得対象 {len(files)} ファイル / 合計 {total / 1e9:.2f} GB")

    downloaded = [download(entry, archive_dir) for entry in files]

    if args.no_extract:
        print(f"\n展開をスキップしました。ZIP は {archive_dir} にあります")
        return 0

    for archive in downloaded:
        extract(archive, bids_dir)

    print(f"\n完了しました。BIDS ツリー: {bids_dir}")
    print("次: python inventory_senora.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
