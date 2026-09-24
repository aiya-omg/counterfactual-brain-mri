"""
download_brats.py
BraTS 2023 データダウンロードスクリプト

使い方:
  py -3.11 download_brats.py
"""

import synapseclient
import getpass
from pathlib import Path

print("=" * 50)
print("  BraTS 2023 Downloader")
print("=" * 50)
print()

# トークン入力（画面には表示されない）
token = getpass.getpass("Synapse Auth Token を貼り付けてEnter: ")

print()
print("ログイン中...")
syn = synapseclient.Synapse()
syn.login(authToken=token.strip())
print("ログイン成功!")

# 保存先
save_dir = Path("./data/BraTS2023")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"保存先: {save_dir.resolve()}")
print()
print("ダウンロード開始... (12GB、時間がかかります)")

# ダウンロードリストから取得
dl_list_file_entities = syn.get_download_list()

print()
print("=" * 50)
print(f"  完了! -> {save_dir.resolve()}")
print("=" * 50)
