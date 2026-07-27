"""
notion_upload.py
batch_run.py の結果をcatbox.moeにアップロードして
Notionページに比較表として貼り付ける。

使い方:
  python notion_upload.py --summary ./demo_output/batch_summary.json
"""

import argparse
import json
from pathlib import Path


def upload_image(image_path: str, retries: int = 3, wait: float = 2.0) -> str:
    """画像をアップロードして公開URLを返す。複数ホストを順番に試す。"""
    import requests, time

    hosts = [
        # 0x0.st（メイン）
        lambda f: _try_0x0(f, requests),
        # litterbox.catbox.moe（72時間保持）
        lambda f: _try_litterbox(f, requests),
    ]

    for attempt in range(retries):
        for host_fn in hosts:
            try:
                with open(image_path, "rb") as f:
                    url = host_fn(f)
                if url and url.startswith("http"):
                    return url
            except Exception as e:
                print(f" ({e})", end="", flush=True)
        print(f" 待機{wait*(attempt+1):.0f}s...", end="", flush=True)
        time.sleep(wait * (attempt + 1))

    raise RuntimeError(f"全ホストでアップロード失敗: {image_path}")


def _try_0x0(f, requests):
    resp = requests.post("https://0x0.st", files={"file": f}, timeout=60)
    url = resp.text.strip()
    return url if url.startswith("http") else None


def _try_litterbox(f, requests):
    resp = requests.post(
        "https://litterbox.catbox.moe/resources/internals/api.php",
        data={"reqtype": "fileupload", "time": "72h"},
        files={"fileToUpload": f},
        timeout=60,
    )
    url = resp.text.strip()
    return url if url.startswith("https://") else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="./demo_output/batch_summary.json")
    args = parser.parse_args()

    with open(args.summary) as f:
        results = json.load(f)

    success_results = [r for r in results if r["success"] and r["folder"]]
    total = len(success_results)

    print(f"アップロード対象: {total} 件")

    uploaded = []
    for i, r in enumerate(success_results):
        t = r["t_noise"]
        folder = Path(r["folder"])
        img_path = folder / "result_analysis.png"

        if not img_path.exists():
            print(f"  [{i+1}/{total}] t={t}: ファイルなし → スキップ")
            continue

        print(f"  [{i+1}/{total}] t={t} アップロード中...", end=" ", flush=True)
        url = upload_image(str(img_path))
        print(url)
        uploaded.append({"t_noise": t, "url": url})

        # レート制限対策：アップロード間に2秒待機
        import time; time.sleep(2)

    # アップロード結果を保存
    upload_result_path = Path(args.summary).parent / "upload_results.json"
    with open(upload_result_path, "w") as f:
        json.dump(uploaded, f, indent=2)

    print(f"\nアップロード完了: {len(uploaded)}/{total}")
    print(f"結果: {upload_result_path}")
    print("\nNotion更新はClaudeに「notion_uploadの結果をNotionに反映して」と伝えてください")


if __name__ == "__main__":
    main()
