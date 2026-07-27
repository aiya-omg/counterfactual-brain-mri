"""
batch_run.py
t_noise を 0〜1000 の 50刻みで全実行し、結果をまとめる。

使い方:
  conda activate cfmri
  python batch_run.py --input_image ./data/slices/tumor/BraTS-GLI-00000-000_slice080.png \
                      --gt_mask ./data/slices/masks/BraTS-GLI-00000-000_slice080.png \
                      --monai_path ./monai_ddpm_model
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--gt_mask",     type=str, default=None)
    parser.add_argument("--monai_path",  type=str, default="./monai_ddpm_model")
    parser.add_argument("--output_dir",  type=str, default="./demo_output")
    parser.add_argument("--t_start",     type=int, default=0)
    parser.add_argument("--t_end",       type=int, default=1000)
    parser.add_argument("--t_step",      type=int, default=50)
    args = parser.parse_args()

    t_values = list(range(args.t_start, args.t_end + 1, args.t_step))
    total = len(t_values)

    print("=" * 60)
    print(f"  Batch Run: t_noise {args.t_start}〜{args.t_end} ({args.t_step}刻み)")
    print(f"  合計: {total} 回")
    print("=" * 60)

    results = []  # {t_noise, folder, success}

    for i, t in enumerate(t_values):
        print(f"\n[{i+1}/{total}] t_noise={t} を実行中...")

        cmd = [
            sys.executable, "demo.py",
            "--input_image", args.input_image,
            "--use_monai",
            "--monai_path", args.monai_path,
            "--output_dir", args.output_dir,
            "--t_noise", str(t),
        ]
        if args.gt_mask:
            cmd += ["--gt_mask", args.gt_mask]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)

        if result.returncode == 0:
            # 作成されたフォルダを特定（最新のフォルダ）
            output_path = Path(args.output_dir)
            folders = sorted(output_path.glob(f"*_MONAI-DDPM_t{t}"), key=lambda p: p.stat().st_mtime)
            folder = str(folders[-1]) if folders else "unknown"
            print(f"  完了 -> {folder}")
            results.append({"t_noise": t, "folder": folder, "success": True})
        else:
            print(f"  失敗: {result.stderr[-200:]}")
            results.append({"t_noise": t, "folder": None, "success": False})

    # 結果サマリーをJSONで保存
    summary_path = Path(args.output_dir) / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"  完了: {success}/{total} 成功")
    print(f"  サマリー: {summary_path}")
    print(f"  次: python notion_upload.py --summary {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
