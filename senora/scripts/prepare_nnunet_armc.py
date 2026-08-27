"""
prepare_nnunet_armc.py
段階1: 登録済み ISLES 2022 FLAIR を nnU-Net v2 の Dataset 形式に並べる。

出力:
  <nnUNet_raw>/Dataset501_ISLES22FLAIR/
    imagesTr/  ISLES_XXXX_0000.nii.gz   # FLAIR
    labelsTr/  ISLES_XXXX.nii.gz        # lesion
    dataset.json

使い方:
  # 環境変数をセットしてから
  set NNUNET_RAW=...\\senora\\data\\nnunet\\nnUNet_raw
  set NNUNET_PREPROCESSED=...\\senora\\data\\nnunet\\nnUNet_preprocessed
  set NNUNET_RESULTS=...\\senora\\data\\nnunet\\nnUNet_results
  python prepare_nnunet_armc.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--src",
        type=Path,
        default=root / "data" / "isles2022_flair",
        help="register_isles_flair.py の出力",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path.home() / "senora_nnunet" / "nnUNet_raw",
        help="nnUNet_raw 相当（Windows では日本語パスを避ける）",
    )
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", type=str, default="ISLES22FLAIR")
    parser.add_argument(
        "--description", type=str, default=None,
        help="dataset.json に載せる説明。既定は等方版の文言",
    )
    parser.add_argument(
        "--skip-empty-labels", action="store_true",
        help="マスクが空の症例を除く。分解能を落とすと薄い病変が消えるため、"
             "病変なしと教える症例が混ざるのを避けたいときに使う",
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"エラー: {args.src} がありません。先に register_isles_flair.py を実行してください",
              file=sys.stderr)
        return 1

    cases = sorted(args.src.glob("sub-*/ses-*/anat/*_FLAIR.nii.gz"))
    if not cases:
        print(f"エラー: FLAIR が見つかりません: {args.src}", file=sys.stderr)
        return 1

    ds_dir = args.raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images = ds_dir / "imagesTr"
    labels = ds_dir / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    file_ending = ".nii.gz"
    channel_names = {"0": "FLAIR"}
    labels_dict = {"background": 0, "lesion": 1}
    training = []

    # 症例IDは受理された症例の並び順に振る。元の被験者IDへ戻せるよう
    # 対応表を残す。段階1の検収で登録QCと突き合わせるのに必要
    mapping: list[dict] = []
    skipped_empty: list[str] = []

    for i, flair in enumerate(tqdm(cases, desc="nnU-Net 配置"), start=1):
        case_id = f"ISLES_{i:04d}"
        mask = flair.with_name(flair.name.replace("_FLAIR.nii.gz", "_label-lesion_roi.nii.gz"))
        if not mask.exists():
            print(f"[warn] マスクなし、スキップ: {flair}")
            continue
        if args.skip_empty_labels and not np.asarray(nib.load(mask).dataobj).any():
            skipped_empty.append(flair.parents[2].name)
            continue
        shutil.copy2(flair, images / f"{case_id}_0000{file_ending}")
        shutil.copy2(mask, labels / f"{case_id}{file_ending}")
        training.append({"image": f"./imagesTr/{case_id}_0000{file_ending}",
                         "label": f"./labelsTr/{case_id}{file_ending}"})
        mapping.append({"case_id": case_id, "subject": flair.parents[2].name})

    dataset = {
        "channel_names": channel_names,
        "labels": labels_dict,
        "numTraining": len(training),
        "file_ending": file_ending,
        "name": args.dataset_name,
        "description": args.description or (
            "ISLES 2022 FLAIR-only stroke lesion segmentation. "
            "Masks rigidly transformed from DWI to FLAIR native space."
        ),
        "reference": "https://doi.org/10.5281/zenodo.7153326",
        "licence": "CC-BY-4.0",
        "tensorImageSize": "3D",
    }
    (ds_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    map_path = ds_dir / "case_to_subject.csv"
    with map_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case_id", "subject"])
        writer.writeheader()
        writer.writerows(mapping)

    print(f"\n完了: {len(training)} 例 → {ds_dir}")
    if skipped_empty:
        print(f"マスクが空のため除外: {len(skipped_empty)} 例")
        print(f"  {', '.join(skipped_empty)}")
    print(f"対応表: {map_path}")
    print("次:")
    print(f"  set nnUNet_raw={args.raw}")
    print(f"  set nnUNet_preprocessed={args.raw.parent / 'nnUNet_preprocessed'}")
    print(f"  set nnUNet_results={args.raw.parent / 'nnUNet_results'}")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
