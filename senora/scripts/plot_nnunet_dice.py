"""
plot_nnunet_dice.py
nnU-Net の学習ログから Dice と損失の推移を描く。

nnU-Net は fold ごとに progress.png を出すが、学習を中断して再開すると
新しいログファイルが作られ、図も再開後の区間しか描かれない。
アームCの fold 0 は途中で PC を落として再開しているため、
複数のログを epoch 番号で束ね直して通しの推移にする。

読むのは <nnUNet_results>/.../fold_N/training_log_*.txt であり、
端末の出力には依存しない。

使い方:
  python plot_nnunet_dice.py
  python plot_nnunet_dice.py --fold 1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EPOCH = re.compile(r"Epoch (\d+)\s*$")
DICE = re.compile(r"Pseudo dice \[(?:np\.float32\()?([0-9.]+)")
TRAIN_LOSS = re.compile(r"train_loss (-?[0-9.]+)")
VAL_LOSS = re.compile(r"val_loss (-?[0-9.]+)")
BEST_EMA = re.compile(r"New best EMA pseudo Dice: ([0-9.]+)")
DONE = re.compile(r"Training done")


def parse_logs(fold_dir: Path) -> dict:
    """
    fold の全ログを時系列に読み、epoch 番号を鍵に値を集める。

    再開すると同じ epoch が2回記録されることがある。後から書かれた側が
    実際に採用された学習なので、辞書への上書きで自然に後勝ちになる。
    """
    logs = sorted(
        fold_dir.glob("training_log_*.txt"), key=lambda p: p.stat().st_mtime
    )
    if not logs:
        raise SystemExit(f"学習ログが見つかりません: {fold_dir}")

    dice: dict[int, float] = {}
    train_loss: dict[int, float] = {}
    val_loss: dict[int, float] = {}
    best_ema: list[tuple[int, float]] = []
    finished = False
    current: int | None = None

    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = EPOCH.search(line)
            if m:
                current = int(m.group(1))
                continue
            if DONE.search(line):
                finished = True
            if current is None:
                continue
            m = DICE.search(line)
            if m:
                dice[current] = float(m.group(1))
                continue
            m = TRAIN_LOSS.search(line)
            if m:
                train_loss[current] = float(m.group(1))
                continue
            m = VAL_LOSS.search(line)
            if m:
                val_loss[current] = float(m.group(1))
                continue
            m = BEST_EMA.search(line)
            if m:
                best_ema.append((current, float(m.group(1))))

    return {
        "dice": dice,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_ema": sorted(best_ema),
        "finished": finished,
        "logs": logs,
    }


def as_series(values: dict[int, float]) -> tuple[list[int], list[float]]:
    keys = sorted(values)
    return keys, [values[k] for k in keys]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset", default="Dataset501_ISLES22FLAIR")
    parser.add_argument("--trainer", default="nnUNetTrainer__nnUNetPlans__3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--results", type=Path, default=Path.home() / "senora_nnunet" / "nnUNet_results"
    )
    parser.add_argument("--out", type=Path, default=root / "results")
    parser.add_argument("--window", type=int, default=11, help="移動平均の窓幅")
    args = parser.parse_args()

    fold_dir = args.results / args.dataset / args.trainer / f"fold_{args.fold}"
    if not fold_dir.is_dir():
        print(f"エラー: {fold_dir} がありません", file=sys.stderr)
        return 1

    parsed = parse_logs(fold_dir)
    epochs, dice = as_series(parsed["dice"])
    if not epochs:
        print("エラー: Dice の記録がありません", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax = axes[0]
    ax.plot(epochs, dice, color="#4C78A8", alpha=0.35, linewidth=1.0,
            label="val Pseudo Dice")

    w = args.window
    if len(dice) >= w:
        smooth = np.convolve(dice, np.ones(w) / w, mode="valid")
        ax.plot(epochs[w - 1:], smooth, color="#4C78A8", linewidth=2.0,
                label=f"rolling mean (w={w})")

    if parsed["best_ema"]:
        ema_epochs = [e for e, _ in parsed["best_ema"]]
        ema_dice = [d for _, d in parsed["best_ema"]]
        ax.step(ema_epochs, ema_dice, where="post", color="#F58518", linewidth=2.0,
                label="best EMA Dice (nnU-Net)")
        ax.scatter([ema_epochs[-1]], [ema_dice[-1]], color="#F58518", s=40, zorder=5)
        ax.annotate(
            f"{ema_dice[-1]:.3f}",
            (ema_epochs[-1], ema_dice[-1]),
            textcoords="offset points",
            xytext=(6, 6),
            color="#F58518",
            fontsize=10,
        )

    state = "done" if parsed["finished"] else f"running, last={epochs[-1]}"
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Arm C nnU-Net fold{args.fold} — Dice over epochs "
        f"(n={len(epochs)}, {state})"
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for values, color, label in (
        (parsed["train_loss"], "#54A24B", "train_loss"),
        (parsed["val_loss"], "#E45756", "val_loss"),
    ):
        if values:
            xs, ys = as_series(values)
            ax2.plot(xs, ys, color=color, linewidth=1.2, label=label)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / f"nnunet_{args.dataset}_fold{args.fold}_dice.png"
    fig.savefig(png, dpi=150)

    print(f"読んだログ: {len(parsed['logs'])} 本")
    for log in parsed["logs"]:
        print(f"  {log.name}")
    print(f"保存: {png}")
    print(f"epoch {epochs[0]}..{epochs[-1]}（記録 {len(epochs)} 点）"
          f"{'／学習完了' if parsed['finished'] else '／学習中'}")
    print(f"Dice 最小 {min(dice):.3f} / 最大 {max(dice):.3f} / 最終 {dice[-1]:.3f}")
    if parsed["best_ema"]:
        print(f"best EMA Dice {parsed['best_ema'][-1][1]:.4f}"
              f"（epoch {parsed['best_ema'][-1][0]}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
