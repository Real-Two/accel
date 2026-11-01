"""
Turns a set of baseline_summary.json files (one per optimization stage) into
the "waterfall" demo artifact: a bar chart of sec/epoch per stage with the
speedup multiplier labeled, plus a markdown table you can paste straight
into slides. IoU is shown per stage so "same accuracy, faster" is visible
at a glance, not just asserted.

Usage -- pass each stage as NAME PATH, in the order you want them plotted
(baseline first):

    python benchmark_report.py \
        --stage "Baseline (no tuning)" baseline_full/baseline_summary.json \
        --stage "+ Thread tuning" stage_threads/baseline_summary.json \
        --stage "+ AMP (bf16)" stage_amp/baseline_summary.json \
        --stage "+ torch.compile" stage_full/baseline_summary.json \
        --out_dir report

Writes report/waterfall.png and report/waterfall_summary.md.
"""

import argparse
import json
import os


def load_stage(path):
    with open(path) as f:
        return json.load(f)


def build_report(stages, out_dir):
    """stages: list of (name, summary_dict)"""
    os.makedirs(out_dir, exist_ok=True)
    baseline_time = stages[0][1]["avg_epoch_seconds"]

    rows = []
    for name, s in stages:
        avg = s["avg_epoch_seconds"]
        iou = s["best_val_iou"]
        speedup = baseline_time / avg if avg else float("nan")
        rows.append({"name": name, "avg_epoch_seconds": avg, "best_val_iou": iou, "speedup": speedup})

    # --- markdown table ---
    md_lines = [
        "| Stage | Avg sec/epoch | Speedup vs baseline | Best val IoU |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['name']} | {r['avg_epoch_seconds']:.1f}s | {r['speedup']:.2f}x | {r['best_val_iou']:.4f} |"
        )
    target_hit = rows[-1]["speedup"] >= 2.5
    md_lines.append("")
    md_lines.append(f"**Final speedup: {rows[-1]['speedup']:.2f}x "
                     f"({'meets' if target_hit else 'short of'} the 2.5x / 150% target)**")
    md_path = os.path.join(out_dir, "waterfall_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print("\n".join(md_lines))
    print(f"\nMarkdown table written to {md_path}")

    # --- chart (optional -- only if matplotlib is available) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["name"] for r in rows]
        times = [r["avg_epoch_seconds"] for r in rows]
        ious = [r["best_val_iou"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        bars = ax1.bar(names, times, color="#4C72B0")
        ax1.set_ylabel("Avg seconds / epoch")
        ax1.set_title("CPU training time per optimization stage")
        plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

        for bar, r in zip(bars, rows):
            ax1.annotate(f"{r['speedup']:.2f}x\nIoU {r['best_val_iou']:.3f}",
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        png_path = os.path.join(out_dir, "waterfall.png")
        fig.savefig(png_path, dpi=150)
        print(f"Chart written to {png_path}")
    except ImportError:
        print("matplotlib not installed -- skipping chart, markdown table above is still usable. "
              "`pip install matplotlib` to get the PNG too.")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", nargs=2, action="append", metavar=("NAME", "PATH"),
                         required=True, help="Repeatable. e.g. --stage \"Baseline\" path/to/summary.json")
    parser.add_argument("--out_dir", default="report")
    args = parser.parse_args()

    stages = [(name, load_stage(path)) for name, path in args.stage]
    build_report(stages, args.out_dir)
