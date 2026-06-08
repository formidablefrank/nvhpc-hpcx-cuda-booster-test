#!/usr/bin/env python3
"""Parse distributed matmul Slurm logs and regenerate timing CSVs/plots.

Usage:
    python plots/generate_timing_plots.py logs/slurm-matmul-44177268.out
    python plots/generate_timing_plots.py logs/slurm-matmul-44177268.out --plots-dir plots
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(r"^=== Scaling run: mode=(\w+) nodes=(\d+) ranks=(\d+) output=(.+?) ===$")
VALIDATION_RE = re.compile(
    r"VALIDATION max_abs_error=\s*([0-9.Ee+-]+) "
    r"max_rel_error=\s*([0-9.Ee+-]+) "
    r"max_abs_expected=\s*([0-9.Ee+-]+)"
)
READBACK_RE = re.compile(r"READBACK max_abs_error=\s*([0-9.Ee+-]+)")
TIMING_RE = re.compile(
    r"TIMING ranks=(\d+)\s+"
    r"init_s=\s*([0-9.Ee+-]+)\s+"
    r"computation_s=\s*([0-9.Ee+-]+)\s+"
    r"communication_s=\s*([0-9.Ee+-]+)\s+"
    r"io_s=\s*([0-9.Ee+-]+)\s+"
    r"io_write_s=\s*([0-9.Ee+-]+)\s+"
    r"io_read_s=\s*([0-9.Ee+-]+)\s+"
    r"validation_s=\s*([0-9.Ee+-]+)\s+"
    r"total_s=\s*([0-9.Ee+-]+)"
)

TIMING_FIELDS = [
    "mode",
    "nodes",
    "ranks",
    "output",
    "timing_ranks",
    "init_s",
    "computation_s",
    "communication_s",
    "io_s",
    "io_write_s",
    "io_read_s",
    "validation_s",
    "total_s",
    "validation_max_abs_error",
    "validation_max_rel_error",
    "validation_max_abs_expected",
    "readback_max_abs_error",
]

SUMMARY_FIELDS = [
    "nodes",
    "baseline",
    "tuned",
    "speedup_tuned_vs_baseline",
    "total_delta_s_baseline_minus_tuned",
    "write_delta_s_baseline_minus_tuned",
    "read_delta_s_baseline_minus_tuned",
    "io_delta_s_baseline_minus_tuned",
]


def parse_log(log_path: Path) -> List[Dict[str, Union[float, int, str]]]:
    rows: List[Dict[str, Union[float, int, str]]] = []
    current: Optional[Dict[str, Union[float, int, str]]] = None

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()

        match = RUN_RE.match(line)
        if match:
            current = {
                "mode": match.group(1),
                "nodes": int(match.group(2)),
                "ranks": int(match.group(3)),
                "output": match.group(4),
            }
            continue

        if current is None:
            continue

        match = VALIDATION_RE.search(line)
        if match:
            current["validation_max_abs_error"] = float(match.group(1))
            current["validation_max_rel_error"] = float(match.group(2))
            current["validation_max_abs_expected"] = float(match.group(3))
            continue

        match = READBACK_RE.search(line)
        if match:
            current["readback_max_abs_error"] = float(match.group(1))
            continue

        match = TIMING_RE.search(line)
        if match:
            current.update(
                {
                    "timing_ranks": int(match.group(1)),
                    "init_s": float(match.group(2)),
                    "computation_s": float(match.group(3)),
                    "communication_s": float(match.group(4)),
                    "io_s": float(match.group(5)),
                    "io_write_s": float(match.group(6)),
                    "io_read_s": float(match.group(7)),
                    "validation_s": float(match.group(8)),
                    "total_s": float(match.group(9)),
                }
            )
            rows.append(current)
            current = None

    rows.sort(key=lambda row: (int(row["nodes"]), 0 if row["mode"] == "baseline" else 1))
    return rows


def validate_rows(rows: List[Dict[str, Union[float, int, str]]]) -> None:
    if not rows:
        raise SystemExit("No timing rows found in log")

    missing = [field for row in rows for field in TIMING_FIELDS if field not in row]
    if missing:
        unique = ", ".join(sorted(set(missing)))
        raise SystemExit(f"Parsed rows are missing required fields: {unique}")

    seen = {(int(row["nodes"]), str(row["mode"])) for row in rows}
    for nodes in sorted({int(row["nodes"]) for row in rows}):
        if (nodes, "baseline") not in seen or (nodes, "tuned") not in seen:
            raise SystemExit(f"Missing baseline/tuned pair for {nodes} node(s)")


def write_csvs(rows: List[Dict[str, Union[float, int, str]]], plots_dir: Path, title_suffix: str) -> List[Dict[str, Union[float, int]]]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (plots_dir / f"timing_components_{title_suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    summary: List[Dict[str, Union[float, int]]] = []
    for nodes in sorted({int(row["nodes"]) for row in rows}):
        baseline = by[(nodes, "baseline")]
        tuned = by[(nodes, "tuned")]
        summary.append(
            {
                "nodes": nodes,
                "baseline": float(baseline["total_s"]),
                "tuned": float(tuned["total_s"]),
                "speedup_tuned_vs_baseline": float(baseline["total_s"]) / float(tuned["total_s"]),
                "total_delta_s_baseline_minus_tuned": float(baseline["total_s"]) - float(tuned["total_s"]),
                "write_delta_s_baseline_minus_tuned": float(baseline["io_write_s"]) - float(tuned["io_write_s"]),
                "read_delta_s_baseline_minus_tuned": float(baseline["io_read_s"]) - float(tuned["io_read_s"]),
                "io_delta_s_baseline_minus_tuned": float(baseline["io_s"]) - float(tuned["io_s"]),
            }
        )

    with (plots_dir / f"timing_total_{title_suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    return summary


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )



def format_bar_value(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def annotate_bars(ax, bars) -> None:
    ymin, ymax = ax.get_ylim()
    is_log = ax.get_yscale() == "log"
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        x = bar.get_x() + bar.get_width() / 2
        if is_log:
            y = height * 1.04
        else:
            y = height + (ymax - ymin) * 0.015
        ax.text(
            x,
            y,
            format_bar_value(float(height)),
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=7,
            clip_on=False,
        )

def plot_components(rows: List[Dict[str, Union[float, int, str]]], plots_dir: Path, title_suffix: str) -> None:
    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    nodes = sorted({int(row["nodes"]) for row in rows})
    x = np.arange(len(nodes))
    width = 0.38
    node_labels = [str(node) for node in nodes]
    colors = {"baseline": "#3b6ea8", "tuned": "#c45a2c"}
    components = [
        ("init_s", "Initialization"),
        ("computation_s", "Computation"),
        ("communication_s", "Communication"),
        ("io_write_s", "Parallel HDF5 Write"),
        ("io_read_s", "Parallel HDF5 Read"),
        ("validation_s", "Validation"),
        ("total_s", "Total"),
    ]

    fig, axes = plt.subplots(1, 7, figsize=(42, 6), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, components):
        baseline_values = [float(by[(node, "baseline")][key]) for node in nodes]
        tuned_values = [float(by[(node, "tuned")][key]) for node in nodes]
        baseline_bars = ax.bar(x - width / 2, baseline_values, width, label="baseline", color=colors["baseline"])
        tuned_bars = ax.bar(x + width / 2, tuned_values, width, label="tuned", color=colors["tuned"])
        ax.set_title(title)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Seconds")
        ax.set_xticks(x, node_labels)
        if max(max(baseline_values), max(tuned_values)) > 20:
            ax.set_yscale("log")
            ax.set_ylabel("Seconds (log)")
        annotate_bars(ax, baseline_bars)
        annotate_bars(ax, tuned_bars)
    # axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(f"Distributed Matmul Timing Components: Baseline vs Tuned ({title_suffix})", y=1.03, fontsize=14)
    fig.savefig(plots_dir / f"timing_components_{title_suffix}.png", bbox_inches="tight")
    plt.close(fig)


def plot_total(
    rows: List[Dict[str, Union[float, int, str]]],
    summary: List[Dict[str, Union[float, int]]],
    plots_dir: Path,
    title_suffix: str,
) -> None:
    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    nodes = sorted({int(row["nodes"]) for row in rows})
    x = np.arange(len(nodes))
    width = 0.38
    colors = {"baseline": "#3b6ea8", "tuned": "#c45a2c"}

    fig, ax1 = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    baseline_totals = [float(by[(node, "baseline")]["total_s"]) for node in nodes]
    tuned_totals = [float(by[(node, "tuned")]["total_s"]) for node in nodes]
    speedups = [float(row["speedup_tuned_vs_baseline"]) for row in summary]
    baseline_bars = ax1.bar(x - width / 2, baseline_totals, width, label="baseline total", color=colors["baseline"])
    tuned_bars = ax1.bar(x + width / 2, tuned_totals, width, label="tuned total", color=colors["tuned"])
    ax1.set_xticks(x, [str(node) for node in nodes])
    ax1.set_xlabel("Nodes")
    ax1.set_ylabel("Total runtime (s)")
    ax1.grid(True, axis="y", alpha=0.25)
    annotate_bars(ax1, baseline_bars)
    annotate_bars(ax1, tuned_bars)
    ax2 = ax1.twinx()
    ax2.plot(x, speedups, marker="o", color="#222222", label="speedup tuned/baseline")
    ax2.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax2.set_ylabel("Speedup: baseline / tuned")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)
    ax1.set_title(f"Total Runtime and Tuned Speedup ({title_suffix})")
    fig.savefig(plots_dir / f"timing_total_{title_suffix}.png", bbox_inches="tight")
    plt.close(fig)


def infer_title_suffix(log_path: Path) -> str:
    match = re.search(r"slurm-matmul-(\d+)\.out$", log_path.name)
    if match:
        # return f"Slurm {match.group(1)}"
        return match.group(1)
    return log_path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate timing CSVs and plots from a matmul Slurm log.")
    parser.add_argument("log", type=Path, help="Path to logs/slurm-matmul-<jobid>.out")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output directory for CSV and PNG files. Defaults to the directory containing this script.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title suffix. Defaults to the Slurm job id.")
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve() if args.plots_dir else Path(__file__).resolve().parent
    title_suffix = args.title or infer_title_suffix(log_path)

    rows = parse_log(log_path)
    validate_rows(rows)
    summary = write_csvs(rows, plots_dir, title_suffix)

    configure_matplotlib()
    plot_components(rows, plots_dir, title_suffix)
    plot_total(rows, summary, plots_dir, title_suffix)

    print(f"Parsed {len(rows)} timing rows from {log_path}")
    print(f"Wrote CSVs and plots to {plots_dir}")
    for row in rows:
        print(
            f"{str(row['mode']):8s} nodes={int(row['nodes'])} "
            f"total={float(row['total_s']):.6f} "
            f"write={float(row['io_write_s']):.6f} "
            f"read={float(row['io_read_s']):.6f} "
            f"readback_err={float(row['readback_max_abs_error']):.1e}"
        )


if __name__ == "__main__":
    main()
