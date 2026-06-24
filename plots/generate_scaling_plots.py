#!/usr/bin/env python3
"""Generate scaling-analysis CSVs and plots from distributed matmul Slurm logs.

This variant supports an arbitrary set of benchmark modes, including the
fine-tuning modes from job_dist_matmul_acc.sh:
baseline, mpi_pmix, numa_mem, gpu_nic_ucx, and all_tuned.

Usage:
    python plots/generate_scaling_plots.py logs/slurm-matmul-45336776.out
    python plots/generate_scaling_plots.py logs/slurm-matmul-45336776.out --plots-dir plots
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


RUN_RE = re.compile(
    r"^=== (?:Stdpar )?Scaling run: mode=([A-Za-z0-9_-]+) nodes=(\d+) ranks=(\d+) output=(.+?) ===$"
)
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
    r"(?:(?:io_write_s=\s*([0-9.Ee+-]+)\s+io_read_s=\s*([0-9.Ee+-]+)\s+))?"
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
    "mode",
    "nodes",
    "ranks",
    "total_s",
    "baseline_total_s",
    "speedup_vs_baseline",
    "delta_s_baseline_minus_mode",
    "strong_scaling_speedup_vs_mode_1node",
    "strong_scaling_efficiency_vs_mode_1node",
]

MODE_ORDER = {
    "baseline": 0,
    "mpi_pmix": 1,
    "numa_mem": 2,
    "gpu_nic_ucx": 3,
    "all_tuned": 4,
    "tuned": 5,
}
MODE_LABELS = {
    "baseline": "Baseline",
    "mpi_pmix": "MPI/PMIX",
    "numa_mem": "NUMA/Mem",
    "gpu_nic_ucx": "GPU/NIC/UCX",
    "all_tuned": "All Tuned",
    "tuned": "Tuned",
}


def mode_sort_key(mode: str) -> tuple:
    return (MODE_ORDER.get(mode, 100), mode)


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode.replace("_", " "))


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
                "validation_max_abs_error": "",
                "validation_max_rel_error": "",
                "validation_max_abs_expected": "",
                "readback_max_abs_error": "",
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
            io_write = match.group(6)
            io_read = match.group(7)
            current.update(
                {
                    "timing_ranks": int(match.group(1)),
                    "init_s": float(match.group(2)),
                    "computation_s": float(match.group(3)),
                    "communication_s": float(match.group(4)),
                    "io_s": float(match.group(5)),
                    "io_write_s": float(io_write) if io_write is not None else "",
                    "io_read_s": float(io_read) if io_read is not None else "",
                    "validation_s": float(match.group(8)),
                    "total_s": float(match.group(9)),
                }
            )
            rows.append(current)
            current = None

    rows.sort(key=lambda row: (int(row["nodes"]), mode_sort_key(str(row["mode"]))))
    return rows


def validate_rows(rows: List[Dict[str, Union[float, int, str]]]) -> None:
    if not rows:
        raise SystemExit("No timing rows found in log")

    required = ["mode", "nodes", "ranks", "timing_ranks", "init_s", "computation_s", "communication_s", "io_s", "validation_s", "total_s"]
    missing = [field for row in rows for field in required if field not in row]
    if missing:
        unique = ", ".join(sorted(set(missing)))
        raise SystemExit(f"Parsed rows are missing required fields: {unique}")

    seen = {(int(row["nodes"]), str(row["mode"])) for row in rows}
    if len(seen) != len(rows):
        raise SystemExit("Duplicate timing rows found for at least one node/mode pair")


def build_summary(rows: List[Dict[str, Union[float, int, str]]]) -> List[Dict[str, Union[float, int, str]]]:
    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    nodes = sorted({int(row["nodes"]) for row in rows})
    modes = sorted({str(row["mode"]) for row in rows}, key=mode_sort_key)
    baseline_by_node = {node: by.get((node, "baseline")) for node in nodes}

    first_node_by_mode = {}
    for mode in modes:
        mode_nodes = [node for node in nodes if (node, mode) in by]
        if mode_nodes:
            first_node_by_mode[mode] = min(mode_nodes)

    summary: List[Dict[str, Union[float, int, str]]] = []
    for node in nodes:
        for mode in modes:
            row = by.get((node, mode))
            if row is None:
                continue

            total = float(row["total_s"])
            baseline = baseline_by_node.get(node)
            baseline_total = float(baseline["total_s"]) if baseline is not None else ""
            speedup_vs_baseline = float(baseline["total_s"]) / total if baseline is not None else ""
            delta_vs_baseline = float(baseline["total_s"]) - total if baseline is not None else ""

            first_node = first_node_by_mode[mode]
            first = by[(first_node, mode)]
            strong_speedup = float(first["total_s"]) / total
            strong_efficiency = strong_speedup / (node / first_node)

            summary.append(
                {
                    "mode": mode,
                    "nodes": node,
                    "ranks": int(row["ranks"]),
                    "total_s": total,
                    "baseline_total_s": baseline_total,
                    "speedup_vs_baseline": speedup_vs_baseline,
                    "delta_s_baseline_minus_mode": delta_vs_baseline,
                    "strong_scaling_speedup_vs_mode_1node": strong_speedup,
                    "strong_scaling_efficiency_vs_mode_1node": strong_efficiency,
                }
            )

    return summary


def write_csvs(rows: List[Dict[str, Union[float, int, str]]], summary: List[Dict[str, Union[float, int, str]]], plots_dir: Path, suffix: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    with (plots_dir / f"scaling_components_{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (plots_dir / f"scaling_summary_{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)


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


def color_for_modes(modes: List[str]) -> Dict[str, str]:
    palette = ["#3b6ea8", "#c45a2c", "#4f8f45", "#7a5aa6", "#b8892f", "#5d8991", "#a34f6b"]
    return {mode: palette[index % len(palette)] for index, mode in enumerate(modes)}


def grouped_bar_layout(nodes: List[int], modes: List[str]) -> tuple:
    x = np.arange(len(nodes))
    width = min(0.8 / max(len(modes), 1), 0.22)
    return x, width


def plot_total_scaling(rows: List[Dict[str, Union[float, int, str]]], plots_dir: Path, suffix: str) -> None:
    nodes = sorted({int(row["nodes"]) for row in rows})
    modes = sorted({str(row["mode"]) for row in rows}, key=mode_sort_key)
    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    colors = color_for_modes(modes)

    x, width = grouped_bar_layout(nodes, modes)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    for index, mode in enumerate(modes):
        offsets = x + (index - (len(modes) - 1) / 2) * width
        values = [float(by[(node, mode)]["total_s"]) if (node, mode) in by else np.nan for node in nodes]
        ax.bar(offsets, values, width, color=colors[mode], label=mode_label(mode))

    ax.set_xlabel("Nodes")
    ax.set_ylabel("Total runtime (s)")
    ax.set_xticks(x, [str(node) for node in nodes])
    ax.set_title(f"Total Runtime Scaling ({suffix})")
    ax.legend(frameon=True, loc="upper right")
    fig.savefig(plots_dir / f"scaling_total_{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def plot_speedup(rows: List[Dict[str, Union[float, int, str]]], plots_dir: Path, suffix: str) -> None:
    nodes = sorted({int(row["nodes"]) for row in rows})
    modes = sorted({str(row["mode"]) for row in rows}, key=mode_sort_key)
    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    colors = color_for_modes(modes)

    x, width = grouped_bar_layout(nodes, modes)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)

    for index, mode in enumerate(modes):
        mode_nodes = [node for node in nodes if (node, mode) in by]
        first_node = min(mode_nodes)
        first_total = float(by[(first_node, mode)]["total_s"])
        speedup = [
            first_total / float(by[(node, mode)]["total_s"]) if (node, mode) in by else np.nan
            for node in nodes
        ]
        efficiency = [
            speedup[node_index] / (node / first_node) if not np.isnan(speedup[node_index]) else np.nan
            for node_index, node in enumerate(nodes)
        ]
        offsets = x + (index - (len(modes) - 1) / 2) * width
        axes[0].bar(offsets, speedup, width, color=colors[mode], label=mode_label(mode))
        axes[1].bar(offsets, efficiency, width, color=colors[mode], label=mode_label(mode))

    axes[0].plot(x, nodes, linestyle="--", color="#666666", linewidth=1, label="Ideal")
    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Strong scaling speedup")
    axes[0].set_xticks(x, [str(node) for node in nodes])
    axes[0].set_title("Speedup vs 1-node point")

    axes[1].axhline(1.0, linestyle="--", color="#666666", linewidth=1)
    axes[1].set_xlabel("Nodes")
    axes[1].set_ylabel("Strong scaling efficiency")
    axes[1].set_xticks(x, [str(node) for node in nodes])
    axes[1].set_title("Efficiency vs ideal")
    axes[1].set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(f"Strong Scaling ({suffix})", y=1.06, fontsize=14)
    fig.savefig(plots_dir / f"scaling_speedup_{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def plot_component_grid(rows: List[Dict[str, Union[float, int, str]]], plots_dir: Path, suffix: str) -> None:
    nodes = sorted({int(row["nodes"]) for row in rows})
    modes = sorted({str(row["mode"]) for row in rows}, key=mode_sort_key)
    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    colors = color_for_modes(modes)
    components = [
        ("init_s", "Initialization"),
        ("computation_s", "Computation"),
        ("communication_s", "Communication"),
        ("io_s", "I/O Total"),
        ("io_write_s", "Parallel HDF5 Write"),
        ("io_read_s", "Parallel HDF5 Read"),
        ("validation_s", "Validation"),
        ("total_s", "Total"),
    ]

    components = [
        (field, title)
        for field, title in components
        if any(row.get(field) not in ("", None) for row in rows)
    ]

    ncols = 4
    nrows = int(np.ceil(len(components) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    x, width = grouped_bar_layout(nodes, modes)
    for ax, (field, title) in zip(axes.ravel(), components):
        for index, mode in enumerate(modes):
            offsets = x + (index - (len(modes) - 1) / 2) * width
            values = [float(by[(node, mode)][field]) if (node, mode) in by else np.nan for node in nodes]
            ax.bar(offsets, values, width, color=colors[mode], label=mode_label(mode))
        ax.set_title(title)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Seconds")
        ax.set_xticks(x, [str(node) for node in nodes])

    for ax in axes[len(components) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(f"Timing Components by Scaling Mode ({suffix})", y=1.04, fontsize=14)
    fig.savefig(plots_dir / f"scaling_components_{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def plot_baseline_delta(summary: List[Dict[str, Union[float, int, str]]], plots_dir: Path, suffix: str) -> None:
    rows = [row for row in summary if row["mode"] != "baseline" and row["speedup_vs_baseline"] != ""]
    if not rows:
        return

    nodes = sorted({int(row["nodes"]) for row in rows})
    modes = sorted({str(row["mode"]) for row in rows}, key=mode_sort_key)
    by = {(int(row["nodes"]), str(row["mode"])): row for row in rows}
    colors = color_for_modes(modes)
    x = np.arange(len(nodes))
    width = min(0.8 / max(len(modes), 1), 0.25)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    for index, mode in enumerate(modes):
        offsets = x + (index - (len(modes) - 1) / 2) * width
        values = [float(by[(node, mode)]["speedup_vs_baseline"]) if (node, mode) in by else np.nan for node in nodes]
        ax.bar(offsets, values, width, color=colors[mode], label=mode_label(mode))

    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xticks(x, [str(node) for node in nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Speedup vs baseline at same node count")
    ax.set_title(f"Fine-Tuning Mode Speedup vs Baseline ({suffix})")
    ax.legend(frameon=True, loc="upper right")
    fig.savefig(plots_dir / f"scaling_vs_baseline_{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def infer_suffix(log_path: Path) -> str:
    match = re.search(r"slurm-[A-Za-z-]*matmul-(\d+)\.out$", log_path.name)
    if match:
        return match.group(1)
    return log_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scaling-analysis CSVs and plots from a matmul Slurm log.")
    parser.add_argument("log", type=Path, help="Path to a slurm matmul stdout log")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output directory for CSV and PNG files. Defaults to the directory containing this script.",
    )
    parser.add_argument("--title", default=None, help="Optional output suffix/title. Defaults to the Slurm job id.")
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve() if args.plots_dir else Path(__file__).resolve().parent
    suffix = args.title or infer_suffix(log_path)

    rows = parse_log(log_path)
    validate_rows(rows)
    summary = build_summary(rows)
    write_csvs(rows, summary, plots_dir, suffix)

    configure_matplotlib()
    plot_total_scaling(rows, plots_dir, suffix)
    plot_speedup(rows, plots_dir, suffix)
    plot_component_grid(rows, plots_dir, suffix)
    plot_baseline_delta(summary, plots_dir, suffix)

    modes = ", ".join(mode_label(mode) for mode in sorted({str(row["mode"]) for row in rows}, key=mode_sort_key))
    print(f"Parsed {len(rows)} timing rows from {log_path}")
    print(f"Modes: {modes}")
    print(f"Wrote scaling CSVs and plots to {plots_dir}")


if __name__ == "__main__":
    main()
