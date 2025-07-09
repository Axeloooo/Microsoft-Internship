#!/usr/bin/env python3
"""Plot latency benchmarks from CSV.

Reads a CSV file containing benchmark results and plots latency (µs) versus
message size (KiB) for different AllReduce implementations. The resulting
figure is saved to the specified output path.

Example
-------
python plot_benchmark.py --csv data/benchmarks.csv
"""
from __future__ import annotations

import argparse
import pathlib
import datetime

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        argparse.Namespace: Namespace containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot AllReduce latency benchmarks.")
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("data/benchmarks.csv"),
        help="Path to the input CSV file.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: pathlib.Path) -> None:
    """Create the parent directory of *path* if it does not already exist.

    Args:
        path (pathlib.Path): Path to the file for which to ensure the parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load benchmark data from a CSV file.

    Args:
        csv_path (pathlib.Path): Path to the CSV file containing benchmark results.

    Raises:
        SystemExit: If the CSV file is not found or required columns are missing.
        SystemExit: If the CSV file does not contain the required columns.

    Returns:
        pd.DataFrame: DataFrame containing the benchmark results with columns.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"CSV file not found: {csv_path}") from exc

    required_cols = {"implementation", "message_bytes", "latency_ms"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {', '.join(missing)}")

    return df


def plot_latency(df: pd.DataFrame) -> None:
    """Plot latency vs. message size from the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing benchmark results with columns.
    """
    impls = df["implementation"].unique()
    fig, ax = plt.subplots(figsize=(8, 5))

    for impl in impls:
        subset = df[df["implementation"] == impl].copy()
        subset = subset.sort_values("message_bytes")

        subset["latency_us"] = subset["latency_ms"] * 1000

        ax.plot(
            subset["message_bytes"],
            subset["latency_us"],
            marker="o",
            label=impl,
        )

    # Axis labels and title
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_ylabel("Latency (µs)")
    ax.set_title("AllReduce Latency vs. Message Size")

    # Disable scientific notation for clarity
    ax.ticklabel_format(style="plain", axis="x")
    ax.ticklabel_format(style="plain", axis="y")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    # Create a unique filename
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"latency_vs_size_{date_str}.png"
    out_path = pathlib.Path(f"../images/{filename}")
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=300)

    print(f"Plot saved to {out_path}")


def main() -> None:
    args = parse_args()
    data = load_data(args.csv)
    plot_latency(data)


if __name__ == "__main__":
    main()
