#!/usr/bin/env python3
"""Plot latency benchmarks from CSV.

Reads a CSV file containing benchmark results and plots latency (ms) versus
message size (bytes) for different AllReduce implementations. The resulting
figure is saved to the specified output path.

Example
-------
python plot_benchmarks.py --csv data/benchmarks.csv --out images/latency_vs_size.png
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:  # noqa: D401
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
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("images/latency_vs_size.png"),
        help="Output path for the generated plot (PNG).",
    )
    return parser.parse_args()


def ensure_parent_dir(path: pathlib.Path) -> None:
    """Create the parent directory of *path* if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load benchmark data from a CSV file.

    Args:
        csv_path (pathlib.Path): Path to the CSV file containing benchmark results.

    Raises:
        SystemExit: If the CSV file is not found or required columns are missing.
        SystemExit: If required columns are missing in the CSV.

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


def plot_latency(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Plot latency vs. message size from the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing benchmark results with columns.
        out_path (pathlib.Path): Path to save the output plot image.
    """
    impls = df["implementation"].unique()
    fig, ax = plt.subplots(figsize=(8, 5))

    for impl in impls:
        subset = df[df["implementation"] == impl]
        # Ensure data are sorted for nicer lines.
        subset = subset.sort_values("message_bytes")
        ax.plot(
            subset["message_bytes"],
            subset["latency_ms"],
            marker="o",
            label=impl,
        )

    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("AllReduce Latency vs. Message Size")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")


def main() -> None:
    args = parse_args()
    data = load_data(args.csv)
    plot_latency(data, args.out)


if __name__ == "__main__":
    main()
