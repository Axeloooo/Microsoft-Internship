#!/usr/bin/env python3
"""Plot latency benchmarks from CSV.

Reads a CSV file containing benchmark results and plots latency (µs) versus
message size (KiB) for different AllReduce implementations. The resulting
figure is saved to the specified output path.

Example
-------
./plot.sh -f benchmarks.csv
"""
from __future__ import annotations

import argparse
import pathlib
import datetime
from typing import List, Tuple, Set

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
        df: pd.DataFrame = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"CSV file not found: {csv_path}") from exc

    required_cols: Set[str] = {"implementation", "message_bytes", "latency_ms"}
    missing: Set[str] = required_cols.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {', '.join(missing)}")

    return df


def plot_latency(df: pd.DataFrame) -> None:
    """Plot latency benchmarks from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing benchmark results with columns:
            - 'implementation': Name of the AllReduce implementation.
            - 'message_bytes': Size of the message in bytes.
            - 'latency_ms': Latency in milliseconds.
    """
    # Implementations present in the CSV
    impls: List[str] = df["implementation"].unique().tolist()

    # Byte boundaries that delimit the three n-ranges
    smallest: int = int(df["message_bytes"].min())  # n = 1
    bound_1: int = smallest * 16  # n = 16
    bound_2: int = smallest * 128  # n = 128

    ranges: List[Tuple[int | None, int | None]] = [
        (None, bound_1),
        (bound_1, bound_2),
        (bound_2, None),
    ]
    titles: List[str] = ["n = 1…15", "n = 16…127", "n = 128…1023"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    for ax, (lo, hi), title in zip(axes, ranges, titles):
        # select rows inside the current range
        rng_mask: pd.Series = (df["message_bytes"] >= (lo or -float("inf"))) & (
            df["message_bytes"] < (hi or float("inf"))
        )
        rng_df: pd.DataFrame = df[rng_mask].copy()

        for idx, impl in enumerate(impls):
            sub: pd.DataFrame = (
                rng_df[rng_df["implementation"] == impl]
                .copy()
                .sort_values("message_bytes")
            )
            if sub.empty:
                continue
            sub["latency_us"] = sub["latency_ms"] * 1000
            ax.plot(
                sub["message_bytes"],
                sub["latency_us"],
                marker="o",
                label=impl,
            )

        ax.set_title(f"Latency vs Message Size ({title})")
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Latency (µs)")
        ax.ticklabel_format(style="plain", axis="x")
        ax.ticklabel_format(style="plain", axis="y")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()

    fig.tight_layout()

    # Create a unique filename
    date_str: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path: pathlib.Path = pathlib.Path(f"../images/latency_vs_size_{date_str}.png")
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")


def main() -> None:
    args: argparse.Namespace = parse_args()
    data: pd.DataFrame = load_data(args.csv)
    plot_latency(data)


if __name__ == "__main__":
    main()
