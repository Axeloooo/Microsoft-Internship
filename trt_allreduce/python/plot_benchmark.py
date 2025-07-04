import sys
import pandas as pd
import matplotlib.pyplot as plt


def _plot(df: pd.DataFrame, metric: str, ylabel: str) -> None:
    """Plot the specified metric from the benchmark results.

    Args:
        df (pd.DataFrame): DataFrame containing benchmark results.
        metric (str): The metric to plot (e.g., "latency_ms" or "bandwidth_TBps").
        ylabel (str): Label for the y-axis of the plot.
    """
    impls = df["implementation"].unique()
    for impl in impls:
        sub = df[df["implementation"] == impl]
        plt.figure()
        plt.plot(sub["message_bytes"] / 1024, sub[metric], marker="o", label=impl)
        plt.title(f"{impl} – {ylabel} vs. message size")
        plt.xlabel("Message size [KiB]")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()


def main(path: str) -> None:
    """Main function to read benchmark results and plot latency and bandwidth.

    Args:
        path (str): Path to the CSV file containing benchmark results.
    """
    df = pd.read_csv(path)
    _plot(df, "latency_ms", "Latency [ms]")
    _plot(df, "bandwidth_TBps", "Bandwidth [TB/s]")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_benchmark.py benchmark_results.csv")
        sys.exit(1)
    main(sys.argv[1])
