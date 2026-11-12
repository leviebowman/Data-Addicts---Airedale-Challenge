"""Plot warranty-cost distributions by cluster and field failure category."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "ClusteredUnits.csv"
OUTPUT_PATH = BASE_DIR / "WarrantyCostDistributionByCluster.png"
FAILURE_OUTPUT_PATH = BASE_DIR / "WarrantyCostDistributionByFailureType.png"

CLUSTER_COLUMN = "Cluster"
FAILURE_COLUMN = "Failure Category"
COST_COLUMN = "Warranty Cost"
CLUSTERS = (1, 2)
BIN_COUNT = 30


def create_plot(data: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    required_columns = {CLUSTER_COLUMN, COST_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Cluster data is missing columns: {sorted(missing_columns)}")

    data = data[[CLUSTER_COLUMN, COST_COLUMN]].copy()
    data[COST_COLUMN] = pd.to_numeric(data[COST_COLUMN], errors="raise")
    if data[[CLUSTER_COLUMN, COST_COLUMN]].isna().any().any():
        raise ValueError("Cluster labels and warranty costs must not be missing")
    if not np.isfinite(data[COST_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Warranty costs must be finite")
    if (data[COST_COLUMN] <= 0).any():
        raise ValueError("Logarithmic cost bins require positive warranty costs")

    observed_clusters = set(data[CLUSTER_COLUMN].unique())
    if not set(CLUSTERS).issubset(observed_clusters):
        raise ValueError("Clustered data must include clusters 1 and 2")

    minimum_cost = float(data[COST_COLUMN].min())
    maximum_cost = float(data[COST_COLUMN].max())
    bins = np.geomspace(minimum_cost, maximum_cost, BIN_COUNT + 1)
    plot_groups = [
        (f"Cluster {cluster}", data.loc[data[CLUSTER_COLUMN] == cluster, COST_COLUMN])
        for cluster in CLUSTERS
    ]
    plot_groups.append(("Clusters 1 and 2 Combined", data[COST_COLUMN]))

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for axis, (label, costs), color in zip(axes, plot_groups, colors):
        axis.hist(
            costs,
            bins=bins,
            weights=np.full(len(costs), 1 / len(costs)),
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        median = float(costs.median())
        axis.axvline(median, color="#333333", linestyle="--", linewidth=1.5)
        axis.set_xscale("log")
        axis.set_title(
            f"{label}\n{len(costs):,} units\n"
            f"Median: {median:,.0f} USD | Mean: {costs.mean():,.0f} USD",
            fontsize=10,
        )
        axis.set_xlabel("Warranty Cost ($, logarithmic scale)")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Share of units per bin")
    figure.suptitle("Warranty Cost Distribution by Cluster", fontsize=16)
    figure.text(
        0.5,
        0.01,
        "Shared logarithmic bins; dashed lines mark medians. "
        "Combined panel includes every unit from clusters 1 and 2.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_failure_type_plot(
    data: pd.DataFrame,
    output_path: Path = FAILURE_OUTPUT_PATH,
) -> None:
    required_columns = {FAILURE_COLUMN, COST_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Cluster data is missing columns: {sorted(missing_columns)}")

    data = data[[FAILURE_COLUMN, COST_COLUMN]].copy()
    data[COST_COLUMN] = pd.to_numeric(data[COST_COLUMN], errors="raise")
    if data[[FAILURE_COLUMN, COST_COLUMN]].isna().any().any():
        raise ValueError("Failure categories and warranty costs must not be missing")
    if not np.isfinite(data[COST_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Warranty costs must be finite")
    if (data[COST_COLUMN] <= 0).any():
        raise ValueError("Logarithmic cost bins require positive warranty costs")

    failure_types = sorted(data[FAILURE_COLUMN].unique())
    if not failure_types:
        raise ValueError("No failure types were found")

    bins = np.geomspace(
        float(data[COST_COLUMN].min()),
        float(data[COST_COLUMN].max()),
        BIN_COUNT + 1,
    )
    figure, axes = plt.subplots(
        1,
        len(failure_types),
        figsize=(6 * len(failure_types), 5),
        sharey=True,
        squeeze=False,
    )

    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")
    for axis, failure_type, color in zip(axes[0], failure_types, colors):
        costs = data.loc[data[FAILURE_COLUMN] == failure_type, COST_COLUMN]
        axis.hist(
            costs,
            bins=bins,
            weights=np.full(len(costs), 1 / len(costs)),
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        median = float(costs.median())
        axis.axvline(median, color="#333333", linestyle="--", linewidth=1.5)
        axis.set_xscale("log")
        axis.set_title(
            f"{failure_type}\n{len(costs):,} units\n"
            f"Median: {median:,.0f} USD | Mean: {costs.mean():,.0f} USD",
            fontsize=10,
        )
        axis.set_xlabel("Warranty Cost ($, logarithmic scale)")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Share of units per bin")
    figure.suptitle("Warranty Cost Distribution by Failure Type", fontsize=16)
    figure.text(
        0.5,
        0.01,
        "Shared logarithmic bins across failure types; dashed lines mark medians.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    data = pd.read_csv(INPUT_PATH)
    create_plot(data)
    create_failure_type_plot(data)
    print(f"Warranty-cost distribution plot saved to: {OUTPUT_PATH}")
    print(f"Failure-type warranty-cost distribution plot saved to: {FAILURE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
