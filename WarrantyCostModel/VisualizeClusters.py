"""Create a beginner-friendly visual summary of the unit clusters."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_PATH = BASE_DIR / "ClusteredUnits.csv"
PROFILES_PATH = BASE_DIR / "ClusterProfiles.csv"
OUTPUT_PATH = BASE_DIR / "ClusterOverview.png"

ID_COLUMN = "Serial Number"
FAILURE_COLUMN = "Failure Category"
COST_COLUMN = "Warranty Cost"
OUTCOME_COLUMNS = {ID_COLUMN, FAILURE_COLUMN, COST_COLUMN, "Cluster"}
PROFILE_FEATURES = [
    "Product Family",
    "Coil Supplier",
    "Facility",
    "Controls Revision",
    "Defect Category",
    "Repair Type",
    "Test Station",
    "Failure Mode",
    "Shift",
]


def get_distinguishing_features(
    assignments: pd.DataFrame,
    clusters: list[int],
) -> pd.DataFrame:
    differences: list[dict[str, str | float]] = []
    for feature in PROFILE_FEATURES:
        if feature not in assignments.columns:
            continue

        proportions = pd.crosstab(
            assignments["Cluster"],
            assignments[feature],
            normalize="index",
        )
        if len(proportions.index) < 2:
            continue

        for category in proportions.columns:
            values = proportions[category].reindex(clusters, fill_value=0)
            differences.append(
                {
                    "feature": feature,
                    "category": str(category),
                    "difference": float(values.max() - values.min()),
                }
            )

    return pd.DataFrame(differences).sort_values(
        "difference", ascending=False
    ).head(6)


def create_cluster_visualization(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    required_assignment_columns = {
        ID_COLUMN,
        FAILURE_COLUMN,
        COST_COLUMN,
        "Cluster",
    }
    missing = required_assignment_columns - set(assignments.columns)
    if missing:
        raise ValueError(f"Cluster assignments are missing columns: {sorted(missing)}")
    if "Cluster" not in profiles.columns or "Units" not in profiles.columns:
        raise ValueError("Cluster profiles must include 'Cluster' and 'Units'")

    clusters = sorted(assignments["Cluster"].dropna().unique().tolist())
    if not clusters:
        raise ValueError("No cluster assignments were found")

    colors = plt.get_cmap("Set2").colors
    cluster_colors = [colors[index % len(colors)] for index in range(len(clusters))]
    labels = [f"Cluster {cluster}" for cluster in clusters]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle(
        "Who is in each cluster, and what happened to them?",
        fontsize=18,
        fontweight="bold",
    )

    counts = assignments["Cluster"].value_counts().reindex(clusters, fill_value=0)
    axes[0, 0].bar(labels, counts.values, color=cluster_colors)
    axes[0, 0].set_title("1. Number of units")
    axes[0, 0].set_ylabel("Units")
    axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%d")

    total_cost = assignments.groupby("Cluster")[COST_COLUMN].sum().reindex(clusters)
    axes[0, 1].bar(labels, total_cost.values, color=cluster_colors)
    axes[0, 1].set_title("2. Observed total warranty cost")
    axes[0, 1].set_ylabel("Warranty cost ($)")
    axes[0, 1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"${value / 1_000_000:.1f}M")
    )
    axes[0, 1].bar_label(
        axes[0, 1].containers[0],
        labels=[f"${value:,.0f}" for value in total_cost.values],
        padding=3,
    )

    failure_mix = pd.crosstab(
        assignments["Cluster"],
        assignments[FAILURE_COLUMN],
        normalize="index",
    ).reindex(clusters, fill_value=0)
    bottom = pd.Series(0.0, index=clusters)
    for failure_type in failure_mix.columns:
        values = failure_mix[failure_type]
        axes[1, 0].bar(
            labels,
            values.values,
            bottom=bottom.values,
            label=failure_type,
        )
        bottom += values
    axes[1, 0].set_title("3. Failure mix within each cluster")
    axes[1, 0].set_ylabel("Share of cluster")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    axes[1, 0].legend(title="Observed failure type")

    distinguishing = get_distinguishing_features(assignments, clusters)
    axis = axes[1, 1]
    if distinguishing.empty:
        axis.text(0.5, 0.5, "No categorical profile features available", ha="center")
        axis.set_axis_off()
    else:
        feature_labels = (
            distinguishing["feature"] + "\n" + distinguishing["category"]
        ).tolist()
        positions = range(len(distinguishing))
        width = 0.8 / len(clusters)
        for index, (cluster, color) in enumerate(zip(clusters, cluster_colors)):
            values = []
            for row in distinguishing.itertuples(index=False):
                share = (
                    assignments.loc[
                        assignments["Cluster"] == cluster, row.feature
                    ]
                    .eq(row.category)
                    .mean()
                )
                values.append(share)
            offsets = [
                position - 0.4 + width * (index + 0.5)
                for position in positions
            ]
            axis.bar(offsets, values, width=width, label=f"Cluster {cluster}", color=color)

        axis.set_xticks(list(positions), feature_labels, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )
        axis.set_title("4. Most distinctive pre-failure traits")
        axis.set_ylabel("Share of cluster with trait")
        axis.legend()

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    figure.text(
        0.5,
        0.015,
        "Clusters were formed from configuration, manufacturing, and test features. "
        "Failure type and warranty cost are shown only afterward; these are observed "
        "outcomes, not estimated investment savings.",
        ha="center",
        fontsize=9,
        wrap=True,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    profiles = pd.read_csv(PROFILES_PATH)
    create_cluster_visualization(assignments, profiles)
    print(f"Cluster overview saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
