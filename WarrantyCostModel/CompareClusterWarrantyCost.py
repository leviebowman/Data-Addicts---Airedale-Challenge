"""Compare kNN-predicted and actual warranty-cost totals by full cluster."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SupervisedModel import DATA_PATH as TRAINING_DATA_PATH
from SupervisedModel import TARGET_COLUMN as AGGREGATE_TARGET
from SupervisedModel import train_and_evaluate


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATA_PATH = BASE_DIR / "Data_Full.xlsx"
CLUSTER_DATA_PATH = BASE_DIR / "ClusteredUnits.csv"
FIELD_DATA_PATH = (
    BASE_DIR.parent / "Data" / "Competition_Dataset1_Field_Service_Final.xlsx"
)
WARRANTY_DATA_PATH = (
    BASE_DIR.parent / "Data" / "Competition_Dataset2_Warranty_Final.xlsx"
)
COMPARISON_PATH = BASE_DIR / "ClusterWarrantyCostComparison.csv"
PLOT_PATH = BASE_DIR / "ClusterWarrantyCostComparison.png"

ID_COLUMN = "Serial Number"
SOURCE_TARGET = "Warranty Cost"
CLUSTER_COLUMN = "Cluster"
ALLOWED_SOURCE_COLUMNS = {ID_COLUMN, "Date", "Product Family", SOURCE_TARGET}


def _full_cluster_aggregates(
    source_data: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if source_data[ID_COLUMN].isna().any() or source_data[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be present and unique in source data")
    if (
        cluster_assignments[ID_COLUMN].isna().any()
        or cluster_assignments[ID_COLUMN].duplicated().any()
    ):
        raise ValueError(f"{ID_COLUMN} must be present and unique in cluster assignments")

    data = source_data.merge(
        cluster_assignments[[ID_COLUMN, CLUSTER_COLUMN]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if data[CLUSTER_COLUMN].isna().any():
        raise ValueError("Some source units are missing cluster assignments")
    if data[SOURCE_TARGET].isna().any():
        raise ValueError(f"{SOURCE_TARGET} cannot be missing")

    source_columns = (
        set(pd.read_excel(FIELD_DATA_PATH, nrows=0).columns)
        | set(pd.read_excel(WARRANTY_DATA_PATH, nrows=0).columns)
    )
    columns_to_drop = [
        column
        for column in data.columns
        if column in source_columns and column not in ALLOWED_SOURCE_COLUMNS
    ]
    data = data.drop(columns=columns_to_drop)

    aggregate_rows: list[dict[str, int | float]] = []
    actual_costs = data.groupby(CLUSTER_COLUMN)[SOURCE_TARGET].sum()
    for cluster_label, frame in data.groupby(CLUSTER_COLUMN, sort=True):
        row: dict[str, int | float] = {
            CLUSTER_COLUMN: int(cluster_label),
            AGGREGATE_TARGET: float(frame[SOURCE_TARGET].mean()),
            "Rows in Subset": len(frame),
        }
        for column in frame.select_dtypes(include="number").columns:
            if column in {CLUSTER_COLUMN, SOURCE_TARGET}:
                continue
            row[f"{column} Mean"] = float(frame[column].mean())
            row[f"{column} Standard Deviation"] = float(frame[column].std())

        for column in frame.select_dtypes(exclude="number").columns:
            if column in {ID_COLUMN, "Date", "Claim Date", CLUSTER_COLUMN}:
                continue
            for value, count in frame[column].value_counts(dropna=False).items():
                category = "(missing)" if pd.isna(value) else str(value)
                row[f"{column}={category} Count"] = int(count)
        aggregate_rows.append(row)

    return pd.DataFrame(aggregate_rows), actual_costs


def _plot_comparison(comparison: pd.DataFrame) -> None:
    labels = comparison["Group"].tolist()
    positions = np.arange(len(labels))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(10, 6))
    actual_bars = axis.bar(
        positions - bar_width / 2,
        comparison["Actual Warranty Cost"],
        bar_width,
        label="Actual",
    )
    predicted_bars = axis.bar(
        positions + bar_width / 2,
        comparison["Predicted Warranty Cost"],
        bar_width,
        label="kNN predicted",
    )
    axis.set_title("Actual vs. kNN-Predicted Warranty Cost")
    axis.set_ylabel("Warranty cost ($)")
    axis.set_xticks(positions, labels)
    axis.ticklabel_format(axis="y", style="plain")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(actual_bars, fmt="${:,.0f}", padding=3, fontsize=8)
    axis.bar_label(predicted_bars, fmt="${:,.0f}", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    training_data = pd.read_csv(TRAINING_DATA_PATH)
    model, _ = train_and_evaluate(training_data)
    source_data = pd.read_excel(SOURCE_DATA_PATH)
    cluster_assignments = pd.read_csv(CLUSTER_DATA_PATH)
    cluster_aggregates, actual_costs = _full_cluster_aggregates(
        source_data,
        cluster_assignments,
    )

    feature_columns = [
        column for column in training_data.columns if column != AGGREGATE_TARGET
    ]
    model_features = cluster_aggregates.reindex(
        columns=feature_columns,
        fill_value=0,
    )
    count_columns = [
        column for column in feature_columns if column.endswith(" Count")
    ]
    model_features[count_columns] = model_features[count_columns].fillna(0)
    if model_features.isna().any().any():
        missing_features = model_features.columns[
            model_features.isna().any()
        ].tolist()
        raise ValueError(f"Full-cluster features contain missing values: {missing_features}")
    predicted_means = model.predict(model_features)

    comparison = pd.DataFrame(
        {
            CLUSTER_COLUMN: cluster_aggregates[CLUSTER_COLUMN].astype(int),
            "Units": cluster_aggregates["Rows in Subset"].astype(int),
            "Actual Warranty Cost": cluster_aggregates[AGGREGATE_TARGET].astype(float)
            * cluster_aggregates["Rows in Subset"].astype(int),
            "Predicted Mean Warranty Cost": predicted_means.astype(float),
            "Predicted Warranty Cost": predicted_means
            * cluster_aggregates["Rows in Subset"].to_numpy(dtype=float),
        }
    )
    comparison["Group"] = "Cluster " + comparison[CLUSTER_COLUMN].astype(str)
    comparison["Absolute Error"] = (
        comparison["Predicted Warranty Cost"] - comparison["Actual Warranty Cost"]
    ).abs()

    total_actual = float(actual_costs.sum())
    total_predicted = float(comparison["Predicted Warranty Cost"].sum())
    total = pd.DataFrame(
        [
            {
                CLUSTER_COLUMN: "Total",
                "Units": int(comparison["Units"].sum()),
                "Actual Warranty Cost": total_actual,
                "Predicted Warranty Cost": total_predicted,
                "Group": "All Clusters",
                "Absolute Error": abs(total_predicted - total_actual),
            }
        ]
    )
    comparison_output = pd.concat([comparison, total], ignore_index=True)
    comparison_output.to_csv(COMPARISON_PATH, index=False)
    _plot_comparison(comparison_output)

    print("Predicted and actual warranty cost by full cluster:")
    for _, row in comparison.iterrows():
        print(
            f"{row['Group']}: {row['Units']:,} units | "
            f"actual ${row['Actual Warranty Cost']:,.2f} | "
            f"predicted ${row['Predicted Warranty Cost']:,.2f}"
        )
    total_error = total_predicted - total_actual
    total_error_percent = 100 * total_error / total_actual if total_actual else float("nan")
    print(
        f"Total: actual ${total_actual:,.2f} | "
        f"predicted ${total_predicted:,.2f} | "
        f"error ${total_error:,.2f} ({total_error_percent:.2f}%)"
    )
    print(f"Comparison saved to: {COMPARISON_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")
    print(
        "Note: source units are reused in the training aggregates, so this is not "
        "an independent holdout evaluation."
    )


if __name__ == "__main__":
    main()
