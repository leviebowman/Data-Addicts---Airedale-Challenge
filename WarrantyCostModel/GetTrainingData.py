"""
Script to generate training data by sampling subsets of the full dataset and aggregating them.
"""

import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data_Full.xlsx"
FIELD_DATA_PATH = (
    BASE_DIR.parent
    / "Data"
    / "Competition_Dataset1_Field_Service_Final.xlsx"
)
WARRANTY_DATA_PATH = (
    BASE_DIR.parent / "Data" / "Competition_Dataset2_Warranty_Final.xlsx"
)
OUTPUT_PATH = BASE_DIR / "TrainingData_Updated.csv"
CLUSTER_DATA_PATH = BASE_DIR / "ClusteredUnits.csv"
TARGET_COLUMN = "Warranty Cost"
KEEP_SOURCE_COLUMNS = {"Serial Number", "Date", "Product Family", TARGET_COLUMN}
CLUSTER_COLUMN = "Cluster"
CLUSTER_LABELS = (1, 2)
SAMPLE_SEED = 42

data = pd.read_excel(DATA_PATH)
cluster_assignments = pd.read_csv(CLUSTER_DATA_PATH)[
    ["Serial Number", CLUSTER_COLUMN]
]
if cluster_assignments["Serial Number"].isna().any() or cluster_assignments[
    "Serial Number"
].duplicated().any():
    raise ValueError("Cluster assignments must have a unique, non-missing Serial Number")
data = data.merge(
    cluster_assignments,
    on="Serial Number",
    how="left",
    validate="one_to_one",
)
if data[CLUSTER_COLUMN].isna().any():
    raise ValueError("Some source rows have no cluster assignment")

excluded_source_columns = (
    set(pd.read_excel(FIELD_DATA_PATH, nrows=0).columns)
    | set(pd.read_excel(WARRANTY_DATA_PATH, nrows=0).columns)
)
columns_to_remove = sorted(
    column
    for column in data.columns
    if column in excluded_source_columns and column not in KEEP_SOURCE_COLUMNS
)
data = data.drop(columns=columns_to_remove)
print(f"Removed Warranty and Field Service columns: {', '.join(columns_to_remove)}")


def getRandomSamplesByCluster(
    data: pd.DataFrame,
    n_per_cluster: int,
    group_size_range: list[int],
    cluster_labels: tuple[int, ...],
) -> list[pd.DataFrame]:
    if n_per_cluster <= 0:
        raise ValueError("n_per_cluster must be greater than zero")
    if len(group_size_range) != 2:
        raise ValueError("group_size_range must contain exactly two values")
    min_group_size, max_group_size = group_size_range
    if min_group_size <= 0 or min_group_size > max_group_size:
        raise ValueError("group_size_range must be positive and ascending")

    rng = random.Random(SAMPLE_SEED)
    samples: list[pd.DataFrame] = []
    for cluster_label in cluster_labels:
        cluster_data = data.loc[data[CLUSTER_COLUMN] == cluster_label]
        if cluster_data.empty:
            raise ValueError(f"No source rows found for cluster {cluster_label}")
        if max_group_size > len(cluster_data):
            raise ValueError(
                f"Maximum sample size {max_group_size} exceeds the "
                f"{len(cluster_data)} rows in cluster {cluster_label}"
            )

        samples.extend(
            cluster_data.sample(
                n=rng.randint(min_group_size, max_group_size),
                random_state=rng.randrange(2**32),
            )
            for _ in range(n_per_cluster)
        )
    rng.shuffle(samples)
    return samples


n = 1500
groupSizeRange = [10, 1000]
groupedDfs = getRandomSamplesByCluster(
    data,
    n_per_cluster=n,
    group_size_range=groupSizeRange,
    cluster_labels=CLUSTER_LABELS,
)


def getAggregateFromDfs(groupedDfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not groupedDfs:
        raise ValueError("groupedDfs must contain at least one DataFrame")

    aggregates: list[dict[str, int | float]] = []

    for frame in groupedDfs:
        cluster_labels = frame[CLUSTER_COLUMN].unique()
        if len(cluster_labels) != 1:
            raise ValueError("Each sampled subset must contain exactly one cluster")

        row: dict[str, int | float] = {}
        row[CLUSTER_COLUMN] = int(cluster_labels[0])
        for column in frame.select_dtypes(include="number").columns:
            if column == CLUSTER_COLUMN:
                continue
            if column == TARGET_COLUMN:
                row[f"{column} Mean"] = frame[column].mean()
                continue
            row[f"{column} Mean"] = frame[column].mean()
            row[f"{column} Standard Deviation"] = frame[column].std()

        for column in frame.select_dtypes(exclude="number").columns:
            if column in {"Serial Number", "Date", "Claim Date", CLUSTER_COLUMN}:
                continue
            for value, count in frame[column].value_counts(dropna=False).items():
                category = "(missing)" if pd.isna(value) else str(value)
                count_column = f"{column}={category} Count"
                row[count_column] = int(count)
        row["Rows in Subset"] = len(frame)
        aggregates.append(row)

    result = pd.DataFrame(aggregates)
    count_columns = [column for column in result.columns if column.endswith(" Count")]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    return result

aggregate = getAggregateFromDfs(groupedDfs)
aggregate.to_csv(OUTPUT_PATH, index=False)
print(f"Generated {len(aggregate)} samples: {n} per cluster.")