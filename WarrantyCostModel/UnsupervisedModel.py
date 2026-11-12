"""Cluster units using pre-field-failure information and profile outcomes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "Data_Full.xlsx"
ASSIGNMENTS_PATH = BASE_DIR / "ClusteredUnits.csv"
PROFILES_PATH = BASE_DIR / "ClusterProfiles.csv"

ID_COLUMN = "Serial Number"
FAILURE_COLUMN = "Failure Category"
COST_COLUMN = "Warranty Cost"

# Only configuration, manufacturing, and end-of-line information goes into clustering.
CATEGORICAL_FEATURES = [
    "Customer Region",
    "Product Family",
    "Facility",
    "Controls Revision",
    "Coil Supplier",
    "Shift",
    "Defect Category",
    "Repair Type",
    "Disposition Code",
    "Test Station",
    "Failure Mode",
]
NUMERIC_FEATURES = ["Labor Hours", "Material Cost", "Rework Hours"]

MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
SILHOUETTE_SAMPLE_SIZE = 1000
RANDOM_STATE = 42


def build_feature_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessing)])


def build_cluster_profiles(
    clustered_data: pd.DataFrame,
) -> pd.DataFrame:
    cluster_summary = clustered_data.groupby("Cluster").agg(
        Units=(ID_COLUMN, "size"),
        Total_Warranty_Cost=(COST_COLUMN, "sum"),
        Mean_Warranty_Cost=(COST_COLUMN, "mean"),
        Median_Warranty_Cost=(COST_COLUMN, "median"),
    )

    failure_counts = pd.crosstab(
        clustered_data["Cluster"], clustered_data[FAILURE_COLUMN]
    ).add_prefix("Failure Count - ")
    failure_shares = pd.crosstab(
        clustered_data["Cluster"],
        clustered_data[FAILURE_COLUMN],
        normalize="index",
    ).add_prefix("Failure Share - ")
    failure_costs = (
        clustered_data.pivot_table(
            index="Cluster",
            columns=FAILURE_COLUMN,
            values=COST_COLUMN,
            aggfunc="sum",
            fill_value=0,
        )
        .add_prefix("Warranty Cost - ")
    )

    profiles = cluster_summary.join(failure_counts).join(failure_shares).join(
        failure_costs
    )
    return profiles.reset_index()


def cluster_and_profile(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    required_columns = {
        ID_COLUMN,
        FAILURE_COLUMN,
        COST_COLUMN,
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Input data must contain at least one unit")
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be present and unique for each unit")
    if data[[FAILURE_COLUMN, COST_COLUMN]].isna().any().any():
        raise ValueError(f"{FAILURE_COLUMN} and {COST_COLUMN} cannot be missing")

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    features = data[feature_columns].copy()
    if features.isna().any().any():
        raise ValueError("Clustering features contain missing values")
    if not np.isfinite(features[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Numeric clustering features must be finite")

    preprocessor = build_feature_pipeline()
    encoded_features = preprocessor.fit_transform(features)
    max_clusters = min(MAX_CLUSTERS, len(data) - 1)
    if max_clusters < MIN_CLUSTERS:
        raise ValueError("At least three units are required to select a cluster count")

    candidates: list[tuple[int, float]] = []
    for cluster_count in range(MIN_CLUSTERS, max_clusters + 1):
        candidate_model = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            random_state=RANDOM_STATE,
        )
        labels = candidate_model.fit_predict(encoded_features)
        score = float(
            silhouette_score(
                encoded_features,
                labels,
                sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(data)),
                random_state=RANDOM_STATE,
            )
        )
        candidates.append((cluster_count, score))

    best_cluster_count, best_silhouette = max(candidates, key=lambda item: item[1])
    model = KMeans(
        n_clusters=best_cluster_count,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    clustered_data = data.copy()
    clustered_data["Cluster"] = model.fit_predict(encoded_features) + 1

    profiles = build_cluster_profiles(clustered_data)
    return clustered_data, profiles, best_cluster_count, best_silhouette


def main() -> None:
    source_data = pd.read_excel(INPUT_PATH, sheet_name="Sheet1", engine="openpyxl")
    clustered_data, profiles, cluster_count, silhouette = cluster_and_profile(
        source_data
    )
    clustered_data.to_csv(ASSIGNMENTS_PATH, index=False)
    profiles.to_csv(PROFILES_PATH, index=False)

    print(f"Selected clusters: {cluster_count}")
    print(f"Sampled silhouette score: {silhouette:.3f}")
    print(f"Unit assignments saved to: {ASSIGNMENTS_PATH}")
    print(f"Cluster outcome profiles saved to: {PROFILES_PATH}")
    print(
        "Profiles describe observed failures and costs; clusters do not estimate "
        "the causal effect of investments."
    )


if __name__ == "__main__":
    main()