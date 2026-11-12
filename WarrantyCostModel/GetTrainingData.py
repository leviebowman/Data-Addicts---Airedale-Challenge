"""
Generate a wide aggregated dataset from Data_Full.xlsx.

No splitting or model training.
Repair-related and post-failure fields are excluded.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "Data_Full.xlsx"
OUTPUT_PATH = BASE_DIR / "TrainingData_Aggregated_Wide.csv"

N_SAMPLES = 3000
GROUP_SIZE_RANGE = (10, 1000)
SEED = 123

# 1 = category proportions only
# 2 = category proportions plus all pairwise joint proportions
# 3 = also include all three-way joint proportions
MAX_INTERACTION_ORDER = 2

# These are the actual source column names.
PRODUCTION_COLUMNS = [
    "Product Family",
    "Customer Region",
    "Facility",
    "Controls Revision",
    "Coil Supplier",
    "Shift",
]

ID_COLUMN = "Serial Number"
SOURCE_TARGET = "Warranty Cost"
OUTPUT_TARGET = "Warranty Cost Mean"


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = data.columns.str.strip()

    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    required = [ID_COLUMN, SOURCE_TARGET] + PRODUCTION_COLUMNS
    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Explicit allowlist prevents accidental inclusion of repair costs,
    # warranty summaries, or post-failure fields.
    data = data[required].copy()

    for column in [ID_COLUMN] + PRODUCTION_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

        invalid = data[column].isna() | data[column].eq("")
        if invalid.any():
            raise ValueError(
                f"{column!r} contains missing or empty values."
            )

    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Expected one record per Serial Number.")

    data[SOURCE_TARGET] = pd.to_numeric(
        data[SOURCE_TARGET], errors="raise"
    )

    costs = data[SOURCE_TARGET].to_numpy(dtype=float)

    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError(
            "Warranty Cost must contain finite, nonnegative values."
        )

    return data.reset_index(drop=True)


def build_indicators(
    data: pd.DataFrame,
    max_order: int,
) -> tuple[list[str], np.ndarray]:
    """
    Build unit-level indicators for categories and joint categories.

    Averaging each indicator over a sample produces its proportion.
    """
    if not 1 <= max_order <= 3:
        raise ValueError("MAX_INTERACTION_ORDER must be 1, 2, or 3.")

    feature_names = []
    indicator_arrays = []

    # Integer codes make category combinations easy to identify.
    codes = {}
    labels = {}

    for column in PRODUCTION_COLUMNS:
        categories = sorted(data[column].unique().tolist())
        categorical = pd.Categorical(
            data[column], categories=categories
        )

        codes[column] = categorical.codes
        labels[column] = categories

    for order in range(1, max_order + 1):
        for columns in combinations(PRODUCTION_COLUMNS, order):
            joint_codes = np.column_stack([
                codes[column] for column in columns
            ])

            # Include combinations observed in the original dataset.
            observed_combinations = np.unique(joint_codes, axis=0)

            for category_codes in observed_combinations:
                indicator = np.all(
                    joint_codes == category_codes,
                    axis=1,
                )

                name_parts = [
                    f"{column}={labels[column][int(code)]}"
                    for column, code in zip(columns, category_codes)
                ]

                feature_names.append(
                    " AND ".join(name_parts) + " Proportion"
                )
                indicator_arrays.append(indicator)

    indicators = np.column_stack(indicator_arrays)
    return feature_names, indicators


def generate_dataset(
    data: pd.DataFrame,
    n_samples: int,
    group_size_range: tuple[int, int],
    seed: int,
    max_order: int,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if len(group_size_range) != 2:
        raise ValueError("Provide a minimum and maximum group size.")

    min_size, max_size = group_size_range

    if not 1 <= min_size <= max_size <= len(data):
        raise ValueError(
            "Group sizes must satisfy "
            "1 <= minimum <= maximum <= number of source units."
        )

    rng = np.random.default_rng(seed)

    feature_names, indicators = build_indicators(data, max_order)
    costs = data[SOURCE_TARGET].to_numpy(dtype=float)

    values = np.empty(
        (n_samples, len(feature_names)),
        dtype=float,
    )
    sizes = np.empty(n_samples, dtype=int)
    target_means = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        size = int(rng.integers(min_size, max_size + 1))

        selected = rng.choice(
            len(data),
            size=size,
            replace=False,
        )

        values[i] = indicators[selected].mean(axis=0)
        sizes[i] = size
        target_means[i] = costs[selected].mean()

    result = pd.DataFrame(values, columns=feature_names)

    # Remove features that do not vary across generated samples.
    constant_columns = result.columns[
        result.nunique(dropna=False) <= 1
    ]
    result = result.drop(columns=constant_columns)

    # Remove exactly identical features so the same information is not
    # automatically counted multiple times in distance calculations.
    duplicate_columns = result.columns[result.T.duplicated()]
    result = result.drop(columns=duplicate_columns)

    result["Rows in Subset"] = sizes
    result[OUTPUT_TARGET] = target_means

    print(f"Candidate proportion features: {len(feature_names):,}")
    print(f"Constant features removed: {len(constant_columns):,}")
    print(f"Identical features removed: {len(duplicate_columns):,}")

    return result


def main() -> None:
    data = prepare_data(
        pd.read_excel(
            INPUT_PATH,
            sheet_name="Sheet1",
            engine="openpyxl",
        )
    )

    dataset = generate_dataset(
        data=data,
        n_samples=N_SAMPLES,
        group_size_range=GROUP_SIZE_RANGE,
        seed=SEED,
        max_order=MAX_INTERACTION_ORDER,
    )

    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Source units: {len(data):,}")
    print(f"Aggregated samples: {len(dataset):,}")
    print(f"Proportion features retained: {dataset.shape[1] - 2:,}")
    print(f"Target: {OUTPUT_TARGET}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()