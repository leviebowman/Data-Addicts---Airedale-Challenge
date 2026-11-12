from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "TrainingData_Aggregated_Wide.csv"
TARGET_COLUMN = "Warranty Cost Mean"

# Set to selected CSV column names, or None to use all columns except the target.
FEATURE_COLUMNS: list[str] | None = None
FEATURE_COUNTS: tuple[int | str, ...] = (5, 10, 20, 40, 80, "all")
K_VALUES = (3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200)
WEIGHT_OPTIONS = ("uniform", "distance")
MINKOWSKI_P_VALUES = (1, 2)
RANDOM_STATE = 42


def train_and_evaluate(
    training_data: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float | int | str]]:
    if TARGET_COLUMN not in training_data.columns:
        raise ValueError(f"Training data must include {TARGET_COLUMN!r}")

    feature_columns = (
        FEATURE_COLUMNS
        if FEATURE_COLUMNS is not None
        else [column for column in training_data.columns if column != TARGET_COLUMN]
    )
    if not feature_columns:
        raise ValueError("Select at least one feature column")

    missing_columns = set(feature_columns + [TARGET_COLUMN]) - set(training_data.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")

    features = training_data[feature_columns]
    target = training_data[TARGET_COLUMN]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All selected feature columns must be numeric")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Selected features and target must not contain missing values")

    train_validation_features, test_features, train_validation_target, test_target = (
        train_test_split(
            features,
            target,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )
    )
    train_features, validation_features, train_target, validation_target = (
        train_test_split(
            train_validation_features,
            train_validation_target,
            test_size=1 / 3,
            random_state=RANDOM_STATE,
        )
    )

    feature_counts = [
        count
        for count in FEATURE_COUNTS
        if count == "all" or count <= len(feature_columns)
    ]
    candidate_ks = [k for k in K_VALUES if k <= len(train_features)]
    if not candidate_ks:
        raise ValueError("No valid k values for the training set")

    best_model: Pipeline | None = None
    best_k = 0
    best_feature_count: int | str = "all"
    best_weights = "uniform"
    best_p = 2
    best_validation_rmse = float("inf")
    for feature_count in feature_counts:
        for k in candidate_ks:
            for weights in WEIGHT_OPTIONS:
                for p in MINKOWSKI_P_VALUES:
                    model = Pipeline(
                        [
                            ("select", SelectKBest(f_regression, k=feature_count)),
                            ("scale", StandardScaler()),
                            (
                                "knn",
                                KNeighborsRegressor(
                                    n_neighbors=k,
                                    weights=weights,
                                    p=p,
                                ),
                            ),
                        ]
                    )
                    model.fit(train_features, train_target)
                    validation_predictions = model.predict(validation_features)
                    validation_rmse = float(
                        np.sqrt(
                            mean_squared_error(
                                validation_target, validation_predictions
                            )
                        )
                    )
                    if validation_rmse < best_validation_rmse:
                        best_model = model
                        best_k = k
                        best_feature_count = feature_count
                        best_weights = weights
                        best_p = p
                        best_validation_rmse = validation_rmse

    if best_model is None:
        raise RuntimeError("Unable to train a kNN model")

    test_predictions = best_model.predict(test_features)
    metrics: dict[str, float | int | str] = {
        "k": best_k,
        "feature_count": (
            len(feature_columns) if best_feature_count == "all" else best_feature_count
        ),
        "weights": best_weights,
        "p": best_p,
        "validation_rmse": best_validation_rmse,
        "test_rmse": float(np.sqrt(mean_squared_error(test_target, test_predictions))),
        "test_mae": float(mean_absolute_error(test_target, test_predictions)),
        "test_r2": float(r2_score(test_target, test_predictions)),
    }
    return best_model, metrics


def main() -> None:
    training_data = pd.read_csv(DATA_PATH)
    _, metrics = train_and_evaluate(training_data)
    print(f"Features used: {metrics['feature_count']}")
    print(f"Best k: {metrics['k']}")
    print(f"Distance weights: {metrics['weights']}")
    print(f"Minkowski p: {metrics['p']}")
    print(f"Validation RMSE: {metrics['validation_rmse']:.2f}")
    print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"Test MAE: {metrics['test_mae']:.2f}")
    print(f"Test R²: {metrics['test_r2']:.3f}")


if __name__ == "__main__":
    main()