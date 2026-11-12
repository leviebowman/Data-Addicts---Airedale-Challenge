from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "TrainingData_Updated.csv"
TARGET_COLUMN = "Warranty Cost Mean"

# Set to selected CSV column names, or None to use all columns except the target.
FEATURE_COLUMNS: list[str] | None = None
K_VALUES = (3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200)
WEIGHT_OPTIONS = ("uniform", "distance")
MINKOWSKI_P_VALUES = (1, 2)
TEST_SIZE = 0.30
CV_FOLDS = 5
RANDOM_STATE = 42


def _make_model(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", MinMaxScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        )

    return Pipeline(
        [
            (
                "preprocessing",
                ColumnTransformer(transformers=transformers, remainder="drop"),
            ),
            ("knn", KNeighborsRegressor()),
        ]
    )


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

    features = training_data[feature_columns].copy()
    target = pd.to_numeric(training_data[TARGET_COLUMN], errors="raise")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Selected features and target must not contain missing values")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target values must be finite")

    # Cluster labels are numeric IDs but represent categories, not magnitudes.
    categorical_columns = [
        column
        for column in feature_columns
        if (
            not pd.api.types.is_numeric_dtype(features[column].dtype)
            or pd.api.types.is_bool_dtype(features[column].dtype)
            or column == "Cluster"
        )
    ]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Numeric feature {column!r} must contain finite values")
        features[column] = values

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    candidate_ks = [k for k in K_VALUES if k <= len(train_features)]
    if not candidate_ks:
        raise ValueError("No valid k values for the training set")

    cross_validation = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        estimator=_make_model(numeric_columns, categorical_columns),
        param_grid={
            "knn__n_neighbors": candidate_ks,
            "knn__weights": WEIGHT_OPTIONS,
            "knn__p": MINKOWSKI_P_VALUES,
        },
        scoring="neg_root_mean_squared_error",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_features, train_target)

    test_predictions = search.predict(test_features)
    metrics: dict[str, float | int | str] = {
        "k": int(search.best_params_["knn__n_neighbors"]),
        "weights": str(search.best_params_["knn__weights"]),
        "p": int(search.best_params_["knn__p"]),
        "cv_rmse": float(-search.best_score_),
        "test_rmse": float(np.sqrt(mean_squared_error(test_target, test_predictions))),
        "test_mae": float(mean_absolute_error(test_target, test_predictions)),
        "test_r2": float(r2_score(test_target, test_predictions)),
        "training_rows": len(train_features),
        "test_rows": len(test_features),
    }
    return search.best_estimator_, metrics


def main() -> None:
    training_data = pd.read_csv(DATA_PATH)
    _, metrics = train_and_evaluate(training_data)
    print(f"Training rows: {metrics['training_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Best k: {metrics['k']}")
    print(f"Distance weights: {metrics['weights']}")
    print(f"Minkowski p: {metrics['p']}")
    print(f"5-fold training CV RMSE: {metrics['cv_rmse']:.2f}")
    print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"Test MAE: {metrics['test_mae']:.2f}")
    print(f"Test R²: {metrics['test_r2']:.3f}")


if __name__ == "__main__":
    main()
