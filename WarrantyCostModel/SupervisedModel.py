"""Predict observed warranty cost using information available before field failure."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data_Full.xlsx"
PREDICTIONS_PATH = BASE_DIR / "SupervisedTestPredictions.csv"

ID_COLUMN = "Serial Number"
TARGET_COLUMN = "Warranty Cost"
CATEGORY_BASELINE_COLUMN = "Product Family"

# These fields describe product configuration, manufacturing rework, or EOLT,
# and are available before the unit's field failure.
CATEGORICAL_FEATURES = [
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
    "Pass After Repair",
    "Retested",
]
NUMERIC_FEATURES = ["Labor Hours", "Material Cost", "Rework Hours"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

K_VALUES = (1, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100)
WEIGHT_OPTIONS = ("uniform", "distance")
MINKOWSKI_P_VALUES = (1, 2)
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_OF_TRAINING = 0.25


def _make_model(k: int, weights: str, p: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
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


def _score_predictions(
    actual: pd.Series, predicted: np.ndarray
) -> dict[str, float]:
    actual_total = float(actual.sum())
    predicted_total = float(predicted.sum())
    total_bias = predicted_total - actual_total
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "Actual test total": actual_total,
        "Predicted test total": predicted_total,
        "Total bias": total_bias,
        "Total bias percent": (
            100 * total_bias / actual_total if actual_total != 0 else float("nan")
        ),
    }


def train_and_evaluate(
    data: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        CATEGORY_BASELINE_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Data is missing required columns: {sorted(missing_columns)}")
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be present and unique for every unit")

    model_data = data[
        [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    ].copy()
    model_data[TARGET_COLUMN] = pd.to_numeric(
        model_data[TARGET_COLUMN], errors="raise"
    )
    if model_data[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("Model features and warranty cost must not contain missing values")
    if not np.isfinite(model_data[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Warranty cost must be finite")
    if (model_data[TARGET_COLUMN] < 0).any():
        raise ValueError("Warranty cost must be nonnegative")

    train_validation, test = train_test_split(
        model_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train, validation = train_test_split(
        train_validation,
        test_size=VALIDATION_SIZE_OF_TRAINING,
        random_state=RANDOM_STATE,
    )

    best_config: tuple[float, int, str, int] | None = None
    for k in K_VALUES:
        if k > len(train):
            continue
        for weights in WEIGHT_OPTIONS:
            for p in MINKOWSKI_P_VALUES:
                candidate = _make_model(k, weights, p)
                candidate.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
                validation_prediction = candidate.predict(validation[FEATURE_COLUMNS])
                validation_rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            validation[TARGET_COLUMN], validation_prediction
                        )
                    )
                )
                config = (validation_rmse, k, weights, p)
                if best_config is None or config[0] < best_config[0]:
                    best_config = config

    if best_config is None:
        raise ValueError("No valid k values for the training set")

    best_validation_rmse, best_k, best_weights, best_p = best_config
    fitted_data = pd.concat([train, validation])
    model = _make_model(best_k, best_weights, best_p)
    model.fit(fitted_data[FEATURE_COLUMNS], fitted_data[TARGET_COLUMN])
    knn_prediction = model.predict(test[FEATURE_COLUMNS])

    overall_mean = float(fitted_data[TARGET_COLUMN].mean())
    overall_prediction = np.full(len(test), overall_mean)
    family_means = fitted_data.groupby(CATEGORY_BASELINE_COLUMN)[
        TARGET_COLUMN
    ].mean()
    family_prediction = (
        test[CATEGORY_BASELINE_COLUMN].map(family_means).fillna(overall_mean).to_numpy()
    )
    train_mean = float(train[TARGET_COLUMN].mean())
    train_family_means = train.groupby(CATEGORY_BASELINE_COLUMN)[TARGET_COLUMN].mean()
    validation_family_prediction = (
        validation[CATEGORY_BASELINE_COLUMN]
        .map(train_family_means)
        .fillna(train_mean)
    )

    metrics = pd.DataFrame(
        {
            "Model": ["Overall mean baseline", "Product Family mean baseline", "kNN"],
            "Validation RMSE": [
                float(
                    np.sqrt(
                        mean_squared_error(
                            validation[TARGET_COLUMN],
                            np.full(len(validation), train_mean),
                        )
                    )
                ),
                float(
                    np.sqrt(
                        mean_squared_error(
                            validation[TARGET_COLUMN],
                            validation_family_prediction,
                        )
                    )
                ),
                best_validation_rmse,
            ],
            "Selected k": [np.nan, np.nan, best_k],
            "Weights": ["", "", best_weights],
            "Minkowski p": [np.nan, np.nan, best_p],
        }
    )

    test_metrics = pd.DataFrame(
        [
            {"Model": "Overall mean baseline", **_score_predictions(test[TARGET_COLUMN], overall_prediction)},
            {"Model": "Product Family mean baseline", **_score_predictions(test[TARGET_COLUMN], family_prediction)},
            {"Model": "kNN", **_score_predictions(test[TARGET_COLUMN], knn_prediction)},
        ]
    )

    predictions = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            "Actual Warranty Cost": test[TARGET_COLUMN].to_numpy(),
            "Overall Mean Prediction": overall_prediction,
            "Product Family Mean Prediction": family_prediction,
            "kNN Prediction": knn_prediction,
        }
    )
    return model, metrics, test_metrics, predictions


def main() -> None:
    data = pd.read_excel(DATA_PATH)
    model, validation_metrics, test_metrics, predictions = train_and_evaluate(data)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Units: {len(data):,}")
    print(f"Pre-failure features: {len(FEATURE_COLUMNS)}")
    print("\nValidation results (used to select kNN settings):")
    print(validation_metrics.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nUntouched test results:")
    print(test_metrics.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nSaved unit-level predictions to {PREDICTIONS_PATH}")
    print(
        "This model estimates warranty cost conditional on an observed failure; "
        "it does not estimate fleet failure probability."
    )


if __name__ == "__main__":
    main()