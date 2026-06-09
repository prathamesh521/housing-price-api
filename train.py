from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from schemas import FEATURE_NAMES

DATA_PATH = Path("data/HousePriceDataset.csv")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
TARGET_COLUMN = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET_COLUMN}")

    return df


def train_and_evaluate(df: pd.DataFrame) -> tuple[LinearRegression, dict[str, float | str | dict]]:
    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics: dict[str, float | str | dict] = {
        "model_name": "LinearRegression",
        "feature_names": FEATURE_NAMES,
        "coefficients": {
            feature: float(coef)
            for feature, coef in zip(FEATURE_NAMES, model.coef_, strict=True)
        },
        "intercept": float(model.intercept_),
        "r2_score": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }

    return model, metrics


def save_artifacts(model: LinearRegression, metrics: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def main() -> None:
    df = load_dataset(DATA_PATH)
    model, metrics = train_and_evaluate(df)
    save_artifacts(model, metrics)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"MAE:      {metrics['mae']:.2f}")
    print(f"RMSE:     {metrics['rmse']:.2f}")


if __name__ == "__main__":
    main()
