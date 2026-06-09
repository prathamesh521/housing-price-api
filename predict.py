from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from schemas import FEATURE_NAMES, HouseFeatures

MODEL_PATH = Path("models/model.pkl")

_model: LinearRegression | None = None


class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is attempted before the model is loaded."""


def load_model(model_path: Path = MODEL_PATH) -> LinearRegression:
    """Load the trained model from disk and cache it in memory."""
    global _model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python train.py` first."
        )

    _model = joblib.load(model_path)
    return _model


def get_model() -> LinearRegression:
    global _model

    if _model is None:
        return load_model()

    return _model


def _features_to_dataframe(features: Sequence[HouseFeatures]) -> pd.DataFrame:
    rows = [house.model_dump() for house in features]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def predict_single(house: HouseFeatures) -> float:
    model = get_model()
    feature_frame = _features_to_dataframe([house])
    prediction = model.predict(feature_frame)[0]
    return float(prediction)


def predict_batch(houses: Sequence[HouseFeatures]) -> list[float]:
    if not houses:
        raise ValueError("Batch prediction requires at least one house.")

    model = get_model()
    feature_frame = _features_to_dataframe(houses)
    predictions = model.predict(feature_frame)
    return [float(value) for value in predictions]
