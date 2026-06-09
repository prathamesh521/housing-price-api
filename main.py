"""FastAPI application for housing price prediction."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Union

from fastapi import FastAPI, HTTPException, status

import predict
from schemas import (
    BatchPredictionResponse,
    HealthResponse,
    HouseFeatures,
    ModelInfoResponse,
    RootResponse,
    SinglePredictionResponse,
)

METRICS_PATH = Path("models/metrics.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the application starts."""
    try:
        predict.load_model()
    except FileNotFoundError as exc:
        app.state.model_load_error = str(exc)
    else:
        app.state.model_load_error = None

    yield


app = FastAPI(
    title="Housing Price Prediction API",
    description="Predict house prices using a trained Linear Regression model.",
    version="1.0.0",
    lifespan=lifespan,
)


def _ensure_model_loaded() -> None:
    """Raise a clear HTTP error if the model is unavailable."""
    try:
        predict.get_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.get("/", response_model=RootResponse, tags=["General"])
def read_root() -> RootResponse:
    """Return a simple welcome message."""
    return RootResponse(message="Housing Price Prediction API Running")


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="healthy")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    """Return saved model metadata and evaluation metrics."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Metrics file not found at {METRICS_PATH}. "
                "Run `python train.py` first."
            ),
        )

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics file is invalid JSON.",
        ) from exc

    required_fields = [
        "model_name",
        "feature_names",
        "coefficients",
        "intercept",
        "r2_score",
        "mae",
        "rmse",
    ]
    missing_fields = [field for field in required_fields if field not in metrics]
    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics file is missing fields: {missing_fields}",
        )

    return ModelInfoResponse(**metrics)


@app.post(
    "/predict",
    response_model=None,
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "examples": {
                        "single": {
                            "summary": "Single prediction",
                            "value": {"predicted_price": 285123.45},
                        },
                        "batch": {
                            "summary": "Batch prediction",
                            "value": {"predictions": [285123.45, 310542.22]},
                        },
                    }
                }
            },
        }
    },
    tags=["Prediction"],
)
def predict_price(
    payload: Union[HouseFeatures, List[HouseFeatures]],
) -> SinglePredictionResponse | BatchPredictionResponse:
    """
    Predict house price(s).

    Accepts either a single house object or a list of house objects.
    """
    _ensure_model_loaded()

    try:
        if isinstance(payload, list):
            predictions = predict.predict_batch(payload)
            return BatchPredictionResponse(predictions=predictions)

        predicted_price = predict.predict_single(payload)
        return SinglePredictionResponse(predicted_price=predicted_price)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an unexpected error.",
        ) from exc
