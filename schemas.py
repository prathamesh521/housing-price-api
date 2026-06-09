from typing import List

from pydantic import BaseModel, ConfigDict, Field

FEATURE_NAMES: list[str] = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]


class HouseFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "square_footage": 1800,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 2001,
                "lot_size": 7000,
                "distance_to_city_center": 4.5,
                "school_rating": 8.2,
            }
        }
    )

    square_footage: float = Field(..., gt=0, description="Living area in square feet")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    year_built: int = Field(..., ge=1800, le=2100, description="Year the house was built")
    lot_size: float = Field(..., gt=0, description="Lot size in square feet")
    distance_to_city_center: float = Field(
        ..., ge=0, description="Distance to city center in miles"
    )
    school_rating: float = Field(..., ge=0, le=10, description="Local school rating")


class SinglePredictionResponse(BaseModel):

    predicted_price: float


class BatchPredictionResponse(BaseModel):

    predictions: List[float]


class HealthResponse(BaseModel):

    status: str


class RootResponse(BaseModel):

    message: str


class ModelInfoResponse(BaseModel):

    model_name: str
    feature_names: List[str]
    coefficients: dict[str, float]
    intercept: float
    r2_score: float
    mae: float
    rmse: float
