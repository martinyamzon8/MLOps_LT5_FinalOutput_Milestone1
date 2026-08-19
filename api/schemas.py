from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PassengerFeatures(BaseModel):
    """Input payload for PAL passenger market segment prediction.

    Validates numeric ranges:
    - PurchaseLeadTime >= 0
    - PAX_Count >= 1
    - AverageFare > 0
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "PurchaseLeadTime": 14.0,
                "PAX_Count": 1,
                "AverageFare": 145.50,
            }
        },
    )

    purchase_lead_time: float = Field(
        ...,
        alias="PurchaseLeadTime",
        validation_alias="PurchaseLeadTime",
        ge=0,
        description="Lead time in days between booking date and flight departure (>= 0)",
    )
    pax_count: int = Field(
        ...,
        alias="PAX_Count",
        validation_alias="PAX_Count",
        ge=1,
        description="Number of passengers on the booking reservation (>= 1)",
    )
    average_fare: float = Field(
        ...,
        alias="AverageFare",
        validation_alias="AverageFare",
        gt=0,
        description="Average ticket fare paid for the booking reservation (> 0)",
    )


class PredictionResponse(BaseModel):
    """Response payload returning predicted passenger market segment and confidence score."""

    status: str = Field(default="ok", description="Status string: ok")
    prediction: int = Field(..., description="Predicted cluster ID (e.g. 0, 1, 2)")
    segment_name: str = Field(..., description="Human-readable business segment label")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0",
    )
    model_version: str = Field(..., description="Active version of the served ML model")
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Echoed input features used for inference",
    )


class BatchPredictionRequest(BaseModel):
    """Batch input payload for multiple passenger records."""

    passengers: list[PassengerFeatures] = Field(
        ...,
        min_length=1,
        description="List of passenger booking feature payloads",
    )


class BatchPredictionResponse(BaseModel):
    """Batch response payload."""

    status: str = Field(default="ok")
    total_predictions: int = Field(..., description="Number of predictions returned")
    predictions: list[PredictionResponse] = Field(..., description="List of individual predictions")
    model_version: str = Field(...)


class HealthResponse(BaseModel):
    """Health check response payload confirming endpoint availability and model version."""

    status: str = Field(default="ok", description="Service health status: ok")
    model_version: str = Field(..., description="Served model version tag")
    model_name: str = Field(..., description="Name of the served machine learning model")
    model_source: str = Field(..., description="Source where model was loaded (mlflow or artifact)")
    timestamp: str = Field(..., description="UTC ISO timestamp of health check")
