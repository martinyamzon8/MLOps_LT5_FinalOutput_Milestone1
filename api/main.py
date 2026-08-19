from __future__ import annotations

import datetime as dt
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model_loader import model_service
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PassengerFeatures,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler to load the model on startup."""
    print("Initializing PAL Passenger Segmentation Serving Endpoint...")
    model_service.load_model()
    yield
    print("Shutting down PAL Passenger Segmentation Serving Endpoint...")


app = FastAPI(
    title="Philippine Airlines Passenger Market Segmentation API",
    description=(
        "Production model serving endpoint for the PAL Passenger Segmentation system. "
        "Predicts passenger segment (e.g. Business/Premium, Standard Leisure, Advance Group) "
        "and confidence score from booking characteristics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for demo and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check and model version verification",
    tags=["System"],
)
def get_health() -> HealthResponse:
    """Check service health and confirm active model version.

    Used during live demos to verify the containerized serving endpoint is operational.
    """
    return HealthResponse(
        status="ok",
        model_version=model_service.model_version,
        model_name=model_service.model_name,
        model_source=model_service.model_source,
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict market segment for an individual passenger booking",
    tags=["Inference"],
)
def predict(passenger: PassengerFeatures) -> PredictionResponse:
    """Predict market segment and confidence score for a single booking reservation.

    - **PurchaseLeadTime**: Number of days between booking and flight departure (>= 0)
    - **PAX_Count**: Total number of passengers on booking reservation (>= 1)
    - **AverageFare**: Average ticket fare paid (> 0)

    Returns HTTP 422 Unprocessable Entity if input data violates schema constraints.
    """
    result = model_service.predict(
        purchase_lead_time=passenger.purchase_lead_time,
        pax_count=passenger.pax_count,
        average_fare=passenger.average_fare,
    )
    return PredictionResponse(
        status="ok",
        prediction=result["prediction"],
        segment_name=result["segment_name"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        features=result["features"],
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch predict market segment for multiple passenger bookings",
    tags=["Inference"],
)
def predict_batch(batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run market segment predictions across a batch of passenger booking records."""
    predictions = []
    for passenger in batch_request.passengers:
        result = model_service.predict(
            purchase_lead_time=passenger.purchase_lead_time,
            pax_count=passenger.pax_count,
            average_fare=passenger.average_fare,
        )
        predictions.append(
            PredictionResponse(
                status="ok",
                prediction=result["prediction"],
                segment_name=result["segment_name"],
                confidence=result["confidence"],
                model_version=result["model_version"],
                features=result["features"],
            )
        )

    return BatchPredictionResponse(
        status="ok",
        total_predictions=len(predictions),
        predictions=predictions,
        model_version=model_service.model_version,
    )


@app.get(
    "/",
    summary="API Root Information",
    tags=["System"],
)
def root_info() -> JSONResponse:
    """Root metadata with links to health checks, Swagger docs, and inference endpoints."""
    return JSONResponse(
        {
            "system": "Philippine Airlines Passenger Market Segmentation API",
            "version": "1.0.0",
            "model_version": model_service.model_version,
            "status": "online",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "predict_batch": "/predict/batch",
                "docs": "/docs",
                "openapi": "/openapi.json",
            },
        }
    )
