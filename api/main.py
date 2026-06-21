"""
api/main.py
-----------
FastAPI endpoint for MVTec anomaly detection.
"""

from typing import Annotated, List, Literal
from datetime import datetime
from contextlib import asynccontextmanager
from sqlalchemy import desc, asc
from sqlalchemy.orm import Session
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException,
    Depends,
    Query,
)
from pydantic import BaseModel, ConfigDict, Field

from api.inference import (
    load_extractor,
    load_memory_bank,
    preprocess_pil,
    run_inference,
    OPTIMAL_THRESHOLDS,
)
from api.database import Prediction, get_db
from PIL import Image

import io


# Verify the api's output
class PredictResponse(BaseModel):
    """Schema for the prediction response."""

    score: float
    verdict: str
    threshold: float
    inference_time: float


class PredictionHistoryItem(BaseModel):
    """Schema for prediction hsitory"""

    model_config = ConfigDict(from_attributes=True)
    timestamp: datetime
    category: str
    filename: str
    score: float
    threshold: float
    verdict: str


#
@asynccontextmanager
async def lifespan(app):
    """Load the feature extractor once at server startup"""
    app.state.extractor = load_extractor()
    yield


app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="PatchCore-based anomaly detection on MVTec AD",
    version="1.0.0",
    lifespan=lifespan,
)


# health endpoint
@app.get("/health")
def health():
    """Check that the API is running."""
    return {"status": "ok", "model": "PatchCore", "categories": 15}


# predict endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    image: UploadFile = File(..., description="Image file to inspect (PNG or JPEG)"),
    category: str = Form(
        ..., description="MVTec product category (e.g. 'bottle', 'capsule')"
    ),
):
    """Run PatchCore inference on an uploaded image and return anomaly score and verdict

    Args:
        request (Request): FastAPI request object used to access the preloaded extractor.
        db (Annotated[Session, Depends(get_db)]): Prediction database saved for monitoring
        image (UploadFile): Image file to inspect (PNG or JPEG).
        category (str): MVTec product category (e.g. 'bottle', 'capsule').

    Raises:
        HTTPException: 404 if the category is not found in the supported categories.

    Returns:
        PredictResponse: Anomaly score, verdict, threshold and inference time.
    """
    # Load model
    extractor = request.app.state.extractor

    # Load memory bank
    if category not in OPTIMAL_THRESHOLDS.keys():
        raise HTTPException(status_code=404, detail="Category not found")
    memory_bank, meta = load_memory_bank(category)

    # Preprocces image
    image_bytes = image.file.read()  # read bytes
    pil_image = Image.open(io.BytesIO(image_bytes))  # convert image byttes to pil
    image_tensor = preprocess_pil(pil_image)

    # inference
    anomaly_score, heatmap, inference_time = run_inference(
        image_tensor, extractor, memory_bank
    )

    threshold = OPTIMAL_THRESHOLDS.get(category)

    verdict = "Defective" if anomaly_score >= threshold else "Normal"
    # Create prediction database
    db_prediction = Prediction(
        category=category,
        filename=image.filename,
        score=anomaly_score,
        threshold=threshold,
        verdict=verdict,
        inference_time=inference_time,
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return {
        "score": anomaly_score,
        "verdict": verdict,
        "threshold": threshold,
        "inference_time": inference_time,
    }


# History endpoint
@app.get("/history", response_model=List[PredictionHistoryItem])
def predictions_history(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(5, ge=1, le=100),
    sort_order: Literal["asc", "desc"] = Query(
        "desc", description="Sort order: asc or desc"
    ),
):
    query = db.query(Prediction)
    if sort_order.lower() == "desc":
        return query.order_by(desc(Prediction.timestamp)).limit(limit)
    return query.order_by(asc(Prediction.timestamp)).limit(limit)
