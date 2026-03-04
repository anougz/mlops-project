import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
model = None  # loaded once at startup


# ---------------------------------------------------------------------------
# Lifespan: load model when the app starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LSTM model from MLflow (or a local artifact) at startup."""
    global model

    model_uri = os.getenv("MODEL_URI", "models:/lstm-pollution/Production")
    # Fallback: if a local path is set, use it directly
    local_model_path = os.getenv("LOCAL_MODEL_PATH", "")

    try:
        if local_model_path:
            logger.info("Loading model from local path: %s", local_model_path)
            model = mlflow.keras.load_model(local_model_path)
        else:
            logger.info("Loading model from MLflow URI: %s", model_uri)
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            model = mlflow.keras.load_model(model_uri)
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        # The app will still start; /predict will return 503 until the model is available.

    yield  # application runs here

    logger.info("Shutting down — releasing model.")
    model = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pollution LSTM Inference API",
    description=(
        "Serves a multivariate LSTM model trained on air-pollution data. "
        "Accepts a sequence of past observations and returns the next-step prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """
    `sequence` is a list of time steps.
    Each time step is a list of feature values (must match the number of
    features the model was trained with).

    Example for 24 time steps × 8 features:
        {"sequence": [[val, val, ...], [val, val, ...], ...]}
    """

    sequence: list[list[float]] = Field(
        ...,
        description="2-D array of shape [n_timesteps, n_features]",
        example=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 24,
    )


class PredictionResponse(BaseModel):
    prediction: list[float] = Field(
        ..., description="Model output — predicted value(s) for the next time step"
    )
    inference_time_ms: float = Field(..., description="Wall-clock inference time in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Liveness / readiness probe."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Run inference with the loaded LSTM model.

    - **sequence**: list of past time-steps, each being a list of feature values.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check MODEL_URI / LOCAL_MODEL_PATH and restart.",
        )

    # --- Build input tensor: (1, timesteps, features) ---
    try:
        arr = np.array(request.sequence, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("sequence must be 2-D (timesteps × features).")
        x = arr[np.newaxis, ...]  # add batch dimension
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid input shape: {exc}")

    # --- Inference ---
    t0 = time.perf_counter()
    try:
        raw = model.predict(x)  # shape: (1, n_outputs)
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    prediction = raw.flatten().tolist()
    logger.info("Prediction: %s  (%.1f ms)", prediction, elapsed_ms)

    return PredictionResponse(prediction=prediction, inference_time_ms=round(elapsed_ms, 2))
