"""FastAPI application — DeFB Eye Detection Service.

Endpoints:
  GET  /api/v1/health       — Health check
  POST /api/v1/detect        — Detect eyes in an uploaded image (returns JSON)
  POST /api/v1/detect/image  — Detect eyes and return annotated image
  GET  /docs                 — Swagger UI (auto-generated)
"""

import io
import logging
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from .config import MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, IMAGE_SIZE, API_HOST, API_PORT
from .schemas import DetectionResponse, HealthResponse, DetectionResult, BoundingBox
from .model import get_model, run_inference, draw_detections

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deb_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    logger.info("Initializing DeFB API service...")
    try:
        model, device = get_model()
        logger.info(f"Model ready on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but /detect endpoints will return 500")
    yield
    logger.info("Shutting down DeFB API service...")


app = FastAPI(
    title="DeFB Eye Detection API",
    description="""
## Non-restricted Blink Detection — Eye Localization Service

A FastAPI service wrapping a YOLOv8-Nano model for real-time eye detection.

### Features
- **POST /api/v1/detect**: Upload an image, get detection results as JSON
- **POST /api/v1/detect/image**: Upload an image, get annotated image back
- **GET /api/v1/health**: Service health check

### Model
- Architecture: YOLOv8-Nano (anchor-free, decoupled head)
- Classes: `left_eye`, `right_eye`
- Input size: 640x640
""",
    version="1.0.0",
    lifespan=lifespan,
)


def read_upload_image(file: UploadFile) -> np.ndarray:
    """Read uploaded file as OpenCV BGR image."""
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Decode image from bytes
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported: JPG, PNG, BMP, WEBP",
        )
    return image


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and model status."""
    from .model import _model, _device
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_path=MODEL_PATH,
        device=_device or "not_loaded",
    )


@app.post(
    "/api/v1/detect",
    response_model=DetectionResponse,
    tags=["Detection"],
    summary="Detect eyes in an image (JSON response)",
)
async def detect_eyes(
    file: UploadFile = File(..., description="Image file (JPG/PNG/BMP)"),
    conf: float = Query(CONF_THRESHOLD, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(IOU_THRESHOLD, ge=0.0, le=1.0, description="NMS IoU threshold"),
):
    """
    Upload an image and get eye detection results as JSON.

    Returns bounding boxes, class names, confidence scores, and inference time.
    """
    try:
        image = read_upload_image(file)
    except HTTPException:
        raise
    finally:
        file.file.close()

    try:
        result = run_inference(image, conf=conf, iou=iou)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {e}")
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    detections = [
        DetectionResult(
            class_id=d["class_id"],
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"]),
        )
        for d in result["detections"]
    ]

    model_name = Path(MODEL_PATH).name
    return DetectionResponse(
        success=True,
        image_width=result["image_width"],
        image_height=result["image_height"],
        detection_count=len(detections),
        detections=detections,
        inference_time_ms=result["inference_time_ms"],
        model_name=model_name,
    )


@app.post(
    "/api/v1/detect/image",
    tags=["Detection"],
    summary="Detect eyes and return annotated image",
)
async def detect_eyes_annotated(
    file: UploadFile = File(..., description="Image file (JPG/PNG/BMP)"),
    conf: float = Query(CONF_THRESHOLD, ge=0.0, le=1.0, description="Confidence threshold"),
):
    """
    Upload an image and get it back with bounding boxes drawn.

    Returns the annotated image as a PNG file.
    """
    try:
        image = read_upload_image(file)
    except HTTPException:
        raise
    finally:
        file.file.close()

    try:
        result = run_inference(image, conf=conf)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    annotated = draw_detections(image, result["detections"])

    # Encode as PNG
    success, buffer = cv2.imencode(".png", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output image")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"X-Detection-Count": str(len(result["detections"]))},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
