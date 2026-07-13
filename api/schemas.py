"""Pydantic schemas for API request/response models."""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in [x1, y1, x2, y2] format (pixel coordinates)."""
    x1: float = Field(..., description="Top-left X")
    y1: float = Field(..., description="Top-left Y")
    x2: float = Field(..., description="Bottom-right X")
    y2: float = Field(..., description="Bottom-right Y")


class DetectionResult(BaseModel):
    """Single detection result."""
    class_id: int = Field(..., description="Class ID (0=left_eye, 1=right_eye)")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., description="Confidence score [0, 1]")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")


class DetectionResponse(BaseModel):
    """API response for /api/v1/detect."""
    success: bool = Field(..., description="Whether detection succeeded")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    detection_count: int = Field(..., description="Number of detections")
    detections: List[DetectionResult] = Field(default_factory=list, description="Detection results")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_name: str = Field(..., description="Model name used for inference")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Model file path")
    device: str = Field(..., description="Inference device (cpu/cuda)")
