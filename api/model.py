"""YOLOv8 model loading and inference wrapper.

Singleton pattern ensures the model is loaded only once at startup.
"""

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, IMAGE_SIZE, CLASS_NAMES

logger = logging.getLogger("deb_api")

# Global model instance (singleton)
_model = None
_device = None


def get_model():
    """Lazy-load YOLO model (singleton)."""
    global _model, _device
    if _model is not None:
        return _model, _device

    from ultralytics import YOLO

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            f"Set MODEL_PATH env var to the correct location."
        )

    logger.info(f"Loading YOLO model from: {model_file}")
    _model = YOLO(str(model_file))

    # Detect available device
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"

    logger.info(f"Model loaded successfully. Device: {_device}")
    return _model, _device


def run_inference(
    image: np.ndarray,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_THRESHOLD,
    imgsz: int = IMAGE_SIZE,
) -> dict:
    """Run YOLO inference on a single image.

    Args:
        image: Input image as numpy array (BGR, HxWxC).
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        imgsz: Input image size for the model.

    Returns:
        Dict with detections, inference time, and image dimensions.
    """
    model, device = get_model()

    start = time.perf_counter()
    results = model(
        image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        device=device,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    r = results[0]
    h, w = image.shape[:2]

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": round(conf_val, 4),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                },
            })

    return {
        "detections": detections,
        "inference_time_ms": round(elapsed_ms, 2),
        "image_width": w,
        "image_height": h,
    }


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on the image (for annotated output).

    Args:
        image: Original image (BGR).
        detections: List of detection dicts from run_inference.

    Returns:
        Annotated image (BGR).
    """
    import cv2

    colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # left=blue, right=green

    annotated = image.copy()
    for det in detections:
        bbox = det["bbox"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        color = colors.get(det["class_id"], (0, 0, 255))

        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated
