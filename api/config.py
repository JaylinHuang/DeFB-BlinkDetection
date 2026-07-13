"""Configuration for DeFB API service."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model weights path (can be overridden by env var)
# Default: use the best checkpoint from YOLOv8 training
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output" / "yolo_eye_training" / "eye_detect" / "weights" / "best.pt"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

# Fallback: check common locations if default doesn't exist
if not Path(MODEL_PATH).exists():
    for candidate in [
        PROJECT_ROOT / "best.pt",
        PROJECT_ROOT / "weights" / "best.pt",
        Path("E:/sus/best.pt"),
    ]:
        if candidate.exists():
            MODEL_PATH = str(candidate)
            break

# Inference settings
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.15"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Class names (from YOLO training)
CLASS_NAMES = {0: "left_eye", 1: "right_eye"}
