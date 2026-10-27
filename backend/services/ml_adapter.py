"""
ml_adapter.py — THE ONLY FILE that may import from the user's ML module.

Contract: every public function returns a dict matching PredictionResult.
- load_model()           → called once at FastAPI startup
- predict_image(path)    → accepts image file path
- predict_video(path)    → accepts video file path (samples frames via OpenCV)
- predict_frame(frame)   → accepts BGR numpy array (for webcam)

To wire in your real model:
  1. Set USE_MOCK_ML=false in .env
  2. Set ML_MODULE_PATH, ML_WEIGHTS_PATH, ML_GRADCAM_LAYER, etc. in .env
  3. Implement _real_predict_frame() below — it's the only function you
     need to change. Do NOT touch anything else.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Internal imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Shared model state (loaded once) ────────────────────────────────────────
_model: Any = None
_device: Any = None
_transform: Any = None
_model_loaded: bool = False
_mock_mode: bool = settings.USE_MOCK_ML

# ─── JSON contract type (returned by every public function) ───────────────────
LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"
LABEL_UNCERTAIN = "UNCERTAIN"
LABEL_NO_FACE = "NO_FACE"


def _empty_result() -> dict:
    return {
        "label": LABEL_UNCERTAIN,
        "confidence": 0.0,
        "fake_probability": 0.0,
        "faces_detected": 0,
        "heatmap_path": None,
        "frame_scores": [],
        "processing_time_ms": 0,
        "warnings": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_predict(source_hint: str = "image", n_frames: int = 1) -> dict:
    """Returns a plausible but random result for UI development / testing."""
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        time.sleep(random.uniform(0.3, 1.2))  # simulate inference latency in UI
    fake_prob = random.uniform(0.05, 0.95)

    if fake_prob < settings.UNCERTAIN_LOW:
        label = LABEL_REAL
        confidence = 1.0 - fake_prob
    elif fake_prob > settings.UNCERTAIN_HIGH:
        label = LABEL_FAKE
        confidence = fake_prob
    else:
        label = LABEL_UNCERTAIN
        confidence = abs(fake_prob - 0.5) * 2  # 0–0.2

    faces = random.randint(0, 3)
    if faces == 0:
        label = LABEL_NO_FACE
        confidence = 0.0

    frame_scores = []
    if n_frames > 1:
        frame_scores = [
            {"frame": i * 10, "fake_probability": random.uniform(0.0, 1.0)}
            for i in range(n_frames)
        ]

    warnings: list[str] = []
    if random.random() < 0.2:
        warnings.append("Low image quality detected — result may be less reliable.")

    # Mock heatmap: a small solid-color PNG
    heatmap_path = _save_mock_heatmap()

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "faces_detected": faces,
        "heatmap_path": heatmap_path,
        "frame_scores": frame_scores,
        "processing_time_ms": 0,  # overwritten by caller
        "warnings": warnings,
    }


def _save_mock_heatmap() -> str | None:
    """Save a fake gradient heatmap image and return its relative path."""
    try:
        heatmap_dir = settings.heatmap_path
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        full_path = heatmap_dir / filename

        # Generate a simple random heatmap gradient
        heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
        for y in range(224):
            for x in range(224):
                heatmap[y, x] = [
                    int(255 * (x / 224)),
                    int(128 * (y / 224)),
                    int(200 * ((224 - x) / 224)),
                ]
        cv2.imwrite(str(full_path), heatmap)
        return str(full_path)
    except Exception as e:
        logger.warning(f"Could not save mock heatmap: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  REAL MODEL ADAPTER  (fill in when your ML module is ready)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_real_model() -> None:
    """
    ── INTEGRATION POINT ──
    Import your ML module and load weights here.
    Adjust the import path and constructor to match your model.
    """
    global _model, _device, _transform, _model_loaded

    try:
        import torch  # noqa: F401

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")

        # ── Replace the block below with your actual model loading ──
        #
        # Example:
        #   sys.path.insert(0, settings.ML_MODULE_PATH)
        #   from your_module import YourModel, get_transform
        #   _model = YourModel()
        #   _model.load_state_dict(torch.load(settings.ML_WEIGHTS_PATH,
        #                                      map_location=_device))
        #   _model.eval().to(_device)
        #   _transform = get_transform()
        #
        # ── MISMATCH REPORTING ──
        # If your module's interface doesn't match the contract,
        # raise ValueError("Mismatch: <describe the mismatch>")
        # The adapter will catch this and log it clearly.

        raise NotImplementedError(
            "Real model not yet configured. "
            "Fill in _load_real_model() in ml_adapter.py "
            "and set USE_MOCK_ML=false in .env"
        )

    except NotImplementedError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Model load failed: {exc}") from exc


def _real_predict_frame(frame_bgr: np.ndarray) -> tuple[float, str | None]:
    """
    ── INTEGRATION POINT ──
    Run inference on a single BGR frame. Return (fake_probability, heatmap_path).

    Example skeleton:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = _transform(pil_img).unsqueeze(0).to(_device)
        with torch.no_grad():
            output = _model(tensor)
            prob = torch.sigmoid(output)[0, settings.ML_LABEL_FAKE_INDEX].item()
        heatmap_path = _generate_gradcam(tensor, frame_bgr)
        return prob, heatmap_path
    """
    raise NotImplementedError("Real prediction not yet implemented.")


def _generate_gradcam(tensor: Any, original_bgr: np.ndarray) -> str | None:
    """
    ── INTEGRATION POINT ──
    Generate Grad-CAM heatmap using pytorch-grad-cam.
    Returns saved heatmap path or None on failure.

    Example:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        target_layer = [getattr(_model, settings.ML_GRADCAM_LAYER)]
        cam = GradCAM(model=_model, target_layers=target_layer)
        grayscale_cam = cam(input_tensor=tensor)[0]
        rgb_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return _save_heatmap_array(visualization)
    """
    return None  # gracefully return None; UI hides heatmap when null


def _save_heatmap_array(img_rgb: np.ndarray) -> str | None:
    try:
        heatmap_dir = settings.heatmap_path
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        full_path = heatmap_dir / filename
        cv2.imwrite(str(full_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return str(full_path)
    except Exception as e:
        logger.warning(f"Could not save heatmap: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE DETECTION (shared between mock and real)
# ═══════════════════════════════════════════════════════════════════════════════

_face_cascade: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _count_faces(frame_bgr: np.ndarray) -> int:
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade = _get_face_cascade()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return len(faces) if isinstance(faces, np.ndarray) else 0
    except Exception:
        return 0


def _check_image_quality(frame_bgr: np.ndarray) -> list[str]:
    warnings: list[str] = []
    h, w = frame_bgr.shape[:2]

    if h < 64 or w < 64:
        warnings.append("Image resolution is very low — result may be unreliable.")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 50:
        warnings.append("Image appears blurry — result may be less reliable.")

    mean_brightness = float(np.mean(gray))
    if mean_brightness < 30:
        warnings.append("Image is very dark — result may be less reliable.")
    elif mean_brightness > 225:
        warnings.append("Image is very bright/overexposed — result may be less reliable.")

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def load_model() -> None:
    """Called once at FastAPI startup. Sets _model_loaded flag."""
    global _model_loaded, _mock_mode

    if _mock_mode:
        logger.info("ML Adapter: MOCK MODE enabled — no real model will be loaded.")
        _model_loaded = True
        return

    try:
        _load_real_model()
        _model_loaded = True
        logger.info("ML Adapter: Real model loaded successfully.")
    except NotImplementedError as e:
        logger.warning(f"ML Adapter: {e}")
        logger.warning("Falling back to MOCK mode automatically.")
        _mock_mode = True
        _model_loaded = True
    except Exception as e:
        logger.error(f"ML Adapter: Model load failed — {e}")
        logger.warning("Falling back to MOCK mode automatically.")
        _mock_mode = True
        _model_loaded = True


def is_mock_mode() -> bool:
    return _mock_mode


def is_model_loaded() -> bool:
    return _model_loaded


def predict_image(path: str | Path) -> dict:
    """
    Predict whether an image is REAL or FAKE.
    Returns the full JSON contract dict.
    """
    start = time.perf_counter()
    result = _empty_result()

    try:
        path = Path(path)
        if not path.exists():
            result["warnings"].append("File not found.")
            return result

        # Load + validate image
        try:
            pil_img = Image.open(path)
            pil_img.verify()
            pil_img = Image.open(path)  # reopen after verify()
        except (UnidentifiedImageError, Exception) as e:
            result["warnings"].append(f"Cannot open image: {e}")
            return result

        # Normalise to RGB BGR numpy array
        pil_img = pil_img.convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Quality checks
        result["warnings"].extend(_check_image_quality(frame_bgr))

        # Face count
        n_faces = _count_faces(frame_bgr)
        result["faces_detected"] = n_faces
        if n_faces == 0:
            result["label"] = LABEL_NO_FACE
            result["warnings"].append(
                "No face detected. DeepShield works best with clear face images."
            )
            result["processing_time_ms"] = int(
                (time.perf_counter() - start) * 1000
            )
            return result

        # Inference
        if _mock_mode:
            r = _mock_predict(source_hint="image")
            r["faces_detected"] = n_faces
            r["warnings"] = result["warnings"] + r["warnings"]
            r["processing_time_ms"] = int((time.perf_counter() - start) * 1000)
            return r

        # ── Real inference ──
        try:
            fake_prob, heatmap_path = _real_predict_frame(frame_bgr)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            result["warnings"].append(
                "Inference failed. Please try again or contact support."
            )
            return result

        label = _prob_to_label(fake_prob)
        confidence = _prob_to_confidence(fake_prob, label)

        result.update(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "fake_probability": round(fake_prob, 4),
                "faces_detected": n_faces,
                "heatmap_path": heatmap_path,
            }
        )

    except Exception as e:
        logger.exception(f"Unexpected error in predict_image: {e}")
        result["warnings"].append("An unexpected error occurred during analysis.")
    finally:
        result["processing_time_ms"] = int((time.perf_counter() - start) * 1000)

    return result


def predict_video(path: str | Path) -> dict:
    """
    Predict whether a video contains deepfake content.
    Samples every ~10th frame, max 100 frames.
    Returns the full JSON contract dict.
    """
    start = time.perf_counter()
    result = _empty_result()
    frame_scores: list[dict] = []

    try:
        path = Path(path)
        if not path.exists():
            result["warnings"].append("Video file not found.")
            return result

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            result["warnings"].append("Cannot open video file. It may be corrupted.")
            return result

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration_s = total_frames / fps

        if total_frames == 0:
            result["warnings"].append("Video has zero frames.")
            cap.release()
            return result

        if duration_s > settings.MAX_VIDEO_DURATION_S:
            result["warnings"].append(
                f"Video is longer than {settings.MAX_VIDEO_DURATION_S}s — "
                f"only the first {settings.MAX_VIDEO_DURATION_S}s will be analyzed."
            )

        # Frame sampling
        max_duration_frames = int(settings.MAX_VIDEO_DURATION_S * fps)
        sample_every = max(1, total_frames // 100)
        frame_indices = list(range(0, min(total_frames, max_duration_frames), sample_every))

        fake_probs: list[float] = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            n_faces = _count_faces(frame)
            if n_faces == 0:
                continue  # skip frames with no face

            if _mock_mode:
                fp = random.uniform(0.0, 1.0)
            else:
                try:
                    fp, _ = _real_predict_frame(frame)
                except Exception:
                    continue

            fake_probs.append(fp)
            frame_scores.append(
                {
                    "frame": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 2),
                    "fake_probability": round(fp, 4),
                }
            )

        cap.release()

        if not fake_probs:
            result["label"] = LABEL_NO_FACE
            result["warnings"].append(
                "No faces detected in any sampled frame. "
                "Cannot determine authenticity."
            )
            result["frame_scores"] = frame_scores
            result["processing_time_ms"] = int(
                (time.perf_counter() - start) * 1000
            )
            return result

        mean_prob = float(np.mean(fake_probs))
        max_prob = float(np.max(fake_probs))
        pct_fake = sum(1 for p in fake_probs if p > settings.ML_THRESHOLD) / len(
            fake_probs
        )

        # Aggregate verdict: mean drives label, max is a warning signal
        label = _prob_to_label(mean_prob)
        confidence = _prob_to_confidence(mean_prob, label)

        if pct_fake > 0.5 and label == LABEL_REAL:
            result["warnings"].append(
                f"{int(pct_fake * 100)}% of frames scored as FAKE — treat result with caution."
            )

        # Heatmap for the frame with max fake probability
        heatmap_path = None
        if _mock_mode:
            heatmap_path = _save_mock_heatmap()

        result.update(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "fake_probability": round(mean_prob, 4),
                "faces_detected": len(fake_probs),  # frames with faces
                "heatmap_path": heatmap_path,
                "frame_scores": frame_scores,
            }
        )

    except Exception as e:
        logger.exception(f"Unexpected error in predict_video: {e}")
        result["warnings"].append("An unexpected error occurred during video analysis.")
    finally:
        result["processing_time_ms"] = int((time.perf_counter() - start) * 1000)

    return result


def predict_frame(frame_bgr: np.ndarray) -> dict:
    """
    Predict a single BGR numpy frame (webcam use).
    Returns the full JSON contract dict (no heatmap for speed).
    """
    start = time.perf_counter()
    result = _empty_result()

    try:
        if frame_bgr is None or frame_bgr.size == 0:
            result["warnings"].append("Empty frame received.")
            return result

        n_faces = _count_faces(frame_bgr)
        result["faces_detected"] = n_faces

        if n_faces == 0:
            result["label"] = LABEL_NO_FACE
            result["processing_time_ms"] = int(
                (time.perf_counter() - start) * 1000
            )
            return result

        if _mock_mode:
            r = _mock_predict(source_hint="frame")
            r["faces_detected"] = n_faces
            r["heatmap_path"] = None  # skip heatmap for live frames
            r["processing_time_ms"] = int((time.perf_counter() - start) * 1000)
            return r

        try:
            fake_prob, _ = _real_predict_frame(frame_bgr)
        except Exception as e:
            logger.error(f"Frame inference error: {e}")
            result["warnings"].append("Frame inference failed.")
            return result

        label = _prob_to_label(fake_prob)
        confidence = _prob_to_confidence(fake_prob, label)

        result.update(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "fake_probability": round(fake_prob, 4),
                "faces_detected": n_faces,
                "heatmap_path": None,
            }
        )

    except Exception as e:
        logger.exception(f"Unexpected error in predict_frame: {e}")
        result["warnings"].append("An unexpected error occurred.")
    finally:
        result["processing_time_ms"] = int((time.perf_counter() - start) * 1000)

    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _prob_to_label(fake_prob: float) -> str:
    if fake_prob >= settings.UNCERTAIN_HIGH:
        return LABEL_FAKE
    if fake_prob <= settings.UNCERTAIN_LOW:
        return LABEL_REAL
    return LABEL_UNCERTAIN


def _prob_to_confidence(fake_prob: float, label: str) -> float:
    if label == LABEL_FAKE:
        return fake_prob
    if label == LABEL_REAL:
        return 1.0 - fake_prob
    # UNCERTAIN: how far from 0.5 on a 0–1 scale
    return round(abs(fake_prob - 0.5) / 0.1, 4)  # 0.0 at 0.5, 1.0 at 0.4/0.6

