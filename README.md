# 🛡️ DeepShield — Production Deepfake Detection Web Application

DeepShield is a production-style, lightweight web application for verifying whether uploaded images or videos (or live webcam streams) are **REAL** or **FAKE** (deepfake). The system produces an authenticity verdict, a confidence score, Grad-CAM heatmaps for explainability, per-frame probability timeline charts for videos, and maintains a complete audit and detection history.

---

## 🌟 Key Features

1. **Multi-Modal Detection**:
   - **Image Analysis**: Drag-and-drop or file upload (JPEG, PNG, WebP, BMP, up to 10 MB). Face detection with OpenCV, artifact inspection, and Grad-CAM attention heatmap.
   - **Video Detection**: MP4, AVI, MOV, MKV up to 100 MB / 60 seconds. Samples frames with OpenCV, tracks per-frame fake probabilities in a timeline chart, and aggregates into a final verdict.
   - **Live Webcam**: Stream video from camera, send frames every ~500ms without queueing (skips frames when previous inference is in flight), real-time HUD badge overlay. Privacy-first: live frames are not saved to disk.
2. **Explainable AI (Grad-CAM)**: Visual side-by-side comparison of original image vs. neural activation heatmap (red highlights regions that triggered deepfake classification).
3. **Audit History & Analytics**:
   - Comprehensive dashboard with total scans, real/fake/uncertain breakdown, average confidence, distribution doughnut chart, and 30-day activity trend line.
   - Filterable & searchable history table (by filename, label, file type, date range), pagination, detailed inspection modal, single delete, clear all, and CSV export.
4. **Hardened Security & Edge Cases**:
   - Direct `bcrypt` password hashing (salt rounds) & HS256 JWT tokens.
   - Multi-face analysis, low-resolution & blur quality checks, corrupt/empty file safeguards, MIME & magic-byte validation, path-traversal sanitization (UUID storage), duplicate file hash caching.
   - Complete cross-tenant isolation (users can only access their own scans).

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Edge, Safari)

### 2. Installation
```bash
# Clone or navigate to the project directory
cd DeepSheild

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Seed Demo Data
Populate the database with a test user and 20 sample scan records:
```bash
python seed.py
```
> **Default Demo Account**:
> - **Email**: `demo@deepshield.com`
> - **Password**: `demo1234`

### 4. Run DeepShield
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Application**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Direct Login**: [http://127.0.0.1:8000/pages/login.html](http://127.0.0.1:8000/pages/login.html)

---

## 🧪 Running Automated Tests

Run the complete 53-test suite covering adapter contracts, API endpoints, auth guards, and synthetic edge cases:
```bash
pytest -v
```
To run specific suites:
```bash
# Adapter contract tests (11 tests)
pytest tests/test_adapter.py -v

# API integration tests (26 tests)
pytest tests/test_api.py -v

# Edge cases & file security tests (16 tests)
pytest tests/test_edge_cases.py -v
```

---

## 🔌 Integrating Your Real ML Module

DeepShield is built with strict architectural boundaries: **`backend/services/ml_adapter.py` is the only file that interfaces with ML models**.

### Step 1: Configure `.env`
Update `.env` with your model paths:
```env
# Toggle MOCK mode off
USE_MOCK_ML=false

# Point to your model and weights
ML_MODULE_PATH="C:/path/to/your/ml/folder"
ML_WEIGHTS_PATH="C:/path/to/weights.pth"
ML_GRADCAM_LAYER="layer4"
ML_LABEL_FAKE_INDEX=1
ML_THRESHOLD=0.5
UNCERTAIN_LOW=0.4
UNCERTAIN_HIGH=0.6
```

### Step 2: Implement Inference in `ml_adapter.py`
Open [backend/services/ml_adapter.py](file:///c:/Users/princ/Documents/one%20drive/OneDrive/Desktop/DeepSheild/backend/services/ml_adapter.py):
1. In `_load_real_model()`: import your module, load weights onto CUDA/CPU, and assign `_model` and `_transform`.
2. In `_real_predict_frame(frame_bgr)`: run forward pass, calculate `fake_probability` (0.0 to 1.0), and generate Grad-CAM heatmap.
3. Every adapter method adheres to this standard JSON contract:
```json
{
  "label": "REAL | FAKE | UNCERTAIN | NO_FACE",
  "confidence": 0.942,
  "fake_probability": 0.942,
  "faces_detected": 1,
  "heatmap_path": "heatmaps/xxx.png",
  "frame_scores": [{"frame": 0, "timestamp_s": 0.0, "fake_probability": 0.92}],
  "processing_time_ms": 312,
  "warnings": []
}
```

---

## ⚠️ Limitations & Ethical Disclaimer

- **Probabilistic Assessment**: DeepShield provides statistical confidence scores, not infallible truth. No deepfake detector is 100% accurate across all synthesis techniques.
- **Not Legal Evidence**: Detector verdicts should **never** be used as sole evidence in legal, judicial, disciplinary, or critical decisions. Independent human forensic review is required.
- **Image Quality Dependencies**: Heavily compressed, low-resolution (<64x64px), motion-blurred, extreme-angle, or dimly lit faces degrade detection fidelity. DeepShield automatically displays warning badges under these conditions.
- **Novel Generation Methods**: Generative models evolve rapidly; emerging deepfake generators not present in training distributions may produce false negatives.
