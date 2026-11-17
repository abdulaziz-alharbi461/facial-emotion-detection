# Real-Time Facial Emotion and Demographic Intelligence System

A high-performance, production-ready computer vision and deep learning system for real-time facial analysis. The application concurrently estimates facial emotion, apparent age, gender, and broad geographic demographics from live video feeds with sub-20ms expression latency and stabilized telemetry.

---

## Overview

Traditional facial analysis pipelines often suffer from frame drops, UI stuttering, erratic age estimations, and resting-face neutral bias that masks subtle micro-expressions. This system addresses these challenges through:

- Multi-Threaded Asynchronous Decoupling: Separates high-speed emotion tracking from compute-heavy demographic inference.
- Calibrated Sensitivity Engine: Eliminates neutral-bias dominance, enabling rapid detection of subtle expressions such as slight smiles and sadness.
- Two-Stage Age Stabilization: Combines rolling median outlier rejection with exponential moving average (EMA) smoothing to eliminate erratic jumps across frames.
- Environmental Normalization: Applies CLAHE in LAB color space and bilateral filtering to maintain accuracy across poor lighting conditions and camera sensor noise.
- Clean HUD Rendering: Employs bounding box coordinate smoothing and high-contrast glassmorphic badges to prevent visual flickering and text collisions.

---

## Core Capabilities

### 1. High-Speed Emotion Recognition
- Evaluates 7 affective states: Happy, Sad, Neutral, Angry, Surprise, Fear, and Disgust.
- Employs neutral-bias compensation: resting human faces naturally score 45-60% neutral; the calibration algorithm detects subtle muscle activations (lip corner droop, brow tension) down to 16% threshold.
- Probability EMA buffer (alpha = 0.40) provides flicker-free transitions between consecutive frames.

### 2. Stabilized Apparent Age Estimation
- Solves single-frame prediction volatility caused by lighting reflections, facial tilt, and sensor grain.
- Stage 1: Rolling Median Filter (window size = 9) discards statistical outliers.
- Stage 2: Low-pass Exponential Moving Average filter (alpha = 0.12) produces a steady, natural age readout.

### 3. Demographic & Gender Profiling
- Classifies gender with probability confidence scores.
- Estimates broad regional geographic origin as a demographic proxy:
  - East / Southeast Asia
  - South Asia
  - Middle Eastern / West Asian
  - Latino / Hispanic
  - European / Caucasian
  - African / Afro-Descendant

### 4. Advanced Preprocessing Pipeline
- Bilateral Filtering: Removes high-frequency camera noise while preserving sharp facial edges and landmarks.
- LAB CLAHE: Performs Contrast Limited Adaptive Histogram Equalization on the luminance channel, recovering facial contours in underexposed or unevenly lit environments.
- Aspect Ratio Validation: Filters out partial hand or smartphone occlusions by enforcing human facial geometry ratios (0.65 to 1.25).

---

## System Architecture

```text
+-------------------------------------------------------------------------+
|                               Webcam Feed                               |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 OpenCV Haar / ROI Detection & Filtering                 |
|         - Enforces aspect ratio validation (0.65 to 1.25)               |
|         - Selects primary face (maximum bounding area)                  |
|         - Applies EMA coordinate smoothing (beta = 0.35)                |
+-------------------------------------------------------------------------+
                                     |
                                     +--------------------+
                                     |                    |
                                     v                    v
+-------------------------------------------+   +---------------------------------------+
|        Tier 1: Fast Emotion Worker        |   | Tier 2: Background Demographic Worker |
| - CLAHE & bilateral preprocessing         |   | - Denoised ROI input                  |
| - Skip detector backend (< 20ms)          |   | - Evaluates Age, Gender, Race         |
| - Calibrated sensitivity engine           |   | - Asynchronous cycle (~3.0s interval) |
| - 6-frame probability EMA buffer          |   | - Rolling median + EMA age filter     |
+-------------------------------------------+   +---------------------------------------+
                                     |                    |
                                     +--------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    Thread-Safe AppState Synchronization                 |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         Unified HUD & Overlay                           |
| - Corner-accented bounding box (dynamic emotion color)                  |
| - Glassmorphic high-contrast badge (non-colliding layout)               |
| - Top system status telemetry bar & real-time FPS counter               |
+-------------------------------------------------------------------------+
```

---

## Repository Structure

```text
facial_emotion_detection/
├── main.py                   # Standalone production pipeline (fully integrated)
├── main_pro.py               # Modular production orchestrator
├── download_demographics.py  # Automated model weights downloader with resume support
├── requirements.txt          # Python environment dependencies
├── README.md                 # System technical documentation
└── app/                      # Modular architecture components
    ├── __init__.py           # Package initializer
    ├── camera.py             # Threaded video capture for zero latency
    ├── analyzer.py           # Multi-threaded DeepFace demographic worker pool
    ├── smoother.py           # Temporal EMA and moving-window smoothing engine
    ├── alignment.py          # 2D affine face normalization and eye alignment
    ├── preprocessor.py       # CLAHE, gamma correction, and bilateral filters
    ├── tta.py                # Test-Time Augmentation utilities
    ├── analytics.py          # Session telemetry and JSON data exporter
    └── ui.py                 # Glassmorphic HUD rendering and overlays
```

---

## Installation and Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11 (64-bit recommended)
- A connected USB or integrated webcam
- Compatible operating system: Windows 10/11, macOS, or Linux

### 1. Clone the Repository
```bash
git clone https://github.com/abdulaziz-alharbi461/facial-emotion-detection.git
cd facial-emotion-detection
```

### 2. Create and Activate a Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Neural Network Model Weights

The system utilizes pre-trained deep convolutional neural networks for inference:
1. `facial_expression_model_weights.h5` (~6 MB): Emotion classification
2. `age_model_weights.h5` (~513 MB): Apparent age estimation
3. `gender_model_weights.h5` (~512 MB): Gender classification
4. `race_model_single_batch.h5` (~512 MB): Broad geographic origin

Weights are stored locally in the standard DeepFace directory (`~/.deepface/weights/`).

To verify or pre-download all weights with integrity checking and resume capability:
```powershell
python download_demographics.py
```

---

## Usage

### Run Standalone Engine (Recommended)
The standalone script contains all lighting normalization, sensitivity calibration, and age stabilization logic in a self-contained execution:
```powershell
python main.py
```

### Run Modular Architecture
The modular version uses decoupled worker classes from the `app/` package:
```powershell
python main_pro.py
```

### Keyboard Shortcuts
- `f`: Toggle between Fullscreen and Windowed display mode.
- `q` or `ESC`: Terminate the video feed cleanly and release resources.

---

## Performance Optimization

For lower-specification hardware without dedicated GPU acceleration:

1. Analysis Interval:
   Adjust `min_interval_sec` in `FastEmotionWorker` or `interval_sec` in `BackgroundDemographicWorker` to balance CPU consumption with responsiveness.

2. Input Resolution:
   In `app/analyzer.py`, resizing the cropped face ROI to `160x160` decreases floating-point convolution operations by approximately 40% with minimal loss in demographic precision.

3. Background Suppression:
   Set `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` and `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"` to eliminate non-critical informational logs.

---

## Session Telemetry and Analytics

Upon exiting a session, aggregate telemetry can be exported into timestamped JSON files (`session_demographic_log_*.json`). The exported data includes:
- Total session duration and frame count
- Dominant emotion distribution percentages
- Stabilized age confidence intervals
- Detected demographic breakdown

---

## Troubleshooting

- Webcam Access Error:
  Ensure no other application (Zoom, Teams, Skype, or browser) is currently holding the camera handle. On Windows, check Privacy & Security > Camera permissions.

- Low Light or Shadows:
  The built-in CLAHE preprocessor automatically enhances contrast in underexposed frames. Ensure the face is generally oriented toward the lens.

- Windows Character Encoding:
  On non-English Windows codepages (e.g. cp1256), standard output is automatically reconfigured to UTF-8 to prevent console logging exceptions.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
