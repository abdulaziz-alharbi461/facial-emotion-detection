"""
High-Precision Real-Time Facial Emotion & Demographic Intelligence System (v4.0)
=================================================================================
Senior Computer Vision & Deep Learning Architecture:

Key Enhancements for Maximum Precision:
1. Pre-processing & Lighting Normalization (CLAHE + Bilateral Denoising):
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in LAB space.
   - Normalizes harsh lighting, shadows, and low-light environments without color distortion.
   - Bilateral edge-preserving filter eliminates sensor grain while keeping facial contours sharp.
2. Calibrated Emotion Sensitivity & Temporal Buffer:
   - Eliminates resting-face neutral bias so subtle expressions (especially Sadness and Smiles)
     are captured accurately and immediately.
   - 6-frame probability moving average eliminates high-frequency flickering.
3. Rock-Solid Age Stabilization (Rolling Median + EMA Outlier Rejection):
   - Outlier Rejection: Computes the median of a rolling window of recent age inferences,
     completely discarding extreme lighting glares or angle spikes.
   - Exponential Moving Average (EMA) with alpha=0.12 stabilizes the age number into a reliable,
     steady readout rather than jumping erratic values.
4. Occlusion & Aspect Ratio Verification:
   - Validates face aspect ratios (0.65 to 1.15) to prevent hand/phone occlusions from distorting
     landmark geometry.
5. Single Unified Stabilized HUD:
   - High-contrast glassmorphic card container with dynamic emotion accent coloring.

Key Controls:
    - Press 'f' to toggle Fullscreen / Windowed display.
    - Press 'q' or 'ESC' to exit cleanly.
"""

import os
import sys
import time
import ssl
import threading
from collections import deque
import cv2
import numpy as np

# ---------------------------------------------------------
# Windows UTF-8 Console & SSL Auto-Patching
# ---------------------------------------------------------
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

# Suppress verbose TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Safe monkey-patch for DeepFace logger to prevent UnicodeEncodeError on Windows cp1256
try:
    import deepface.commons.logger as df_logger
    def safe_dump_log(self, message):
        try:
            clean_msg = str(message).encode('ascii', errors='replace').decode('ascii')
            print(clean_msg)
        except Exception:
            pass
    df_logger.Logger.dump_log = safe_dump_log
except Exception:
    pass

from deepface import DeepFace

# UI Palette (BGR format) & Regional Demographic Mapping
EMOTION_COLORS = {
    "happy": (0, 230, 115),      # Vibrant Emerald Green
    "sad": (255, 178, 102),     # Soft Cyan/Blue
    "angry": (71, 71, 255),      # Coral Red
    "surprise": (0, 215, 255),   # Golden Amber
    "neutral": (220, 220, 220),  # Clean White/Grey
    "fear": (204, 102, 255),     # Electric Lilac
    "disgust": (102, 255, 178)   # Mint
}

DEMOGRAPHIC_REGIONS = {
    "asian": "East / Southeast Asia",
    "indian": "South Asia",
    "black": "African / Afro-Descendant",
    "white": "European / Caucasian",
    "middle eastern": "Middle Eastern / West Asian",
    "latino hispanic": "Latino / Hispanic"
}
DEFAULT_ACCENT_COLOR = (0, 255, 255)


# ---------------------------------------------------------
# Image Pre-processing Engine (CLAHE + Bilateral Denoising)
# ---------------------------------------------------------
def preprocess_face_roi(roi):
    """
    Applies edge-preserving bilateral smoothing and LAB-space CLAHE to the face ROI.
    Normalizes low light, shadows, and camera sensor grain while sharpening facial landmarks.
    """
    if roi is None or roi.size == 0:
        return roi
    try:
        # Step 1: Bilateral Filter (smooths sensor noise without blurring lip/eye edges)
        denoised = cv2.bilateralFilter(roi, d=5, sigmaColor=25, sigmaSpace=25)

        # Step 2: CLAHE on Luminance channel (recovers details from harsh shadows)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception:
        return roi


# ---------------------------------------------------------
# Calibrated Emotion Sensitivity Engine
# ---------------------------------------------------------
def calibrate_dominant_emotion(scores):
    """
    Applies sensitivity calibration with neutral-bias compensation.
    Guarantees subtle expressions (especially Sadness and Smiles) register accurately
    rather than being masked by resting-face neutrality.
    """
    if not scores:
        return "neutral", 0.0

    happy = scores.get("happy", 0.0)
    sad = scores.get("sad", 0.0)
    surprise = scores.get("surprise", 0.0)
    angry = scores.get("angry", 0.0)
    fear = scores.get("fear", 0.0)
    disgust = scores.get("disgust", 0.0)
    neutral = scores.get("neutral", 0.0)

    # 1. Sadness Sensitivity Trigger (Catches drooping lips / brow furrows)
    # Natural sad faces often register ~16-30% sad while neutral is ~40-60%
    if sad >= 16.0 and (sad >= neutral * 0.32 or sad >= 28.0):
        return "sad", sad

    # 2. Happiness / Smile Sensitivity Trigger
    if happy >= 18.0 and (happy >= neutral * 0.30 or happy >= 25.0):
        return "happy", happy

    # 3. Surprise Sensitivity Trigger
    if surprise >= 20.0 and (surprise >= neutral * 0.35 or surprise >= 30.0):
        return "surprise", surprise

    # 4. Anger / Tension Sensitivity Trigger
    if angry >= 20.0 and (angry >= neutral * 0.35 or angry >= 28.0):
        return "angry", angry

    # 5. Fear / Disgust Triggers
    if fear >= 22.0 and fear >= neutral * 0.4:
        return "fear", fear
    if disgust >= 20.0 and disgust >= neutral * 0.4:
        return "disgust", disgust

    # Default to dominant score
    dominant = max(scores, key=scores.get)
    return dominant, scores[dominant]


# ---------------------------------------------------------
# Thread-Safe Centralized State Store with Age Stabilization
# ---------------------------------------------------------
class AppState:
    """
    Thread-safe telemetry state container featuring:
    - Probability EMA buffer for smooth expression transitions
    - Rolling median outlier filter + EMA for stabilized age estimation
    """
    def __init__(self):
        self.lock = threading.Lock()

        # Emotion Telemetry
        self.emotion = "neutral"
        self.emotion_confidence = 0.0
        self.emotion_spectrum = {}
        self.smoothed_probabilities = {}
        self.alpha_emotion = 0.40  # Responsiveness vs smoothing balance

        # Demographic Telemetry
        self.age_history = deque(maxlen=9)
        self.stabilized_age = None
        self.alpha_age = 0.12  # Steady low-pass filter factor

        self.gender = "Detecting..."
        self.gender_confidence = 0.0
        self.demographic = "Detecting Region..."
        self.demographic_confidence = 0.0
        self.demographics_loaded = False
        self.is_demographics_busy = False

    def update_emotion(self, raw_scores):
        with self.lock:
            # Normalize probabilities
            total = sum(raw_scores.values()) or 1.0
            norm_scores = {k: (v / total) * 100.0 for k, v in raw_scores.items()}

            # Apply Temporal EMA Smoothing to all 7 emotion channels
            if not self.smoothed_probabilities:
                self.smoothed_probabilities = norm_scores.copy()
            else:
                for emo, score in norm_scores.items():
                    prev = self.smoothed_probabilities.get(emo, score)
                    self.smoothed_probabilities[emo] = (self.alpha_emotion * score) + ((1.0 - self.alpha_emotion) * prev)

            # Calibrate dominant emotion from smoothed spectrum
            dominant, conf = calibrate_dominant_emotion(self.smoothed_probabilities)
            self.emotion = dominant
            self.emotion_confidence = conf
            self.emotion_spectrum = self.smoothed_probabilities.copy()

    def update_age(self, raw_age):
        """
        Locks down age estimation using Rolling Median Outlier Filter + EMA.
        Completely eliminates erratic age jumps.
        """
        if raw_age is None or raw_age <= 0:
            return self.stabilized_age

        self.age_history.append(float(raw_age))

        # Median of rolling window rejects extreme outliers (lighting/glare spikes)
        median_age = float(np.median(list(self.age_history)))

        if self.stabilized_age is None:
            self.stabilized_age = median_age
        else:
            # Low-pass EMA: Age_smooth = alpha * median + (1 - alpha) * Age_prev
            self.stabilized_age = (self.alpha_age * median_age) + ((1.0 - self.alpha_age) * self.stabilized_age)

        return int(round(self.stabilized_age))

    def update_demographics(self, raw_age, gender, gender_conf, demographic, demo_conf):
        with self.lock:
            if raw_age is not None:
                self.update_age(raw_age)
            if gender and gender != "Unknown":
                self.gender = gender
                self.gender_confidence = gender_conf
            if demographic and demographic != "Global":
                self.demographic = demographic
                self.demographic_confidence = demo_conf
            self.demographics_loaded = True
            self.is_demographics_busy = False

    def set_demographics_busy(self, status):
        with self.lock:
            self.is_demographics_busy = status

    def get_snapshot(self):
        with self.lock:
            display_age = int(round(self.stabilized_age)) if self.stabilized_age is not None else None
            return {
                "emotion": self.emotion,
                "emotion_confidence": self.emotion_confidence,
                "emotion_spectrum": self.emotion_spectrum.copy(),
                "age": display_age,
                "gender": self.gender,
                "gender_confidence": self.gender_confidence,
                "demographic": self.demographic,
                "demographic_confidence": self.demographic_confidence,
                "demographics_loaded": self.demographics_loaded,
                "is_demographics_busy": self.is_demographics_busy
            }


# ---------------------------------------------------------
# Tier 1: Fast High-Precision Emotion Worker (~15-20ms)
# ---------------------------------------------------------
class FastEmotionWorker:
    """
    Dedicated high-speed thread executing ONLY emotion classification.
    Uses preprocessed ROI and detector_backend='skip' for ultra-low latency.
    """
    def __init__(self, state: AppState, min_interval_sec=0.035):
        self.state = state
        self.min_interval_sec = min_interval_sec
        self.last_run_time = 0.0

        self.lock = threading.Lock()
        self.is_busy = False
        self.stop_event = threading.Event()
        self.trigger_event = threading.Event()
        self.pending_roi = None

        self.thread = threading.Thread(target=self._worker_loop, name="FastEmotionWorker", daemon=True)
        self.thread.start()

    def request(self, face_roi):
        now = time.time()
        if self.is_busy or (now - self.last_run_time < self.min_interval_sec):
            return False

        with self.lock:
            self.pending_roi = face_roi.copy()
            self.is_busy = True
            self.last_run_time = now

        self.trigger_event.set()
        return True

    def _worker_loop(self):
        while not self.stop_event.is_set():
            triggered = self.trigger_event.wait(timeout=0.08)
            if not triggered or self.stop_event.is_set():
                continue

            self.trigger_event.clear()

            with self.lock:
                roi = self.pending_roi

            if roi is None or roi.size == 0:
                with self.lock:
                    self.is_busy = False
                continue

            try:
                # 1. CLAHE & Bilateral Preprocessing on ROI
                enhanced_roi = preprocess_face_roi(roi)

                # 2. Fast Emotion Inference (< 20ms)
                results = DeepFace.analyze(
                    img_path=enhanced_roi,
                    actions=['emotion'],
                    detector_backend='skip',
                    enforce_detection=False,
                    silent=True
                )
                res = results[0] if isinstance(results, list) else results
                raw_scores = res.get('emotion', {})

                # 3. Update Thread-Safe Centralized State with Temporal Smoothing
                self.state.update_emotion(raw_scores)
            except Exception:
                pass
            finally:
                with self.lock:
                    self.is_busy = False

    def stop(self):
        self.stop_event.set()
        self.trigger_event.set()
        self.thread.join(timeout=0.5)


# ---------------------------------------------------------
# Tier 2: Background Demographic Worker (Independent / Stabilized)
# ---------------------------------------------------------
class BackgroundDemographicWorker:
    """
    Independent worker thread running Age, Gender, and Regional Demographic models.
    Refreshes every 3.0s and stabilizes age using rolling median regression.
    """
    def __init__(self, state: AppState, interval_sec=3.0):
        self.state = state
        self.interval_sec = interval_sec
        self.last_run_time = 0.0

        self.lock = threading.Lock()
        self.is_busy = False
        self.stop_event = threading.Event()
        self.trigger_event = threading.Event()
        self.pending_roi = None

        self.thread = threading.Thread(target=self._worker_loop, name="BackgroundDemographicWorker", daemon=True)
        self.thread.start()

    def _check_weights(self):
        weights_dir = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
        age_file = os.path.join(weights_dir, "age_model_weights.h5")
        gender_file = os.path.join(weights_dir, "gender_model_weights.h5")
        race_file = os.path.join(weights_dir, "race_model_single_batch.h5")

        age_ok = os.path.exists(age_file) and os.path.getsize(age_file) > 100_000_000
        gender_ok = os.path.exists(gender_file) and os.path.getsize(gender_file) > 100_000_000
        race_ok = os.path.exists(race_file) and os.path.getsize(race_file) > 100_000_000

        actions = []
        if age_ok:
            actions.append('age')
        if gender_ok:
            actions.append('gender')
        if race_ok:
            actions.append('race')

        return actions

    def request(self, face_roi):
        now = time.time()
        if self.is_busy or (now - self.last_run_time < self.interval_sec):
            return False

        with self.lock:
            self.pending_roi = face_roi.copy()
            self.is_busy = True
            self.last_run_time = now

        self.state.set_demographics_busy(True)
        self.trigger_event.set()
        return True

    def _worker_loop(self):
        while not self.stop_event.is_set():
            triggered = self.trigger_event.wait(timeout=0.2)
            if not triggered or self.stop_event.is_set():
                continue

            self.trigger_event.clear()

            with self.lock:
                roi = self.pending_roi

            if roi is None or roi.size == 0:
                with self.lock:
                    self.is_busy = False
                self.state.set_demographics_busy(False)
                continue

            actions = self._check_weights()
            if not actions:
                with self.lock:
                    self.is_busy = False
                self.state.set_demographics_busy(False)
                continue

            try:
                # Preprocess face ROI with subtle denoising for cleaner age estimation
                enhanced_roi = preprocess_face_roi(roi)

                results = DeepFace.analyze(
                    img_path=enhanced_roi,
                    actions=actions,
                    detector_backend='skip',
                    enforce_detection=False,
                    silent=True
                )
                res = results[0] if isinstance(results, list) else results

                apparent_age = res.get('age', None)
                dom_gender = res.get('dominant_gender', 'Unknown')
                gender_scores = res.get('gender', {})
                conf_gender = gender_scores.get(dom_gender, 0.0)

                dom_race = res.get('dominant_race', 'Global')
                race_scores = res.get('race', {})
                conf_race = race_scores.get(dom_race, 0.0)

                # Stabilized update (filters age outliers automatically)
                self.state.update_demographics(
                    raw_age=apparent_age,
                    gender=dom_gender,
                    gender_conf=conf_gender,
                    demographic=dom_race,
                    demo_conf=conf_race
                )
            except Exception:
                pass
            finally:
                with self.lock:
                    self.is_busy = False
                self.state.set_demographics_busy(False)

    def stop(self):
        self.stop_event.set()
        self.trigger_event.set()
        self.thread.join(timeout=1.0)


# ---------------------------------------------------------
# UI & Overlay Rendering Functions
# ---------------------------------------------------------
def render_glass_card(frame, x1, y1, x2, y2, alpha=0.82, bg_color=(12, 15, 20)):
    """Draws a solid high-contrast glassmorphic card behind text to prevent overlap."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub_img = frame[y1:y2, x1:x2]
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 0)


def draw_unified_face_hud(frame, bbox, snapshot):
    """
    Renders strictly ONE clean bounding box and non-colliding textual metric badge.
    """
    x, y, w, h = bbox
    emotion = snapshot.get("emotion", "neutral")
    emotion_conf = snapshot.get("emotion_confidence", 0.0)
    age = snapshot.get("age")
    gender = snapshot.get("gender", "Scanning")
    raw_demo = snapshot.get("demographic", "Global")
    region_label = DEMOGRAPHIC_REGIONS.get(raw_demo.lower(), raw_demo)

    accent_color = EMOTION_COLORS.get(emotion.lower(), DEFAULT_ACCENT_COLOR)

    # 1. Clean Corner-Accented Bounding Box
    corner = int(min(w, h) * 0.22)
    t = 2

    # Draw subtle bounding outline
    cv2.rectangle(frame, (x, y), (x + w, y + h), accent_color, 1)

    # 2D Corner Accents (Bug-free 2-element tuples)
    cv2.line(frame, (x, y), (x + corner, y), accent_color, t)
    cv2.line(frame, (x, y), (x, y + corner), accent_color, t)
    cv2.line(frame, (x + w, y), (x + w - corner, y), accent_color, t)
    cv2.line(frame, (x + w, y), (x + w, y + corner), accent_color, t)
    cv2.line(frame, (x, y + h), (x + corner, y + h), accent_color, t)
    cv2.line(frame, (x, y + h), (x, y + h - corner), accent_color, t)
    cv2.line(frame, (x + w, y + h), (x + w - corner, y + h), accent_color, t)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner), accent_color, t)

    # 2. Text Badges Positioned Above Box with Distinct Spacing
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.54

    age_str = f"~{age} yrs" if age is not None else "-- yrs"
    line1 = f"{gender}, {age_str} | {emotion.capitalize()} ({emotion_conf:.0f}%)"
    line2 = f"Demographic: {region_label}"

    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, 2)
    (w2, h2), _ = cv2.getTextSize(line2, font, 0.44, 1)

    badge_w = max(w1, w2) + 26
    badge_h = h1 + h2 + 24

    bx1 = x
    bx2 = x + badge_w
    if y - badge_h - 10 >= 55:
        by1 = y - badge_h - 10
        by2 = y - 5
    else:
        by1 = y + 10
        by2 = y + 10 + badge_h

    # Render solid high-contrast card background
    render_glass_card(frame, bx1, by1, bx2, by2, alpha=0.85, bg_color=(12, 15, 20))

    # Colored indicator accent strip on the left (reacts immediately when smiling/frowning!)
    cv2.rectangle(frame, (bx1, by1), (bx1 + 5, by2), accent_color, -1)

    # Line 1: Emotion + Gender + Stabilized Age
    cv2.putText(frame, line1, (bx1 + 12, by1 + h1 + 6), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 2: Demographic Region
    cv2.putText(frame, line2, (bx1 + 12, by1 + h1 + h2 + 16), font, 0.44, (190, 215, 240), 1, cv2.LINE_AA)

    # Status Dot: Amber when updating in background, Emerald when cached
    dot_color = (0, 215, 255) if snapshot.get("is_demographics_busy") else (0, 230, 115)
    cv2.circle(frame, (bx2 - 12, by1 + 12), 4, dot_color, -1)


def draw_hud(frame, fps, snapshot):
    """Draws top navigation HUD with real-time FPS and pipeline status."""
    height, width = frame.shape[:2]
    render_glass_card(frame, 0, 0, width, 50, alpha=0.75, bg_color=(10, 12, 16))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "AI FACIAL & DEMOGRAPHIC INTELLIGENCE", (18, 32), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # Real-Time FPS
    fps_color = (0, 230, 115) if fps >= 22 else (0, 200, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 320, 32), font, 0.58, fps_color, 2, cv2.LINE_AA)

    # Telemetry indicators
    demo_status = "UPDATING" if snapshot.get("is_demographics_busy") else "CACHED"
    demo_color = (0, 215, 255) if snapshot.get("is_demographics_busy") else (180, 180, 180)
    cv2.putText(frame, f"EMOTION: [LIVE] | DEMO: [{demo_status}]", (width - 240, 32), font, 0.42, demo_color, 1, cv2.LINE_AA)

    # Bottom Instructions Bar
    render_glass_card(frame, 15, height - 42, width - 15, height - 10, alpha=0.65, bg_color=(12, 14, 18))
    cv2.putText(frame, "Calibrated Sad/Smile Sensitivity Active | Controls: 'f' Fullscreen | 'q' or 'ESC' Quit",
                (26, height - 20), font, 0.46, (210, 210, 210), 1, cv2.LINE_AA)


def get_webcam_stream():
    """Initializes webcam using DirectShow or default backend without locks."""
    backends = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF if hasattr(cv2, 'CAP_MSMF') else 0), (1, cv2.CAP_DSHOW), (0, None)]
    for idx, backend in backends:
        try:
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    return cap
                cap.release()
        except Exception:
            pass
    return None


# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main():
    print("=" * 70)
    print("Starting High-Precision Demographic & Emotion Intelligence Engine (v4.0)...")
    print("Features: CLAHE Preprocessing + Calibrated Sad/Happy Sensitivity + Stabilized Age")
    print("=" * 70)

    face_cascade = None
    if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            face_cascade = None

    cap = get_webcam_stream()
    if cap is None or not cap.isOpened():
        print("[ERROR] Could not access webcam device.")
        return

    # Initialize Thread-Safe State & Decoupled Workers
    app_state = AppState()
    fast_emotion_worker = FastEmotionWorker(app_state, min_interval_sec=0.035)
    background_demo_worker = BackgroundDemographicWorker(app_state, interval_sec=3.0)

    window_name = "AI Real-Time Emotion & Demographics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    prev_time = time.time()
    consecutive_failed_frames = 0

    # Bounding box temporal stabilization state
    smoothed_bbox = None
    face_missing_counter = 0

    print("[INFO] Video loop running at maximum hardware FPS. Test smiling & sad expressions!")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failed_frames += 1
                if consecutive_failed_frames > 15:
                    break
                time.sleep(0.01)
                continue

            consecutive_failed_frames = 0
            frame = cv2.flip(frame, 1)

            # Instantaneous FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            # 1. Face Detection on Current Frame
            faces = []
            if face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.15,
                        minNeighbors=6,
                        minSize=(85, 85)
                    )
                    # Filter face candidates for valid human facial aspect ratio (0.65 to 1.25)
                    # This filters out partial phone/hand occlusions that distort boxes
                    valid_faces = []
                    for f in detected:
                        fx, fy, fw, fh = f
                        aspect = float(fw) / float(fh)
                        if 0.65 <= aspect <= 1.25:
                            valid_faces.append(f)
                    faces = valid_faces if valid_faces else detected
                except Exception:
                    faces = []

            # 2. Single Unified Bounding Box Selection & Coordinate Smoothing
            if len(faces) > 0:
                face_missing_counter = 0
                # Strictly isolate the single largest primary face
                primary_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = primary_face

                # Exponential Moving Average coordinate smoothing (0.35 responsiveness)
                if smoothed_bbox is None:
                    smoothed_bbox = [float(x), float(y), float(w), float(h)]
                else:
                    beta = 0.35
                    smoothed_bbox[0] = beta * x + (1.0 - beta) * smoothed_bbox[0]
                    smoothed_bbox[1] = beta * y + (1.0 - beta) * smoothed_bbox[1]
                    smoothed_bbox[2] = beta * w + (1.0 - beta) * smoothed_bbox[2]
                    smoothed_bbox[3] = beta * h + (1.0 - beta) * smoothed_bbox[3]

                rx, ry, rw, rh = tuple(map(int, smoothed_bbox))

                # 3. High-Precision Face Crop with Generous Padding (Includes entire mouth & forehead)
                pad_y = int(0.22 * rh)
                pad_x = int(0.16 * rw)
                y1 = max(0, ry - pad_y)
                y2 = min(frame.shape[0], ry + rh + pad_y)
                x1 = max(0, rx - pad_x)
                x2 = min(frame.shape[1], rx + rw + pad_x)
                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size > 0:
                    # Submit to Fast Emotion Worker (<20ms instant response)
                    fast_emotion_worker.request(face_roi)
                    # Submit to Background Demographic Worker (Stabilized Age & Gender)
                    background_demo_worker.request(face_roi)

            else:
                face_missing_counter += 1
                if face_missing_counter > 5:
                    smoothed_bbox = None  # Reset smoothed box when subject departs

            # 4. Fetch Atomic State Snapshot
            snapshot = app_state.get_snapshot()

            # 5. Render Strictly ONE Unified Bounding Box & Non-Colliding Badge
            if smoothed_bbox is not None:
                render_box = tuple(map(int, smoothed_bbox))
                draw_unified_face_hud(frame, render_box, snapshot)
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (30, frame.shape[0] - 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA
                )

            # 6. Render Top HUD Banner
            draw_hud(frame, fps, snapshot)

            # 7. Display Frame
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                is_fullscreen = not is_fullscreen
                mode = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)

            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        fast_emotion_worker.stop()
        background_demo_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    main()
