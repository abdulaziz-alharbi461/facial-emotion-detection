"""
Production Dual-Tier Demographic & Emotion Intelligence Pipeline (v3.5)
========================================================================
Architecture:
- Single Unified Bounding Box with EMA Temporal Coordinate Smoothing
- Tier 1 (High Priority): Fast Emotion Worker (~20ms latency)
- Tier 2 (Low Priority): Background Demographic Worker (~3.5s interval)
- Thread-Safe Atomic State Store (AppState)
"""

import sys
import time
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from app.camera import ThreadedWebcam
from app.analytics import DemographicAnalyticsTracker
from app.ui import draw_hud
from main import AppState, FastEmotionWorker, BackgroundDemographicWorker, draw_unified_face_hud


def main():
    print("=" * 70)
    print("Starting Stabilized Production Engine (v3.5)...")
    print("=" * 70)

    camera = ThreadedWebcam(src=0).start()
    time.sleep(0.4)

    app_state = AppState()
    fast_emotion = FastEmotionWorker(app_state, min_interval_sec=0.04)
    slow_demographics = BackgroundDemographicWorker(app_state, interval_sec=3.5)
    analytics = DemographicAnalyticsTracker()

    face_cascade = None
    if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            face_cascade = None

    window_name = "AI Real-Time Emotion & Demographics (Pro)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    prev_time = time.time()
    frame_count = 0
    smoothed_bbox = None
    face_missing_counter = 0

    try:
        while True:
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            # 1. Face Detection
            faces = []
            if face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected = face_cascade.detectMultiScale(gray, scaleFactor=1.18, minNeighbors=6, minSize=(80, 80))
                    if len(detected) > 0:
                        faces = detected
                except Exception:
                    faces = []

            # 2. Single Unified Bounding Box & Coordinate Smoothing
            if len(faces) > 0:
                face_missing_counter = 0
                primary_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = primary_face

                if smoothed_bbox is None:
                    smoothed_bbox = [float(x), float(y), float(w), float(h)]
                else:
                    beta = 0.35
                    smoothed_bbox[0] = beta * x + (1.0 - beta) * smoothed_bbox[0]
                    smoothed_bbox[1] = beta * y + (1.0 - beta) * smoothed_bbox[1]
                    smoothed_bbox[2] = beta * w + (1.0 - beta) * smoothed_bbox[2]
                    smoothed_bbox[3] = beta * h + (1.0 - beta) * smoothed_bbox[3]

                rx, ry, rw, rh = tuple(map(int, smoothed_bbox))
                pad = 12
                y1 = max(0, ry - pad)
                y2 = min(frame.shape[0], ry + rh + pad)
                x1 = max(0, rx - pad)
                x2 = min(frame.shape[1], rx + rw + pad)
                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size > 0:
                    fast_emotion.request(face_roi)
                    slow_demographics.request(face_roi)
            else:
                face_missing_counter += 1
                if face_missing_counter > 5:
                    smoothed_bbox = None

            # 3. Read Atomic State
            snapshot = app_state.get_snapshot()

            if frame_count % 15 == 0:
                analytics.record_sample(snapshot)

            # 4. Render Unified Box & Badges
            if smoothed_bbox is not None:
                render_box = tuple(map(int, smoothed_bbox))
                draw_unified_face_hud(frame, render_box, snapshot)
            else:
                cv2.putText(frame, "No face detected", (30, frame.shape[0] - 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 165, 255), 2, cv2.LINE_AA)

            draw_hud(frame, fps, snapshot)

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
        camera.stop()
        fast_emotion.stop()
        slow_demographics.stop()
        cv2.destroyAllWindows()
        analytics.export_session_log()
        print("[INFO] Production engine shutdown complete.")


if __name__ == "__main__":
    main()
