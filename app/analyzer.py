"""
Asynchronous Deep Learning Demographic & Emotion Inference Engine
Performs concurrent inference for:
- Facial Emotion Recognition
- Apparent Age Estimation
- Gender Classification
- Broad Geographic / Ethnic Demographic Estimation

Features:
- Decoupled worker thread pool with non-blocking queue
- Adaptive action selection based on available model weights
- CLAHE low-light enhancement
- Windows UTF-8 and SSL certificate auto-patching
"""

import os
import sys
import ssl
import queue
import threading
import cv2
import numpy as np

# 1. Windows console encoding safety
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

# 2. SSL certificate verification patch for DeepFace automated weight downloads
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

from deepface import DeepFace


def apply_clahe_preprocessing(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L-channel
    in LAB color space to enhance low-light webcam frames while preserving color balance.
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a_channel, b_channel))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception:
        return image


class AsyncDemographicAnalyzer:
    def __init__(self, actions=None, max_queue_size=2):
        """
        :param actions: List of actions, e.g. ['emotion', 'age', 'gender', 'race'].
                        Defaults to all 4 demographic & emotion analyzers.
        """
        self.requested_actions = actions or ['emotion', 'age', 'gender', 'race']
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.stopped = False

        # Inspect available model weights in DeepFace directory
        self.active_actions = self._detect_supported_actions()
        print(f"[ANALYZER] Active inference actions: {self.active_actions}")

        # Start asynchronous worker thread
        self.worker_thread = threading.Thread(
            target=self._worker,
            name="DeepFaceDemographicWorker",
            daemon=True
        )
        self.worker_thread.start()

    def _detect_supported_actions(self):
        """
        Checks which model weights are installed or prepares them for on-demand execution.
        """
        home = os.path.expanduser("~")
        weights_dir = os.path.join(home, ".deepface", "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # Map actions to model weight filenames
        action_weights = {
            'emotion': 'facial_expression_model_weights.h5',
            'age': 'age_model_weights.h5',
            'gender': 'gender_model_weights.h5',
            'race': 'race_model_single_batch.h5'
        }

        active = []
        for action in self.requested_actions:
            weight_file = action_weights.get(action)
            if weight_file:
                weight_path = os.path.join(weights_dir, weight_file)
                # If weight exists or is emotion, enable it
                if os.path.exists(weight_path) and os.path.getsize(weight_path) > 100000:
                    active.append(action)
                elif action == 'emotion':
                    active.append(action)
                else:
                    # Mark that weights are needed
                    print(f"[ANALYZER NOTICE] Weight '{weight_file}' for action '{action}' will be fetched or can be pre-downloaded via download_demographics.py.")
                    active.append(action)

        return active if active else ['emotion']

    def submit_roi(self, face_roi, bbox_coords, face_id=0):
        """
        Submits a face Region of Interest for non-blocking asynchronous analysis.
        """
        if not self.input_queue.full():
            try:
                self.input_queue.put_nowait((face_roi.copy(), bbox_coords, face_id))
            except queue.Full:
                pass

    def get_latest_results(self):
        """
        Returns all completed inference payloads currently available in the output queue.
        """
        results = []
        while not self.output_queue.empty():
            try:
                results.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _worker(self):
        """
        Dedicated inference loop running DeepFace.analyze across active actions.
        """
        while not self.stopped:
            try:
                item = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            face_roi, bbox_coords, face_id = item
            if face_roi is None or face_roi.size == 0:
                self.input_queue.task_done()
                continue

            # Apply low-light & contrast enhancement
            enhanced_roi = apply_clahe_preprocessing(face_roi)

            try:
                analysis = DeepFace.analyze(
                    img_path=enhanced_roi,
                    actions=self.active_actions,
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(analysis, list) and len(analysis) > 0:
                    res = analysis[0]
                else:
                    res = analysis

                # Extract emotion metrics
                raw_emotion_scores = res.get('emotion', {})
                dominant_emotion = res.get('dominant_emotion', 'neutral')

                # Extract age metrics
                apparent_age = res.get('age', None)

                # Extract gender metrics
                dominant_gender = res.get('dominant_gender', None)
                raw_gender_scores = res.get('gender', {})

                # Extract demographic / race metrics (used as regional/geographic proxy)
                dominant_race = res.get('dominant_race', None)
                raw_race_scores = res.get('race', {})

                result_payload = {
                    'face_id': face_id,
                    'bbox': bbox_coords,
                    'dominant': dominant_emotion,
                    'scores': raw_emotion_scores,
                    'raw_age': apparent_age,
                    'dominant_gender': dominant_gender,
                    'raw_gender': raw_gender_scores,
                    'dominant_race': dominant_race,
                    'raw_race': raw_race_scores
                }

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.output_queue.put_nowait(result_payload)

            except Exception as e:
                # If a model fails or download times out, attempt fallback to emotion only
                if len(self.active_actions) > 1 and any(act in self.active_actions for act in ['age', 'gender', 'race']):
                    try:
                        fallback_res = DeepFace.analyze(
                            img_path=enhanced_roi,
                            actions=['emotion'],
                            enforce_detection=False,
                            silent=True
                        )
                        res = fallback_res[0] if isinstance(fallback_res, list) else fallback_res
                        result_payload = {
                            'face_id': face_id,
                            'bbox': bbox_coords,
                            'dominant': res.get('dominant_emotion', 'neutral'),
                            'scores': res.get('emotion', {}),
                            'raw_age': None,
                            'dominant_gender': 'Analyzing...',
                            'raw_gender': {},
                            'dominant_race': 'Analyzing...',
                            'raw_race': {}
                        }
                        if not self.output_queue.full():
                            self.output_queue.put_nowait(result_payload)
                    except Exception:
                        pass
            finally:
                self.input_queue.task_done()

    def stop(self):
        self.stopped = True
        print("[ANALYZER] Async demographic & emotion worker pool shutdown.")


# Compatibility alias
AsyncEmotionAnalyzer = AsyncDemographicAnalyzer
