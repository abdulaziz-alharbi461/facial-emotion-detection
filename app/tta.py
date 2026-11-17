"""
Test-Time Augmentation (TTA) & Prediction Filtering Engine
Enhances inference accuracy by averaging predictions across original and horizontally flipped face ROIs.
"""

import cv2
import numpy as np
from deepface import DeepFace


class TTAInferenceEngine:
    def __init__(self, confidence_threshold=35.0):
        self.confidence_threshold = confidence_threshold

    def analyze_with_tta(self, face_roi):
        """
        Executes TTA: computes DeepFace emotion probabilities for both original
        and horizontally mirrored images, averaging the score vectors.
        """
        if face_roi is None or face_roi.size == 0:
            return "neutral", 0.0, {}

        # Augmentation 1: Original Face ROI
        res_orig = DeepFace.analyze(img_path=face_roi, actions=['emotion'], enforce_detection=False, silent=True)
        scores_orig = (res_orig[0] if isinstance(res_orig, list) else res_orig).get('emotion', {})

        # Augmentation 2: Horizontally Mirrored Face ROI
        flipped_roi = cv2.flip(face_roi, 1)
        res_flip = DeepFace.analyze(img_path=flipped_roi, actions=['emotion'], enforce_detection=False, silent=True)
        scores_flip = (res_flip[0] if isinstance(res_flip, list) else res_flip).get('emotion', {})

        # Ensemble Average across augmented views
        emotions = set(scores_orig.keys()).union(scores_flip.keys())
        averaged_scores = {}
        for emotion in emotions:
            val1 = scores_orig.get(emotion, 0.0)
            val2 = scores_flip.get(emotion, 0.0)
            averaged_scores[emotion] = (val1 + val2) / 2.0

        dominant_emotion = max(averaged_scores, key=averaged_scores.get)
        confidence = averaged_scores[dominant_emotion]

        # Low-confidence thresholding
        if confidence < self.confidence_threshold:
            dominant_emotion = "neutral"

        return dominant_emotion, confidence, averaged_scores
