"""
Temporal Smoothing Engine for Demographic and Emotional Attributes
Provides Exponential Moving Average (EMA) and Sliding Window Majority Voting
for Emotion, Age, Gender, and Geographic Demographics to eliminate jitter across video frames.
"""

from collections import deque, Counter
import numpy as np


class TemporalSmoother:
    def __init__(self, alpha_emotion=0.35, alpha_age=0.15, window_size=9):
        """
        :param alpha_emotion: EMA smoothing factor for emotion (0 < alpha <= 1).
        :param alpha_age: EMA smoothing factor for age (lower = more stable).
        :param window_size: Size of rolling history window for voting.
        """
        self.alpha_emotion = alpha_emotion
        self.alpha_age = alpha_age
        self.window_size = window_size

        # Emotion history
        self.emotion_history = deque(maxlen=window_size)
        self.smoothed_emotion_probs = {}
        self.emotions_list = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # Demographic history
        self.age_history = deque(maxlen=window_size)
        self.smoothed_age = None

        self.gender_history = deque(maxlen=window_size)
        self.smoothed_gender_probs = {"Man": 0.5, "Woman": 0.5}

        self.race_history = deque(maxlen=window_size)
        self.smoothed_race_probs = {}
        self.race_list = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]

    def update_emotion(self, raw_emotion_scores):
        if not raw_emotion_scores:
            return "neutral", 0.0, self.smoothed_emotion_probs

        total = sum(raw_emotion_scores.values()) or 1.0
        norm_scores = {k: v / total for k, v in raw_emotion_scores.items()}

        if not self.smoothed_emotion_probs:
            for emotion in self.emotions_list:
                self.smoothed_emotion_probs[emotion] = norm_scores.get(emotion, 0.0)
        else:
            for emotion in self.emotions_list:
                cur = norm_scores.get(emotion, 0.0)
                prev = self.smoothed_emotion_probs.get(emotion, 0.0)
                self.smoothed_emotion_probs[emotion] = (self.alpha_emotion * cur) + ((1 - self.alpha_emotion) * prev)

        dominant_emotion = max(self.smoothed_emotion_probs, key=self.smoothed_emotion_probs.get)
        confidence = self.smoothed_emotion_probs[dominant_emotion] * 100.0
        self.emotion_history.append(dominant_emotion)

        return dominant_emotion, confidence, self.smoothed_emotion_probs

    def update_age(self, raw_age):
        if raw_age is None:
            return int(self.smoothed_age) if self.smoothed_age is not None else 25

        self.age_history.append(raw_age)

        # Use median of recent frames to reject outliers
        median_age = float(np.median(list(self.age_history)))

        if self.smoothed_age is None:
            self.smoothed_age = median_age
        else:
            self.smoothed_age = (self.alpha_age * median_age) + ((1.0 - self.alpha_age) * self.smoothed_age)

        return int(round(self.smoothed_age))

    def update_gender(self, raw_gender_dict, dominant_gender=None):
        if not raw_gender_dict:
            dominant = dominant_gender or "Unknown"
            return dominant, 50.0

        w_prob = raw_gender_dict.get("Woman", 50.0) / 100.0
        m_prob = raw_gender_dict.get("Man", 50.0) / 100.0

        # Update EMA
        alpha = 0.25
        self.smoothed_gender_probs["Woman"] = alpha * w_prob + (1 - alpha) * self.smoothed_gender_probs.get("Woman", 0.5)
        self.smoothed_gender_probs["Man"] = alpha * m_prob + (1 - alpha) * self.smoothed_gender_probs.get("Man", 0.5)

        dominant = "Woman" if self.smoothed_gender_probs["Woman"] >= self.smoothed_gender_probs["Man"] else "Man"
        confidence = self.smoothed_gender_probs[dominant] * 100.0
        self.gender_history.append(dominant)

        return dominant, confidence

    def update_race(self, raw_race_scores, dominant_race=None):
        if not raw_race_scores:
            dominant = dominant_race or "Global"
            return dominant.capitalize(), 50.0, self.smoothed_race_probs

        total = sum(raw_race_scores.values()) or 1.0
        norm_scores = {k: v / total for k, v in raw_race_scores.items()}

        alpha = 0.25
        if not self.smoothed_race_probs:
            for r in self.race_list:
                self.smoothed_race_probs[r] = norm_scores.get(r, 0.0)
        else:
            for r in self.race_list:
                cur = norm_scores.get(r, 0.0)
                prev = self.smoothed_race_probs.get(r, 0.0)
                self.smoothed_race_probs[r] = alpha * cur + (1 - alpha) * prev

        dominant = max(self.smoothed_race_probs, key=self.smoothed_race_probs.get)
        confidence = self.smoothed_race_probs[dominant] * 100.0
        self.race_history.append(dominant)

        # Format label cleanly (e.g. 'middle eastern' -> 'Middle Eastern')
        formatted_dominant = " ".join([word.capitalize() for word in dominant.split()])
        return formatted_dominant, confidence, self.smoothed_race_probs

    def update_all(self, result_payload):
        """
        Takes raw DeepFace dictionary result and returns smoothed demographic & emotional profile.
        """
        raw_emotion = result_payload.get("scores", {})
        raw_age = result_payload.get("raw_age")
        raw_gender = result_payload.get("raw_gender", {})
        raw_race = result_payload.get("raw_race", {})

        dom_emotion, conf_emotion, smooth_emotions = self.update_emotion(raw_emotion)
        smooth_age = self.update_age(raw_age)
        dom_gender, conf_gender = self.update_gender(raw_gender, result_payload.get("dominant_gender"))
        dom_race, conf_race, smooth_race = self.update_race(raw_race, result_payload.get("dominant_race"))

        return {
            "emotion": dom_emotion,
            "emotion_confidence": conf_emotion,
            "emotion_spectrum": smooth_emotions,
            "age": smooth_age,
            "gender": dom_gender,
            "gender_confidence": conf_gender,
            "demographic": dom_race,
            "demographic_confidence": conf_race,
            "demographic_spectrum": smooth_race
        }

    def reset(self):
        self.emotion_history.clear()
        self.smoothed_emotion_probs.clear()
        self.age_history.clear()
        self.smoothed_age = None
        self.gender_history.clear()
        self.smoothed_gender_probs = {"Man": 0.5, "Woman": 0.5}
        self.race_history.clear()
        self.smoothed_race_probs.clear()
