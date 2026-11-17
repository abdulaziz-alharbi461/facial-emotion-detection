"""
Demographic & Emotional Analytics Session Engine
Logs real-time temporal estimates for Age, Gender, Region/Demographics, and Emotion,
and exports structured intelligence reports upon session exit.
"""

import json
from collections import Counter
from datetime import datetime
import numpy as np


class DemographicAnalyticsTracker:
    def __init__(self):
        self.session_start = datetime.now()
        self.log_entries = []
        self.emotion_counts = Counter()
        self.gender_counts = Counter()
        self.demographic_counts = Counter()
        self.age_records = []

    def record_sample(self, demographic_profile):
        """
        Records a smoothed sample payload.
        :param demographic_profile: Dictionary containing smoothed attributes.
        """
        timestamp = datetime.now().isoformat()
        emotion = demographic_profile.get("emotion", "neutral")
        gender = demographic_profile.get("gender", "Unknown")
        demographic = demographic_profile.get("demographic", "Global")
        age = demographic_profile.get("age")

        self.emotion_counts[emotion] += 1
        if gender and gender != "Analyzing...":
            self.gender_counts[gender] += 1
        if demographic and demographic != "Analyzing...":
            self.demographic_counts[demographic] += 1
        if age is not None and isinstance(age, (int, float)):
            self.age_records.append(float(age))

        entry = {
            "timestamp": timestamp,
            "apparent_age": age,
            "gender": gender,
            "gender_confidence": round(demographic_profile.get("gender_confidence", 0.0), 1),
            "geographic_demographic": demographic,
            "demographic_confidence": round(demographic_profile.get("demographic_confidence", 0.0), 1),
            "dominant_emotion": emotion,
            "emotion_confidence": round(demographic_profile.get("emotion_confidence", 0.0), 1),
            "emotion_spectrum": {k: round(v, 3) for k, v in demographic_profile.get("emotion_spectrum", {}).items()}
        }
        self.log_entries.append(entry)

    def get_session_summary(self):
        total_samples = sum(self.emotion_counts.values()) or 1
        duration_sec = (datetime.now() - self.session_start).total_seconds()

        # Emotion distribution percentages
        emotion_pct = {
            emo: round((count / total_samples) * 100.0, 1)
            for emo, count in self.emotion_counts.items()
        }

        # Gender distribution percentages
        total_genders = sum(self.gender_counts.values()) or 1
        gender_pct = {
            g: round((count / total_genders) * 100.0, 1)
            for g, count in self.gender_counts.items()
        }

        # Demographic distribution percentages
        total_demographics = sum(self.demographic_counts.values()) or 1
        demographic_pct = {
            d: round((count / total_demographics) * 100.0, 1)
            for d, count in self.demographic_counts.items()
        }

        # Age statistics
        avg_age = round(float(np.mean(self.age_records)), 1) if self.age_records else None
        median_age = round(float(np.median(self.age_records)), 1) if self.age_records else None

        # Wellness index
        positive = emotion_pct.get('happy', 0.0) + emotion_pct.get('surprise', 0.0)
        negative = (emotion_pct.get('sad', 0.0) + emotion_pct.get('angry', 0.0) +
                    emotion_pct.get('fear', 0.0) + emotion_pct.get('disgust', 0.0))
        neutral = emotion_pct.get('neutral', 0.0)

        return {
            "start_time": self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration_sec, 1),
            "total_frames_analyzed": total_samples,
            "demographic_insights": {
                "average_estimated_age": avg_age,
                "median_estimated_age": median_age,
                "gender_breakdown_pct": gender_pct,
                "geographic_demographic_pct": demographic_pct
            },
            "emotional_insights": {
                "dominant_emotion_overall": self.emotion_counts.most_common(1)[0][0] if self.emotion_counts else "neutral",
                "emotion_breakdown_pct": emotion_pct,
                "wellness_index": {
                    "positive_pct": round(positive, 1),
                    "neutral_pct": round(neutral, 1),
                    "negative_pct": round(negative, 1)
                }
            }
        }

    def export_session_log(self, filepath_prefix="session_demographic_log"):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"{filepath_prefix}_{timestamp_str}.json"

        summary = self.get_session_summary()
        payload = {
            "summary": summary,
            "timeline": self.log_entries
        }

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[ANALYTICS] Demographic & emotion session log exported to {json_path}")
            return json_path
        except Exception as e:
            print(f"[ANALYTICS ERROR] Failed to export session log: {e}")
            return None


# Backward compatibility alias
MoodAnalyticsTracker = DemographicAnalyticsTracker
