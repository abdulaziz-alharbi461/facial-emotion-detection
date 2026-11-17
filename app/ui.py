"""
Professional UI & HUD Engine for Demographic & Emotion Intelligence
Renders:
- Clean bounding boxes with modern HUD corner accents
- Multi-metric badge overlays: [Gender], [Age] yrs, [Dominant Emotion], [Demographic/Region]
- Live Demographic Profile card (Age, Gender, Region/Race spectrum)
- Live Emotion Spectrum bar chart
"""

import cv2
import numpy as np

EMOTION_COLORS = {
    "happy": (0, 230, 115),      # Vibrant Emerald Green
    "sad": (255, 178, 102),     # Soft Blue/Cyan
    "angry": (71, 71, 255),      # Bright Coral Red
    "surprise": (0, 215, 255),   # Golden Amber
    "neutral": (220, 220, 220),  # Crisp Off-White/Grey
    "fear": (204, 102, 255),     # Electric Lilac
    "disgust": (102, 255, 178)   # Mint
}

GENDER_COLORS = {
    "Man": (255, 180, 50),       # Royal Blue (BGR)
    "Woman": (210, 100, 255),    # Vibrant Magenta/Pink (BGR)
    "Unknown": (180, 180, 180)
}

DEMOGRAPHIC_REGIONS = {
    "asian": "East / Southeast Asia",
    "indian": "South Asia",
    "black": "African / Afro-Descendant",
    "white": "European / Caucasian",
    "middle eastern": "Middle Eastern / West Asian",
    "latino hispanic": "Latino / Hispanic"
}

RECOMMENDATIONS = {
    "happy": "Mood: Positive & Energized. Ideal for collaborative engagement.",
    "sad": "Mood: Low energy detected. Consider taking a brief ergonomic break.",
    "angry": "Mood: Elevated stress indicators. Mindfulness pause suggested.",
    "surprise": "Mood: Highly alert & attentive. Optimal for information intake.",
    "neutral": "Mood: Calm & Balanced. Prime state for focused deep work.",
    "fear": "Mood: Hesitation/Tension detected. Reset posture and breathe deeply.",
    "disgust": "Mood: Discomfort indicators observed. Check environmental factors."
}


def render_glass_card(frame, x1, y1, x2, y2, alpha=0.6, bg_color=(20, 20, 24)):
    """Draws a semi-transparent glassmorphic background card."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return

    sub_img = frame[y1:y2, x1:x2]
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    blended = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 0)
    frame[y1:y2, x1:x2] = blended


def draw_face_annotations(frame, bbox, demographic_data):
    """
    Overlays bounding box and textual demographic/emotion metrics above each face.
    Format:
        Line 1: [Gender], [Age] years, [Emotion] ([Conf]%)
        Line 2: Demographic: [Estimated Region / Demographic Proxy]
    """
    x, y, w, h = bbox
    emotion = demographic_data.get("emotion", "neutral")
    emotion_conf = demographic_data.get("emotion_confidence", 0.0)
    age = demographic_data.get("age", None)
    gender = demographic_data.get("gender", "Analyzing")
    demographic = demographic_data.get("demographic", "Global")

    primary_color = EMOTION_COLORS.get(emotion.lower(), (0, 255, 255))

    # 1. Modern Corner-Accented Bounding Box
    corner_len = int(min(w, h) * 0.22)
    thickness = 2

    # Draw semi-transparent inner fill
    render_glass_card(frame, x, y, x + w, y + h, alpha=0.15, bg_color=(0, 0, 0))

    # Top-Left
    cv2.line(frame, (x, y), (x + corner_len, y), primary_color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_len), primary_color, thickness)
    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), primary_color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), primary_color, thickness)
    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), primary_color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), primary_color, thickness)
    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), primary_color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), primary_color, thickness)

    # 2. Textual Metric Badges Above Bounding Box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55

    # Line 1 text
    age_str = f"{age} yrs" if age is not None else "-- yrs"
    line1 = f"{gender}, {age_str} | {emotion.capitalize()} ({emotion_conf:.0f}%)"

    # Line 2 text (Demographic / Geographic Proxy)
    region_label = DEMOGRAPHIC_REGIONS.get(demographic.lower(), demographic)
    line2 = f"Demographic: {region_label}"

    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, 2)
    (w2, h2), _ = cv2.getTextSize(line2, font, 0.45, 1)

    badge_w = max(w1, w2) + 20
    badge_h = h1 + h2 + 22

    # Position badge above face box (or inside if near top border)
    bx1 = x
    bx2 = x + badge_w
    if y - badge_h - 10 >= 55:
        by1 = y - badge_h - 10
        by2 = y - 5
    else:
        by1 = y + 10
        by2 = y + 10 + badge_h

    # Render glassmorphic badge container
    render_glass_card(frame, bx1, by1, bx2, by2, alpha=0.75, bg_color=(15, 18, 22))

    # Colored indicator accent bar on the left of badge
    cv2.rectangle(frame, (bx1, by1), (bx1 + 4, by2), primary_color, -1)

    # Render Line 1 (Gender, Age, Emotion)
    cv2.putText(frame, line1, (bx1 + 10, by1 + h1 + 6), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Render Line 2 (Broad Geographic Demographic)
    cv2.putText(frame, line2, (bx1 + 10, by1 + h1 + h2 + 15), font, 0.45, (190, 215, 240), 1, cv2.LINE_AA)


def draw_hud(frame, fps, demographic_data, total_faces=1):
    """
    Renders top HUD navigation bar and live side analytics cards for Emotion and Demographics.
    """
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    emotion = demographic_data.get("emotion", "neutral")
    emotion_probs = demographic_data.get("emotion_spectrum", {})
    age = demographic_data.get("age", None)
    gender = demographic_data.get("gender", "Analyzing...")
    demographic = demographic_data.get("demographic", "Analyzing...")

    # 1. Top HUD Header Bar
    render_glass_card(frame, 0, 0, width, 55, alpha=0.75, bg_color=(10, 12, 16))
    cv2.putText(frame, "AI DEMOGRAPHIC & EMOTION INTELLIGENCE", (20, 35), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Real-time metrics in top bar
    fps_color = (0, 230, 115) if fps >= 20 else (0, 200, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 230, 35), font, 0.58, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {total_faces}", (width - 110, 35), font, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    # 2. Side Panel - Live Demographic Profile Card
    card_w = 260
    card_h = 320
    px1, py1 = width - card_w - 15, 70
    px2, py2 = width - 15, py1 + card_h
    render_glass_card(frame, px1, py1, px2, py2, alpha=0.75, bg_color=(15, 18, 24))

    # Section 1: Demographics Summary
    cv2.putText(frame, "DEMOGRAPHIC PROFILE", (px1 + 14, py1 + 22), font, 0.45, (140, 180, 240), 1, cv2.LINE_AA)

    # Gender & Age
    gender_color = GENDER_COLORS.get(gender, (200, 200, 200))
    cv2.putText(frame, f"Gender: {gender}", (px1 + 14, py1 + 48), font, 0.5, gender_color, 2, cv2.LINE_AA)
    age_display = f"{age} years" if age is not None else "Calculating..."
    cv2.putText(frame, f"Apparent Age: {age_display}", (px1 + 14, py1 + 72), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

    # Geographic / Demographic proxy
    region_display = DEMOGRAPHIC_REGIONS.get(demographic.lower(), demographic)
    cv2.putText(frame, "Geo/Ethnic Demographic:", (px1 + 14, py1 + 96), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{region_display}", (px1 + 14, py1 + 116), font, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    # Section 2: Emotion Spectrum Bars
    cv2.line(frame, (px1 + 14, py1 + 130), (px2 - 14, py1 + 130), (60, 65, 75), 1)
    cv2.putText(frame, "EMOTIONAL SPECTRUM", (px1 + 14, py1 + 148), font, 0.42, (140, 180, 240), 1, cv2.LINE_AA)

    y_offset = py1 + 170
    bar_max_w = 115
    for emo, color in EMOTION_COLORS.items():
        prob = emotion_probs.get(emo, 0.0)
        bar_w = int(prob * bar_max_w)

        cv2.putText(frame, f"{emo[:4].capitalize()}", (px1 + 14, y_offset + 10), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        # Background bar
        cv2.rectangle(frame, (px1 + 60, y_offset + 2), (px1 + 60 + bar_max_w, y_offset + 11), (45, 48, 55), -1)
        # Active progress bar
        if bar_w > 0:
            cv2.rectangle(frame, (px1 + 60, y_offset + 2), (px1 + 60 + bar_w, y_offset + 11), color, -1)
        # Percentage
        cv2.putText(frame, f"{int(prob * 100)}%", (px1 + 60 + bar_max_w + 6, y_offset + 10), font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        y_offset += 20

    # 3. Bottom Contextual Recommendation Bar
    rec_text = RECOMMENDATIONS.get(emotion.lower(), "AI Engine active. Observing behavioral signals...")
    render_glass_card(frame, 15, height - 48, width - 15, height - 10, alpha=0.75, bg_color=(12, 14, 18))
    main_color = EMOTION_COLORS.get(emotion.lower(), (0, 255, 255))
    cv2.circle(frame, (35, height - 29), 6, main_color, -1)
    cv2.putText(frame, rec_text, (55, height - 24), font, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
