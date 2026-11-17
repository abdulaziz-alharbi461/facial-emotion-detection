"""
Advanced Face Alignment & Normalization Engine
Uses 2D Affine Similarity Transformation to align facial eye landmarks horizontally,
eliminating head tilt and rotation errors before deep learning inference.
"""

import cv2
import numpy as np


class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=224, desired_face_height=224):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

        # Load eye cascade for fast landmark localization
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def align(self, image, face_bbox):
        """
        Aligns a face ROI based on eye coordinates using affine rotation and scaling.
        :param image: Full BGR frame image.
        :param face_bbox: Tuple (x, y, w, h) of the face bounding box.
        :return: Canonical, upright 224x224 normalized face ROI.
        """
        x, y, w, h = face_bbox
        face_roi = image[y:y+h, x:x+w]

        if face_roi.size == 0:
            return None

        # Convert face ROI to grayscale for eye detection
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=6, minSize=(15, 15))

        if len(eyes) < 2:
            # Fallback if 2 distinct eyes are not isolated: resize crop without rotation
            return cv2.resize(face_roi, (self.desired_face_width, self.desired_face_height))

        # Sort eyes left-to-right based on x-coordinate in ROI
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye_center = (x + eyes[0][0] + eyes[0][2] // 2, y + eyes[0][1] + eyes[0][3] // 2)
        right_eye_center = (x + eyes[-1][0] + eyes[-1][2] // 2, y + eyes[-1][1] + eyes[-1][3] // 2)

        # 1. Calculate angle between eye centers (theta = arctan2(dY, dX))
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(d_y, d_x))

        # 2. Calculate desired distance scaling factor
        dist = np.sqrt((d_x ** 2) + (d_y ** 2))
        desired_dist = (1.0 - (2 * self.desired_left_eye[0])) * self.desired_face_width
        scale = desired_dist / max(dist, 1.0)

        # 3. Midpoint between eye centers
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # 4. Construct 2D Affine Rotation Matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Adjust matrix translation offset for desired eye position
        t_x = self.desired_face_width * 0.5
        t_y = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (t_x - eyes_center[0])
        M[1, 2] += (t_y - eyes_center[1])

        # 5. Apply Affine Transformation
        aligned_face = cv2.warpAffine(
            image, M,
            (self.desired_face_width, self.desired_face_height),
            flags=cv2.INTER_CUBIC
        )

        return aligned_face
