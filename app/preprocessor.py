"""
Precision Image Pre-processing Pipeline
Handles extreme lighting, overexposure, low contrast, and sensor noise.
"""

import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, gamma=1.2, clahe_clip=2.5):
        self.gamma = gamma
        self.clahe_clip = clahe_clip

        # Build lookup table for fast Gamma Correction
        inv_gamma = 1.0 / self.gamma
        self.gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def gamma_correct(self, image):
        """Adjusts non-linear luminance curve to normalize over/under-exposed frames."""
        return cv2.LUT(image, self.gamma_lut)

    def denoise_bilateral(self, image):
        """Applies edge-preserving Bilateral Filter to smooth facial skin while keeping eyes/mouth edges sharp."""
        return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)

    def clahe_enhance(self, image):
        """Applies Contrast Limited Adaptive Histogram Equalization to LAB L-channel."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced

    def process(self, image):
        """Full 3-stage enhancement pipeline."""
        if image is None or image.size == 0:
            return image

        # 1. Gamma Correction
        step1 = self.gamma_correct(image)
        # 2. Edge-preserving Denoising
        step2 = self.denoise_bilateral(step1)
        # 3. Local Contrast Enhancement
        step3 = self.clahe_enhance(step2)

        return step3
