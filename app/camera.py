"""
Threaded Webcam Stream Module
Decouples video capture from processing to eliminate camera I/O blocking.
"""

import time
import threading
import cv2


class ThreadedWebcam:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()
        self._init_camera()

    def _init_camera(self):
        backends = [
            (self.src, cv2.CAP_DSHOW),
            (self.src, cv2.CAP_MSMF if hasattr(cv2, 'CAP_MSMF') else 0),
            (1, cv2.CAP_DSHOW),
            (self.src, None)
        ]
        for idx, backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        self.grabbed = True
                        self.frame = test_frame
                        print(f"[CAMERA] Initialized device index {idx} with backend {backend}")
                        return
                    cap.release()
            except Exception:
                pass
        raise RuntimeError("[CAMERA ERROR] Failed to initialize any webcam video stream.")

    def start(self):
        self.stopped = False
        thread = threading.Thread(target=self._update, name="WebcamCaptureThread", daemon=True)
        thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.grabbed = True
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[CAMERA] Webcam thread stopped and resources released.")
