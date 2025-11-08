import numpy as np
import cv2 as cv


class Video:
    def __init__(self, vf_name=0, camera=False):
        self.vf_name = vf_name
        self.camera = camera
        self.cap = cv.VideoCapture(0 if camera else vf_name)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open {'camera' if camera else 'video file'}: {vf_name}")

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.release()
            raise StopIteration
        return frame

    def __del__(self):
        self.release()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()