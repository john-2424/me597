import cv2 as cv
import numpy as np

LOWER_RED_1 = np.array([0,   90,  80], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([170, 90,  80], dtype=np.uint8)
UPPER_RED_2 = np.array([180,255, 255], dtype=np.uint8)
K = 5  # morphology kernel


def _red_mask_hsv(frame_bgr):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    m1  = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    m2  = cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv.bitwise_or(m1, m2)
    kernel = np.ones((K, K), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask

def find_object_hsv(frame_bgr):
    """
    HSV-only: returns (cx, cy, w, h, (x,y,w,h)) or None
    """
    mask = _red_mask_hsv(frame_bgr)
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if not cnts: return None
    c = max(cnts, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    cx, cy = x + w/2.0, y + h/2.0
    return (cx, cy, w, h, (x, y, w, h))

def find_object_hsv_triangle(frame_bgr, tri_tol=0.04):
    """
    HSV + “must be triangle”: same return as above or None
    """
    mask = _red_mask_hsv(frame_bgr)
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if not cnts: return None
    # largest few, then choose first with ~3 vertices
    for c in sorted(cnts, key=cv.contourArea, reverse=True):
        peri = cv.arcLength(c, True)                           # (NB: present in your NB)
        approx = cv.approxPolyDP(c, tri_tol * peri, True)      # (NB: same method)
        if len(approx) == 3:
            x, y, w, h = cv.boundingRect(c)
            cx, cy = x + w/2.0, y + h/2.0
            return (cx, cy, w, h, (x, y, w, h))
    return None
