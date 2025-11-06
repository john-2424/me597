import cv2 as cv
import numpy as np

LOWER_RED_1 = np.array([0,   160, 130], dtype=np.uint8)
UPPER_RED_1 = np.array([8,   255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([172, 160, 130], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)
K = 7


# def _red_mask_hsv(frame_bgr):
#     hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

#     # suppress very dark or very bright highlights in V before thresholding
#     v = hsv[...,2]
#     v = np.clip(v, 30, 245)
#     hsv[...,2] = v

#     m1  = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
#     m2  = cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
#     mask = cv.bitwise_or(m1, m2)
#     mask = cv.medianBlur(mask, 5)  # smooth fine brick texture; keeps ball

#     kernel = np.ones((K, K), np.uint8)
#     mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
#     mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
#     return mask

def _red_mask_hsv(frame_bgr):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    hsv[...,2] = np.clip(hsv[...,2], 30, 245)

    m1 = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    m2 = cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv.bitwise_or(m1, m2)

    mask = cv.medianBlur(mask, 5)
    kernel = np.ones((3,3), np.uint8)  # 3x3, not 7x7
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

def find_object_hsv_circle(frame_bgr,
                           circ_tol=0.72,     # was 0.86
                           ar_tol=0.28,       # was 0.18
                           min_px=200,        # was 600
                           fill_tol=0.25,     # was 0.16
                           s_min=120, v_min=90):  # was 170,150
    """
    HSV + “must be circle”: returns (cx, cy, w, h, (x,y,w,h)) or None
    """
    hsv  = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

    # tame highlights & deep shadows to stabilize thresholding (you had this in _red_mask_hsv)
    v = hsv[..., 2]
    v = np.clip(v, 30, 245)
    hsv[..., 2] = v

    m1   = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    m2   = cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv.bitwise_or(m1, m2)

    # smooth speckle before morphology (you did this in the HSV-only path)
    mask = cv.medianBlur(mask, 5)

    kernel = np.ones((5, 5), np.uint8)  # was K=7 → less erosion of small balls
    mask   = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
    mask   = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return None

    def red_hue_score(h):
        # allow wider hue tolerance; wrap-around safe
        h = h.astype(np.float32)
        return np.minimum(np.minimum(np.abs(h - 0), np.abs(h - 180)), np.abs(h - 179))

    best = None
    best_area = 0

    for c in sorted(cnts, key=cv.contourArea, reverse=True):
        area = cv.contourArea(c)
        if area < min_px:
            continue

        x, y, w, h = cv.boundingRect(c)
        if h <= 0:
            continue
        ar = w / float(h)
        if abs(ar - 1.0) > ar_tol:
            continue

        peri = cv.arcLength(c, True)
        if peri <= 0:
            continue

        # (1) circularity
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < circ_tol:
            continue

        # (2) bbox fill ~ π/4 for a circle; give more slack
        fill = area / float(w * h)
        if abs(fill - (np.pi / 4.0)) > fill_tol:
            continue

        # (3) convexity — use the raw contour; approx can introduce artifacts
        if not cv.isContourConvex(c):
            # optional: allow slight non-convexity if circularity is strong
            if circularity < (circ_tol + 0.05):
                continue

        # (4) color quality (relaxed)
        mask_roi = mask[y:y+h, x:x+w]
        if mask_roi.size == 0:
            continue
        hsv_roi  = hsv[y:y+h, x:x+w]
        sel      = mask_roi > 0
        if sel.sum() == 0:
            continue

        mean_s = float(np.mean(hsv_roi[..., 1][sel]))
        mean_v = float(np.mean(hsv_roi[..., 2][sel]))
        if mean_s < s_min or mean_v < v_min:
            continue

        # allow wider hue error (lighting shifts, camera WB)
        if float(np.mean(red_hue_score(hsv_roi[..., 0][sel]))) > 12:  # was 6
            continue

        if area > best_area:
            cx, cy = x + w / 2.0, y + h / 2.0
            best = (cx, cy, w, h, (x, y, w, h))
            best_area = area

    return best
