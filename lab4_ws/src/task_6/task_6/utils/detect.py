import cv2 as cv
import numpy as np

LOWER_RED_1 = np.array([0,   140, 110], dtype=np.uint8)
UPPER_RED_1 = np.array([10,  255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([170, 140, 110], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)
K = 5

def _red_mask_hsv(frame_bgr):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

    # suppress very dark or very bright highlights in V before thresholding
    v = hsv[...,2]
    v = np.clip(v, 30, 245)
    hsv[...,2] = v

    m1  = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    m2  = cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv.bitwise_or(m1, m2)
    mask = cv.medianBlur(mask, 5)  # smooth fine brick texture; keeps ball

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

def find_object_hsv_circle(frame_bgr, circ_tol=0.85, ar_tol=0.15):
    """
    HSV + strong circularity gating:
      - circularity >= circ_tol
      - aspect ratio ~ 1 (|ar-1| <= ar_tol)
      - high solidity (compact, no gaps)
      - extent (area / bbox area) reasonably high
      - radial std/mean small (uniform radius)
      - size within a sane band relative to image
    Returns (cx, cy, w, h, (x,y,w,h)) or None
    """
    mask = _red_mask_hsv(frame_bgr)
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return None

    H, W = frame_bgr.shape[:2]
    img_area = float(W * H)
    # size band: ignore tiny specks and huge walls
    min_area = 0.0003 * img_area     # ~0.03% of frame
    max_area = 0.06   * img_area     # ~6% of frame

    best = None
    best_score = -1.0

    for c in sorted(cnts, key=cv.contourArea, reverse=True):
        area = cv.contourArea(c)
        if area < min_area or area > max_area:
            continue

        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull) or 1.0
        solidity = area / hull_area           # circles ~1.0

        peri = cv.arcLength(c, True) or 1.0
        circularity = 4.0 * np.pi * area / (peri * peri)

        x, y, w, h = cv.boundingRect(c)
        ar = w / float(h) if h > 0 else 0.0
        extent = area / float(w * h)          # circles ~π/4 ≈ 0.785

        # radial-uniformity: distances from centroid should be consistent
        M = cv.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        pts = c.reshape(-1, 2)
        r = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        r_mean = np.mean(r)
        r_nstd = (np.std(r) / (r_mean + 1e-6))   # lower is more circle-like

        # hard gates
        if circularity < circ_tol:             # very round only
            continue
        if abs(ar - 1.0) > ar_tol:             # not square-ish -> reject
            continue
        if solidity < 0.92:                    # fill must be compact
            continue
        if extent < 0.60:                      # reject long rectangles/grids
            continue
        if r_nstd > 0.20:                      # radius must be uniform
            continue

        # score & keep best (more round + compact)
        score = (circularity * 0.5) + (solidity * 0.3) + (extent * 0.2)
        if score > best_score:
            best_score = score
            best = (cx, cy, w, h, (x, y, w, h))

    return best
