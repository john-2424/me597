import cv2 as cv
import numpy as np

LOWER_RED_1 = np.array([0,   140, 110], dtype=np.uint8)
UPPER_RED_1 = np.array([8,   255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([172, 140, 110], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)
K = 3
MIN_BB_SIZE = 15

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

    mask = cv.medianBlur(mask, 3)
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

# def find_object_hsv_circle(frame_bgr):
#     """
#     Color + circle: returns (cx, cy, w, h, (x,y,w,h)) or None
#     Uses your HSV mask, then selects the most circle-like red region
#     and derives the box from the fitted circle so glare doesn't crop the BB.
#     """
#     H, W = frame_bgr.shape[:2]

#     # 1) Color: reuse your robust red mask
#     mask = _red_mask_hsv(frame_bgr)

#     # Light cleanup to close glare gaps (keeps your style/params minimal)
#     kernel = np.ones((K, K), np.uint8)  # you defined K=7
#     mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

#     # Optional small hole fill (helps with specular highlight pinholes)
#     # We fill only tiny holes to avoid merging separate objects.
#     holes = cv.bitwise_not(mask)
#     cnts_holes = cv.findContours(holes, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)[0]
#     for c in cnts_holes:
#         if cv.contourArea(c) < 0.005 * (H * W):  # tiny hole
#             cv.drawContours(mask, [c], -1, 255, thickness=-1)

#     # 2) Shape: score contours by circularity and fit vs min-enclosing circle
#     cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
#     if not cnts:
#         return None

#     best = None
#     best_score = -1.0
#     best_center = None
#     best_r = None
#     MIN_AREA = 400  # tune if your ball is far away

#     for c in cnts:
#         area = cv.contourArea(c)
#         if area < MIN_AREA:
#             continue

#         peri = cv.arcLength(c, True)
#         if peri <= 0:
#             continue

#         circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)  # 1.0 = perfect circle
#         (xc, yc), r = cv.minEnclosingCircle(c)
#         if r <= 0:
#             continue

#         circle_area = np.pi * (r * r)
#         fill_ratio = float(area) / (circle_area + 1e-6)           # <= 1.0
#         x, y, w, h = cv.boundingRect(c)
#         ar = w / float(h + 1e-6)
#         aspect_penalty = 1.0 - min(ar, 1.0 / ar)                  # 0 best

#         # Weighted score (inspired by your notebook’s circularity/inertia ideas)
#         score = (0.65 * circularity) + (0.35 * fill_ratio) - (0.25 * aspect_penalty)

#         if score > best_score:
#             best_score = score
#             best_center = (xc, yc)
#             best_r = r
#             best = c

#     # Accept only sufficiently circle-like regions
#     MIN_SCORE = 0.35  # bump to 0.45 if you see false positives
#     if best is not None and best_score >= MIN_SCORE and best_r is not None:
#         x1 = int(round(best_center[0] - best_r))
#         y1 = int(round(best_center[1] - best_r))
#         x2 = int(round(best_center[0] + best_r))
#         y2 = int(round(best_center[1] + best_r))
#         x1 = max(0, x1); y1 = max(0, y1)
#         x2 = min(W - 1, x2); y2 = min(H - 1, y2)
#         bbox = (x1, y1, x2 - x1, y2 - y1)
#         return (float(best_center[0]), float(best_center[1]),
#                 float(bbox[2]), float(bbox[3]), bbox)

#     # 3) Fallback: Hough on red edges, then require it be mostly on-mask
#     edges = cv.Canny(mask, 50, 150)
#     edges = cv.GaussianBlur(edges, (9, 9), 2)
#     circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT,
#                               dp=1.2, minDist=H // 4,
#                               param1=120, param2=15,
#                               minRadius=10, maxRadius=0)

#     if circles is not None:
#         circles = np.uint16(np.around(circles[0, :]))
#         # Pick the circle with the most "red mask" inside its box
#         best_c = None
#         best_val = -1
#         for (x, y, r) in circles:
#             x1 = max(0, int(x - r)); y1 = max(0, int(y - r))
#             x2 = min(W - 1, int(x + r)); y2 = min(H - 1, int(y + r))
#             roi = mask[y1:y2, x1:x2]
#             if roi.size == 0:
#                 continue
#             mean_mask = float(np.mean(roi))
#             if mean_mask > best_val:
#                 best_val = mean_mask
#                 best_c = (x, y, r)
#         if best_c is not None:
#             x, y, r = best_c
#             x1 = max(0, int(x - r)); y1 = max(0, int(y - r))
#             x2 = min(W - 1, int(x + r)); y2 = min(H - 1, int(y + r))
#             bbox = (x1, y1, x2 - x1, y2 - y1)
#             return (float(x), float(y), float(bbox[2]), float(bbox[3]), bbox)

#     return None

def _red_seed_mask_strict(frame_bgr):
    """
    Very strict seeds: only vivid red pixels.
    Combines tight HSV + RGB opponent-red.
    """
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

    # Tight hue near red + higher S,V floors
    SEED_LOW_1 = np.array([0,   180, 150], dtype=np.uint8)
    SEED_UP_1  = np.array([6,   255, 255], dtype=np.uint8)
    SEED_LOW_2 = np.array([174, 180, 150], dtype=np.uint8)
    SEED_UP_2  = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv.inRange(hsv, SEED_LOW_1, SEED_UP_1)
    m2 = cv.inRange(hsv, SEED_LOW_2, SEED_UP_2)
    hsv_seed = cv.bitwise_or(m1, m2)

    # RGB opponent red (intensity): R ≫ G,B and R is bright
    b, g, r = cv.split(frame_bgr)
    max_gb  = cv.max(g, b)
    r_dom   = cv.compare(cv.subtract(r, max_gb), 60, cv.CMP_GE)  # R - max(G,B) ≥ 60
    r_hi    = cv.compare(r, 170, cv.CMP_GE)                      # R ≥ 170
    rgb_seed = cv.bitwise_and(r_dom, r_hi)

    seed = cv.bitwise_and(hsv_seed, rgb_seed)
    seed = cv.morphologyEx(seed, cv.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return seed

def _seed_rois(seed_mask, pad, frame_shape, max_rois=4):
    H, W = frame_shape[:2]
    cnts = cv.findContours(seed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if not cnts: return []
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:max_rois]
    rois = []
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(W-1, x + w + pad); y2 = min(H-1, y + h + pad)
        rois.append((x1, y1, x2, y2))
    return rois

def _verify_red_ball_score(frame_bgr, mask, center, r):
    """
    Returns score in [0,1] (higher = more likely to be the red ball).
    Uses inner disk stats to be robust to glare/edges.
    """
    H, W = frame_bgr.shape[:2]
    cx, cy = center
    if r <= 0 or cx < 0 or cy < 0 or cx >= W or cy >= H:
        return 0.0

    # Size sanity only (don't zero-out tiny; just down-weight later)
    if r > 0.6 * min(H, W):
        return 0.0

    # ROI crop
    x1 = max(0, int(cx - r)); y1 = max(0, int(cy - r))
    x2 = min(W - 1, int(cx + r)); y2 = min(H - 1, int(cy + r))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    rh, rw = roi_mask.shape[:2]
    yy, xx = np.ogrid[0:rh, 0:rw]
    ccx = cx - x1; ccy = cy - y1
    r_in = max(3.0, 0.8 * r)
    inner = ((xx - ccx)**2 + (yy - ccy)**2) <= (r_in**2)
    if not np.any(inner):
        return 0.0

    hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    Hh, Sh, Vh = hsv[...,0], hsv[...,1], hsv[...,2]

    # Cues (each 0..1)
    fill = float(np.mean((roi_mask > 0)[inner]))                         # red-mask support
    sat  = float(np.mean(Sh[inner]) / 255.0)                             # saturation
    hue  = Hh[inner]
    hue_dist = np.minimum(hue, 180.0 - hue)                              # distance to red
    f_hue = np.clip(1.0 - (np.median(hue_dist) / 30.0), 0.0, 1.0)
    gray = cv.cvtColor(roi_bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv.Laplacian(gray[inner], cv.CV_32F, ksize=3)
    lap_var = float(np.var(lap))
    f_smooth = np.clip(1.0 - ((lap_var - 100.0) / (600.0 - 100.0)), 0.0, 1.0)

    # Combine
    score = (0.40 * fill) + (0.30 * f_hue) + (0.20 * sat) + (0.10 * f_smooth)
    return float(score)

def find_object_hsv_circle(frame_bgr):
    """
    Color + circle: returns (cx, cy, w, h, (x,y,w,h)) or None
    Maintains your return signature and naming.
    """
    H, W = frame_bgr.shape[:2]

    # --- Seed stage: find vivid-red pixels and search locally around them ---
    seed = _red_seed_mask_strict(frame_bgr)
    rois = _seed_rois(seed, pad=24, frame_shape=frame_bgr.shape, max_rois=4)

    for (x1, y1, x2, y2) in rois:
        roi = frame_bgr[y1:y2, x1:x2]

        # Use your normal (looser) mask inside the ROI to grow the ball
        mask_roi = _red_mask_hsv(roi)
        # Protect small targets: pick kernel by red coverage
        red_ratio = float(np.mean(mask_roi > 0))
        ker = np.ones((3,3), np.uint8) if red_ratio < 0.01 else np.ones((K,K), np.uint8)
        mask_roi = cv.morphologyEx(mask_roi, cv.MORPH_CLOSE, ker, iterations=1)

        cnts = cv.findContours(mask_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        if not cnts: 
            continue

        # Choose the most circle-like blob
        best, best_score, best_center, best_r = None, -1.0, None, None
        for c in cnts:
            area = cv.contourArea(c)
            if area < 60: 
                continue
            peri = cv.arcLength(c, True)
            if peri <= 0: 
                continue
            circ = 4.0*np.pi*area/(peri*peri+1e-6)
            (xc, yc), r = cv.minEnclosingCircle(c)
            if r <= 0: 
                continue
            circle_area = np.pi*(r*r)
            fill_ratio = float(area)/(circle_area+1e-6)
            x,y,w,h = cv.boundingRect(c)
            ar = w/float(h+1e-6)
            aspect_pen = 1.0 - min(ar, 1.0/ar)
            score = (0.65*circ) + (0.35*fill_ratio) - (0.25*aspect_pen)
            if score > best_score:
                best, best_score, best_center, best_r = c, score, (xc+x1, yc+y1), r

        if best is None:
            continue

        # Verify + return (you already have _verify_red_ball or _verify_red_ball_score)
        v = _verify_red_ball_score(frame_bgr, _red_mask_hsv(frame_bgr), best_center, best_r)
        if v >= 0.50:
            xA = int(round(best_center[0] - best_r)); yA = int(round(best_center[1] - best_r))
            xB = int(round(best_center[0] + best_r)); yB = int(round(best_center[1] + best_r))
            xA = max(0, xA); yA = max(0, yA); xB = min(W-1, xB); yB = min(H-1, yB)
            bbox = (xA, yA, xB-xA, yB-yA)

            # Final BB floor to kill wall specks
            if bbox[2] >= MIN_BB_SIZE and bbox[3] >= MIN_BB_SIZE:
                return (float(best_center[0]), float(best_center[1]),
                        float(bbox[2]), float(bbox[3]), bbox)
    
    return None