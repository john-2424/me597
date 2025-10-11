import numpy as np
import itertools
from math import sqrt, atan2, cos, sin
from collections import deque
from geometry_msgs.msg import PoseStamped

def _pose_xy(p: PoseStamped):
    return p.pose.position.x, p.pose.position.y

def _dist(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _project_on_segment(P, A, B):
    ax, ay = A; bx, by = B; px, py = P
    abx, aby = bx-ax, by-ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0.0:
        return A, 0.0
    apx, apy = px-ax, py-ay
    t = (apx*abx + apy*aby) / ab2
    t = max(0.0, min(1.0, t))
    return (ax + t*abx, ay + t*aby), t

def _yaw_from_quat(q):
    # ZYX yaw from quaternion (assumes planar)
    # yaw = atan2(2(wz+xy), 1-2(y^2+z^2)); simplified because x=y=0 in planar, but keep general:
    siny_cosp = 2*(q.w*q.z + q.x*q.y)
    cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z)
    return atan2(siny_cosp, cosy_cosp)

def _unit(vec):
    n = sqrt(vec[0]*vec[0] + vec[1]*vec[1])
    return (vec[0]/n, vec[1]/n) if n > 1e-9 else (0.0, 0.0)

def _curvature_three_points(P0, P1, P2):
    # discrete curvature using turning angle over arc length
    a = _dist(P0, P1)
    b = _dist(P1, P2)
    if a < 1e-6 or b < 1e-6:
        return 0.0
    ang1 = atan2(P1[1]-P0[1], P1[0]-P0[0])
    ang2 = atan2(P2[1]-P1[1], P2[0]-P1[0])
    dtheta = np.arctan2(np.sin(ang2-ang1), np.cos(ang2-ang1))
    arc = a + b
    return abs(dtheta) / max(arc, 1e-6)

class Follower:
    def __init__(self,
                 lookahead_min=0.4, lookahead_max=1.2,
                 kv=0.8, kc=0.4, eps_k=1e-3,
                 goal_tol=0.15,
                 history_sec=1.0,   # how long to keep for speed est
                 hz=10.0):
        self.last_idx = 0
        self.goal_tol = goal_tol
        self.lookahead_min = lookahead_min
        self.lookahead_max = lookahead_max
        self.kv = kv
        self.kc = kc
        self.eps_k = eps_k
        # pose history for speed: store (t, x, y)
        self.hist = deque(maxlen=int(max(3, history_sec*hz)))
        self._ema_v = 0.0  # smoothed speed

    def _approx_speed(self, now_sec, x, y):
        """Finite-difference speed with EMA smoothing."""
        self.hist.append((now_sec, x, y))
        if len(self.hist) < 2:
            return 0.0
        # use the oldest enough to cover ~0.3–0.5 s to reduce noise
        t0, x0, y0 = self.hist[0]
        dt = now_sec - t0
        if dt <= 1e-3:
            v_inst = 0.0
        else:
            v_inst = _dist((x, y), (x0, y0)) / dt
        # EMA smoothing
        alpha = 0.3
        self._ema_v = alpha*v_inst + (1-alpha)*self._ema_v
        return self._ema_v

    def _local_curvature(self, pts, base_idx, win=1):
        N = len(pts)
        if N < 3:
            return 0.0
        # choose a triplet that exists; bias slightly ahead of base_idx
        i0 = base_idx + win - 1
        i0 = max(0, min(N - 3, i0))   # clamp so i0, i0+1, i0+2 are valid
        p0, p1, p2 = pts[i0], pts[i0+1], pts[i0+2]

        # discrete curvature = turning angle / arc length
        a = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) ** 0.5
        b = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5
        if a < 1e-6 or b < 1e-6:
            return 0.0
        ang1 = np.arctan2(p1[1]-p0[1], p1[0]-p0[0])
        ang2 = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        dth = np.arctan2(np.sin(ang2-ang1), np.cos(ang2-ang1))
        return abs(dth) / max(a + b, 1e-6)

    def _closest_point_on_path(self, pts, P, start_hint, back=10, fwd=60):
        N = len(pts)
        if N == 1:
            return 0, 0.0, pts[0]
        lo = max(0, start_hint - back)
        hi = min(N - 2, start_hint + fwd)    # N-2, not N-1
        best = (1e18, 0, 0.0, pts[0])
        for i in range(lo, hi + 1):
            proj, t = _project_on_segment(P, pts[i], pts[i+1])
            d2 = (proj[0]-P[0])**2 + (proj[1]-P[1])**2
            if d2 < best[0]:
                best = (d2, i, t, proj)
        _, seg_i, t_on, closest = best
        return seg_i, t_on, closest

    def _accumulate_to_lookahead(self, pts, start_point, start_seg_idx, L):
        N = len(pts)
        acc = 0.0
        cur = start_point
        i = start_seg_idx
        j = i+1
        while True:
            if j >= N:
                return N-1
            step = _dist(cur, pts[j])
            if acc + step >= L:
                return j  # nearest waypoint ahead on this segment
            acc += step
            cur = pts[j]
            i += 1; j += 1

    def get_path_idx(self, path, vehicle_pose, now_sec):
        pts = [ (p.pose.position.x, p.pose.position.y) for p in path.poses ]
        N = len(pts)
        if N == 0:
            return 0
        if N == 1:
            self.last_idx = 0
            return 0
        if N <= 1:
            self.last_idx = 0
            return 0

        # Robot pose
        rx = vehicle_pose.pose.position.x
        ry = vehicle_pose.pose.position.y
        yaw = _yaw_from_quat(vehicle_pose.pose.orientation)
        h = (cos(yaw), sin(yaw))

        # --- 1) closest point on path near last_idx
        seg_i, t_on, closest = self._closest_point_on_path(pts, (rx, ry), min(self.last_idx, N-1))

        # --- 2) curvature ahead (discrete)
        kappa = self._local_curvature(pts, seg_i)

        # --- 3) approximate speed from history
        v = self._approx_speed(now_sec, rx, ry)

        # --- 4) dynamic lookahead
        L_raw = self.kv*v + self.kc / (abs(kappa) + self.eps_k)
        # small curvature-aware shrink for very sharp turns
        if abs(kappa) > 0.7:         # threshold in 1/m (tune)
            L_raw *= 0.75
        L = float(np.clip(L_raw, self.lookahead_min, self.lookahead_max))

        # --- 5) pick target by accumulating forward distance L
        target_idx = self._accumulate_to_lookahead(pts, closest, seg_i, L)

        # --- 6) heading check: ensure target is in front
        seg_vec = _unit((pts[target_idx][0]-rx, pts[target_idx][1]-ry))
        dot = seg_vec[0]*h[0] + seg_vec[1]*h[1]
        if dot < 0.0 and target_idx+1 < N:
            target_idx += 1  # nudge forward

        # --- 7) never go backwards; goal tolerance
        idx = max(target_idx, self.last_idx)

        # approximate remaining distance to end
        tail = 0.0
        cur = closest
        for k in range(seg_i, N-1):
            a = cur if k == seg_i else pts[k]
            b = pts[k+1]
            tail += _dist(a, b)
        if tail <= self.goal_tol:
            idx = N-1

        idx = max(target_idx, self.last_idx)
        idx = min(idx, N - 1)     # clamp
        self.last_idx = idx

        return idx
