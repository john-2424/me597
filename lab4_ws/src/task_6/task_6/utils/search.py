import math
from dataclasses import dataclass, field

@dataclass
class TargetHypothesis:
    """Last-seen ball hypothesis in the odom frame (Kalman-lite α-β)."""
    has_value: bool = False
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    recency: float = 0.0    # [0,1], decays over time
    cov: float = 1.0        # scalar “radius” for gating (m)

    # tunables (can be tied to ROS params if desired)
    alpha: float = 0.35
    beta: float = 0.10
    decay_per_s: float = 0.15
    cov_min: float = 0.5
    cov_max: float = 3.0
    proc_noise: float = 0.25

    def predict(self, dt: float):
        if not self.has_value:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.cov = min(self.cov_max, self.cov + self.proc_noise * dt)
        self.recency = max(0.0, self.recency - self.decay_per_s * dt)
        if self.recency <= 0.0:
            self.has_value = False

    def update(self, x_meas: float, y_meas: float, dt: float):
        # α-β filter update (position + velocity)
        if not self.has_value:
            self.x, self.y = x_meas, y_meas
            self.vx, self.vy = 0.0, 0.0
            self.cov = self.cov_min
            self.has_value = True
            self.recency = 1.0
            return
        rx = x_meas - self.x
        ry = y_meas - self.y
        self.x += self.alpha * rx
        self.y += self.alpha * ry
        self.vx += (self.beta / max(dt, 1e-3)) * rx
        self.vy += (self.beta / max(dt, 1e-3)) * ry
        self.cov = max(self.cov_min, min(self.cov_max, 0.8 * self.cov))
        self.recency = min(1.0, self.recency + 0.35)

    def gating_score(self, gx: float, gy: float) -> float:
        """Returns [0,1]: 1.0 if gap waypoint inside hypothesis circle (x,y,cov)."""
        if not self.has_value:
            return 0.0
        d = math.hypot(gx - self.x, gy - self.y)
        if d >= self.cov:
            return max(0.0, 1.0 - (d - self.cov) / (self.cov + 1e-6))
        return 1.0


@dataclass
class SearchContext:
    state: str = 'TRACK'  # TRACK, SPIN_SCAN, SELECT_GAP, GO_TO_GAP, LOCAL_SCAN, RECOVER
    state_start_sec: float = 0.0

    # odom pose (x,y,yaw)
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    have_odom: bool = False

    # grid memory (visited cells)
    grid_res: float = 0.5        # meters per cell
    grid_half: float = 10.0      # 10 m in each direction
    visited: set = field(default_factory=set)

    # scanning
    spin_accum_yaw: float = 0.0
    spin_target: float = 2 * math.pi

    # current target gap
    goal_angle: float = 0.0         # in robot frame (radians)
    goal_waypoint: tuple = (0.0, 0.0)
    commit_sec: float = 0.0

    # auxiliary buffers
    scan_entropy: float = 0.0
    replan_ticker: float = 0.0

    # last commands (for safety smoothing)
    last_v: float = 0.0
    last_w: float = 0.0