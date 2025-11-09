import math
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from task_6.utils.detect import find_object_hsv, find_object_hsv_triangle, find_object_hsv_circle, LOWER_RED_1, LOWER_RED_2, UPPER_RED_1, UPPER_RED_2
from task_6.utils.pid import PID
from task_6.utils.helper import clamp01, wrap_pi
from task_6.utils.search import SearchContext, TargetHypothesis

class RedBallTracker(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('red_ball_tracker')

        # Subscribe to a topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            20
        )
        self.subscription  # prevent unused variable warning

        # Bridge to convert image type between CV and ROS
        self.bridge = CvBridge()

        # Parameter to show video in this node
        self.show_video = True

        # Mode of detection
        self.detector_mode = 'hsv_circle'  # hsv_circle, hsv_triangle or hsv

        # Runtime options/ROS Params
        self.declare_parameter('controller', 'pp')     # 'pid', 'pidgs', 'pp'
        self.declare_parameter('search_mode', 'frontier')   # 'none', 'frontier'
        self.controller  = self.get_parameter('controller').get_parameter_value().string_value
        self.search_mode = self.get_parameter('search_mode').get_parameter_value().string_value

        # Frame info params
        self.first_frame = True
        self.frame_width = None
        self.frame_height = None

        # Tracking Reference params
        self.speed_reference = None
        self.speed_tol = 1
        self.heading_reference = None
        self.heading_tol = 1
        self.prev_speed = 0.0
        self.prev_heading = 0.0
        self.log_prev_speed = 0.0
        self.log_prev_heading = 0.0
        
        # PID parameters
        self.prev_sec = None
        self.speed_max = 0.20
        self.heading_max = 1.0
        self.turn_coeff = 1.0    # 0.20 coeff to stop bot by zeroing bot vel, 1.0 to ignore this logic
        self.speed_db   = 0.05      # ~5% size error
        self.heading_db = 0.02      # ~2% of half-frame
        self.max_speed_slew   = 0.05     # m/s per cycle
        self.max_heading_slew = 0.20     # rad/s per cycle
        # PIDs for speed and yaw
        # normalized-error tuning (err in [-1,1])
        self.pid_speed = PID(
            kp=0.18, ki=0.03, kd=0.08,
            i_limit=0.8,
            out_limit=(-self.speed_max, self.speed_max)
        )
        self.pid_heading = PID(
            kp=1.00, ki=0.02, kd=0.25,
            i_limit=0.8,
            out_limit=(-self.heading_max, self.heading_max)
        )

        # Proximity scheduling (for speed only)
        # define "far" and "near" box widths in pixels; set after first frame
        self.bw_far = None
        self.bw_near = None
        # blend weight: how much proximity matters vs turning (for speed scheduling)
        self.alpha_dist = 0.6  # 0..1; starting with 0.6 (proximity matters, but turn still dominates)
        # Heading normalization
        self.heading_norm = None   # set after first frame
        # Gain scheduling knobs (turn-severity only for first pass) ---
        # threshold where turning is considered "serious" (normalized error)
        self.heading_thr = 0.30  # try 0.25~0.35

        # Heading PID anchor gains (EASY vs HARD)
        self.H_KP_E, self.H_KI_E, self.H_KD_E = 1.00, 0.00, 0.20
        self.H_KP_H, self.H_KI_H, self.H_KD_H = 1.60, 0.00, 0.35
        # Speed PID anchor gains (EASY vs HARD) – be gentler while turning
        self.S_KP_E, self.S_KI_E, self.S_KD_E = 0.18, 0.03, 0.00
        self.S_KP_H, self.S_KI_H, self.S_KD_H = 0.10, 0.01, 0.00

        self.pp_hfov = 60.0 * math.pi/180.0  # approximate; tune with your camera
        self.pp_Ld   = 0.5  # L_d in meters
        self.pp_vmax = 0.22  # cap linear speed for PP
        self.pp_vmin = 0.05  # floor linear speed so it moves
        self.pp_turn_slow = 0.8  # how much turning reduces speed (0..1)
        self.pp_w_dist = 0.6  # blend weight of distance vs turn
        self.pp_kdist      = 0.8       # gain on normalized distance error
        self.pp_stop_db    = 0.08     # deadband on |(bw - ref)/ref| ; e.g., ±8% width
        self.pp_vback_max  = 0.22  # max reverse speed (0 to disable backing up)
        self.pp_omega_vnom = 0.18  # nominal speed used to scale omega (steer strength)
        self.pp_reverse_ok = True  # allow reverse when too close

        # Runtime options
        self.gap_min_width_m = 0.5
        self.gap_min_depth_m = 0.8
        self.spin_rate_hz    = 1.5
        self.local_spin_deg  = 360
        self.replan_hz       = 3.0
        self.min_commit_s    = 1.0
        self.ttc_stop_d      = 0.35
        self.safe_accel      = 0.6
        self.inflation_m     = 0.2

        # utility weights
        self.w_width   = 0.35
        self.w_depth   = 0.20
        self.w_head    = 0.15
        self.w_novel   = 0.15
        self.w_info    = 0.10
        self.w_red     = 0.05
        self.w_tgt_gate = 0.20
        self.w_tgt_bear = 0.15

        self.scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # Command Velocity Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Search context & hypothesis
        self.sc = SearchContext()
        self.tgt = TargetHypothesis()
    
    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    @staticmethod
    def _yaw_from_quat(x, y, z, w):
        # Standard yaw extraction
        s1 = 2.0 * (w * z + x * y)
        c1 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(s1, c1)
    
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quat(q.x, q.y, q.z, q.w)
        self.sc.x, self.sc.y, self.sc.yaw = p.x, p.y, yaw
        self.sc.have_odom = True
    
    def _grid_index(self, x: float, y: float):
        gx = int(math.floor((x) / self.sc.grid_res))
        gy = int(math.floor((y) / self.sc.grid_res))
        if abs(gx * self.sc.grid_res) > self.sc.grid_half or abs(gy * self.sc.grid_res) > self.sc.grid_half:
            return None
        return (gx, gy)

    def _mark_visited(self, x: float, y: float):
        idx = self._grid_index(x, y)
        if idx is not None:
            self.sc.visited.add(idx)

    def _is_visited(self, x: float, y: float) -> bool:
        idx = self._grid_index(x, y)
        return (idx is not None) and (idx in self.sc.visited)

    def _red_likelihood(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        m = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        return float(cv.countNonZero(m)) / float(m.size)

    def _scan_gaps(self, min_width_m):
        if self.scan is None: return []
        r = np.array(self.scan.ranges, dtype=np.float32)
        ang0 = self.scan.angle_min
        dang = self.scan.angle_increment

        # Inflate obstacles: shrink effective free space
        r_eff = r.copy()
        valid = np.isfinite(r_eff)
        r_eff[~valid] = np.inf
        # Minimum depth threshold
        open_mask = (r_eff > max(self.gap_min_depth_m, 0.0))

        gaps = []
        i = 0
        n = len(open_mask)
        while i < n:
            if open_mask[i]:
                j = i
                while j < n and open_mask[j]:
                    j += 1
                seg = r_eff[i:j]
                seg_valid = np.isfinite(seg)
                if not np.any(seg_valid):
                    i = j
                    continue
                seg = seg[seg_valid]
                depth_med = float(np.median(seg))
                depth_var = float(np.var(seg))
                span = (j - i) * dang
                gap_width = depth_med * span

                if gap_width >= min_width_m and depth_med >= self.gap_min_depth_m:
                    ang_center = ang0 + (i + j - 1) * 0.5 * dang
                    # gap waypoint projected ahead at depth_med
                    gaps.append({
                        'theta': float(ang_center),
                        'span': float(span),
                        'width': float(gap_width),
                        'depth': float(depth_med),
                        'variance': float(depth_var),
                    })
                i = j
            else:
                i += 1

        # Sort widest-first as a baseline ordering
        gaps.sort(key=lambda g: g['width'], reverse=True)
        return gaps

    # Scoring with novelty, info-gain, head-cost, red-likelihood, and target hypothesis
    def _score_gaps(self, frame, gaps):
        if not gaps:
            return None
        # Normalization bases
        max_w = max(g['width'] for g in gaps) or 1.0
        max_d = max(g['depth'] for g in gaps) or 1.0
        max_v = max(g['variance'] for g in gaps) or 1.0

        # red cue
        red_lik = self._red_likelihood(frame)

        best, best_score = None, -1.0
        for g in gaps:
            # novelty: penalize if projected waypoint is in visited grid
            gx = self.sc.x + g['depth'] * math.cos(self.sc.yaw + g['theta'])
            gy = self.sc.y + g['depth'] * math.sin(self.sc.yaw + g['theta'])
            visited_pen = 1.0 if not self._is_visited(gx, gy) else 0.0

            # heading cost (prefer smaller |theta|)
            head = 1.0 - min(abs(g['theta']) / (math.pi), 1.0)

            # information gain via variance
            info = min(g['variance'] / max_v, 1.0)

            # base utilities
            U = (
                self.w_width * (g['width'] / max_w) +
                self.w_depth * (g['depth'] / max_d) +
                self.w_head  * head +
                self.w_novel * visited_pen +
                self.w_info  * info +
                self.w_red   * red_lik
            )

            # target hypothesis bias (if fresh)
            if self.tgt.has_value and self.tgt.recency > 0.15:
                gate = self.tgt.gating_score(gx, gy)  # spatial gate
                # bearing agreement
                last_bearing = math.atan2(self.tgt.y - self.sc.y, self.tgt.x - self.sc.x) - self.sc.yaw
                last_bearing = math.atan2(math.sin(last_bearing), math.cos(last_bearing))
                bear_agree = 1.0 - min(abs(last_bearing - g['theta']) / math.pi, 1.0)
                U += self.w_tgt_gate * gate + self.w_tgt_bear * bear_agree

            if U > best_score:
                best, best_score = g, U

        return best

    # Safety wrapper: Follow-the-Gap local steering + Time-to-Collision speed clip
    def _safety_wrap(self, v_cmd, w_cmd):
        if self.scan is None:
            return v_cmd, w_cmd

        # TTC speed limit
        r = np.array(self.scan.ranges, dtype=np.float32)
        valid = np.isfinite(r)
        r[~valid] = np.inf

        # look ahead in ±15° around current heading command
        if self.scan.angle_increment <= 0.0:
            return v_cmd, w_cmd
        half_cone = int(max(1, math.radians(15.0) / self.scan.angle_increment))
        zero_idx = int((-self.scan.angle_min) / self.scan.angle_increment)
        steer_idx = zero_idx + int(w_cmd / max(1e-6, self.heading_max) * half_cone)
        sl = max(0, steer_idx - half_cone)
        sr = min(len(r) - 1, steer_idx + half_cone)
        dmin = float(np.nanmin(r[sl:sr+1])) if np.any(np.isfinite(r[sl:sr+1])) else np.inf

        if np.isfinite(dmin):
            v_ttc = math.sqrt(max(0.0, 2.0 * self.safe_accel * max(0.0, dmin - self.inflation_m)))
            v_cmd = max(-self.pp_vback_max, min(v_cmd, v_ttc))

        # If commanded heading points into clutter, bias toward local free center
        window = 25  # samples around forward
        fwd_idx = zero_idx
        wl = max(0, fwd_idx - window)
        wr = min(len(r) - 1, fwd_idx + window)
        band = r[wl:wr+1]
        if np.any(np.isfinite(band)):
            # find largest consecutive finite-and-far subwindow
            thr = max(self.gap_min_depth_m, self.ttc_stop_d + self.inflation_m)
            mask = np.isfinite(band) & (band > thr)
            if np.any(mask):
                best_len, best_c = 0, None
                i = 0
                while i < len(mask):
                    if mask[i]:
                        j = i
                        while j < len(mask) and mask[j]:
                            j += 1
                        seg_len = j - i
                        if seg_len > best_len:
                            best_len = seg_len
                            best_c = (i + j - 1) * 0.5
                        i = j
                    else:
                        i += 1
                if best_c is not None:
                    # map to angular offset relative to forward
                    off = (best_c - (wr - wl) * 0.5) * self.scan.angle_increment
                    # blend small nudge
                    w_cmd = 0.8 * w_cmd + 0.2 * off

        # Slew-limited output (using search-context smoothing)
        v_out = self.__slew(self.sc.last_v, v_cmd, self.max_speed_slew)
        w_out = self.__slew(self.sc.last_w, w_cmd, self.max_heading_slew)
        self.sc.last_v, self.sc.last_w = v_out, w_out

        return v_out, w_out

    def _detect(self, frame):
        if self.detector_mode == 'hsv_circle':
            return find_object_hsv_circle(frame)
        if self.detector_mode == 'hsv_triangle':
            return find_object_hsv_triangle(frame)
        return find_object_hsv(frame)

    def __slew(self, prev, new, max_delta):
        delta = max(-max_delta, min(max_delta, new - prev))
        return prev + delta
    
    def __plan_pid(self, cx, bw, bh, dt):
        # now_sec = self.get_clock().now().nanoseconds * 1e-9
        # dt = now_sec - self.prev_sec if self.prev_sec is not None else 1.0 / 10.0
        # self.prev_sec = now_sec

        speed = 0.0
        heading = 0.0        
            
        # speed_curr = 0.5 * bw + 0.5 * bh
        speed_curr = bw
        speed_err = (self.speed_reference - speed_curr) / max(self.speed_reference, 1e-6)
        speed_err = max(-1.5, min(1.5, speed_err))  # clamp outliers
        self.get_logger().info(f'[Speed] Ref: {self.speed_reference}; Curr: {speed_curr}; Err: {speed_err}')

        heading_curr = cx
        # heading_err = (self.heading_reference - heading_curr) / max(heading_curr, 1e-6)
        # normalize by half width so err ~ [-1,1] when target spans center ± half-width
        heading_err = (self.heading_reference - heading_curr) / max(self.heading_norm, 1e-6)
        heading_err = max(-1.5, min(1.5, heading_err))
        self.get_logger().info(f'[Heading] Ref: {self.heading_reference}; Curr: {heading_curr}; Err: {heading_err}')

        speed = self.pid_speed.step(speed_err, dt)
        heading = self.pid_heading.step(heading_err, dt)

        # deadbands (on normalized error)
        if abs(speed_err) < self.speed_db: speed = 0.0
        if abs(heading_err) < self.heading_db: heading = 0.0
        
        if abs(heading) > self.turn_coeff*self.heading_max:
            speed = 0.0
        
        # slew-rate limit commands
        speed   = self.__slew(self.prev_speed, speed, self.max_speed_slew)
        heading = self.__slew(self.prev_heading, heading, self.max_heading_slew)

        self.prev_speed, self.prev_heading = speed, heading

        return speed, heading

    def __plan_pidgs(self, cx, bw, bh, dt):
        """Mnemonic: “Turn = Tighten heading, Tame speed.” Heading: P↑ D↑ as σ_turn↑. Speed: P↓ I↓ as σ_turn↑."""
        # normalized errors
        speed_curr = bw
        speed_err = (self.speed_reference - speed_curr) / max(self.speed_reference, 1e-6)
        speed_err = max(-1.5, min(1.5, speed_err))

        heading_curr = cx
        heading_err = (self.heading_reference - heading_curr) / max(self.heading_norm, 1e-6)
        heading_err = max(-1.5, min(1.5, heading_err))

        # schedule variable: turn severity in [0,1]
        sigma_turn = clamp01(abs(heading_err) / max(self.heading_thr, 1e-6))

        # proximity in [0,1]: 0=far (small bw), 1=near (large bw)
        sigma_dist = clamp01((bw - self.bw_far) / max(self.bw_near - self.bw_far, 1e-6))
        # sigma_speed = max(sigma_turn, self.alpha_dist * sigma_dist)
        sigma_speed = clamp01((1.0 - self.alpha_dist) * sigma_turn + self.alpha_dist * sigma_dist)

        # interpolate heading gains
        h_kp = self.H_KP_E + sigma_turn * (self.H_KP_H - self.H_KP_E)
        h_ki = self.H_KI_E + sigma_turn * (self.H_KI_H - self.H_KI_E)
        h_kd = self.H_KD_E + sigma_turn * (self.H_KD_H - self.H_KD_E)
        self.pid_heading.set_gains(h_kp, h_ki, h_kd)

        # interpolate speed gains (more cautious when turning)
        # s_kp = self.S_KP_E + sigma_turn * (self.S_KP_H - self.S_KP_E)
        # s_ki = self.S_KI_E + sigma_turn * (self.S_KI_H - self.S_KI_E)
        # s_kd = self.S_KD_E + sigma_turn * (self.S_KD_H - self.S_KD_E)
        # self.pid_speed.set_gains(s_kp, s_ki, s_kd)
        s_kp = self.S_KP_E + sigma_speed * (self.S_KP_H - self.S_KP_E)
        s_ki = self.S_KI_E + sigma_speed * (self.S_KI_H - self.S_KI_E)
        s_kd = self.S_KD_E + sigma_speed * (self.S_KD_H - self.S_KD_E)
        self.pid_speed.set_gains(s_kp, s_ki, s_kd)

        self.get_logger().info(
            f"[Scheduler] sigma_turn={sigma_turn:.2f} sigma_dist={sigma_dist:.2f} sigma_speed={sigma_speed:.2f} "
            f"| HeadingGains(Kp,Ki,Kd)=({h_kp:.2f},{h_ki:.2f},{h_kd:.2f}) "
            f"| SpeedGains(Kp,Ki,Kd)=({s_kp:.2f},{s_ki:.2f},{s_kd:.2f})"
        )

        # compute speed and heading with a PID step
        speed = self.pid_speed.step(speed_err, dt)
        heading = self.pid_heading.step(heading_err, dt)

        # deadbands on normalized errors
        if abs(speed_err) < self.speed_db: speed = 0.0
        if abs(heading_err) < self.heading_db: heading = 0.0

        # zero linear speed if turning too hard (safety / stability)
        if abs(heading) > self.turn_coeff * self.heading_max:
            speed = 0.0

        # slew-rate limits
        speed   = self.__slew(self.prev_speed, speed, self.max_speed_slew)
        heading = self.__slew(self.prev_heading, heading, self.max_heading_slew)

        self.prev_speed, self.prev_heading = speed, heading

        # if int(self.get_clock().now().nanoseconds * 1e-9) % 1 == 0:
        #     self.get_logger().info(
        #         f"[CmdOut] Speed={speed:.3f} Heading={heading:.3f} | "
        #         f"sigma_turn={sigma_turn:.2f}, sigma_dist={sigma_dist:.2f}"
        #     )

        return speed, heading

    # def __plan_pp(self, cx, bw, bh, dt):
    #     """
    #     Pure Pursuit steering:
    #     alpha = bearing of target centroid relative to camera optical axis.
    #     kappa = 2*sin(alpha)/L_d
    #     omega = v * kappa
    #     We schedule v down with |alpha| and proximity (from bbox width).
    #     """
    #     # Pixel -> angle
    #     # Map center offset to angle using camera HFOV
    #     # center_x in pixels, normalized u in [-1,1] across width
    #     u = (cx - 0.5*self.frame_width) / max(0.5*self.frame_width, 1e-6)
    #     alpha = -u * (self.pp_hfov * 0.5)   # small-angle ok; exact linear map of pixels to angle
        
    #     # Turn severity in [0,1]
    #     sigma_turn = clamp01(abs(alpha) / max(self.pp_hfov*0.5, 1e-6))  # 0 at center, 1 at edge
        
    #     # Proximity in [0,1] using the same bw_far/bw_near you set
    #     sigma_dist = 0.0
    #     if self.bw_far is not None and self.bw_near is not None:
    #         sigma_dist = clamp01((bw - self.bw_far) / max(self.bw_near - self.bw_far, 1e-6))
        
    #     # Blend speed penalty: more penalty if turning and/or near
    #     sigma_speed = clamp01((1.0 - self.pp_w_dist) * sigma_turn + self.pp_w_dist * sigma_dist)
        
    #     # Speed schedule: vmax -> vmin as sigma_speed -> 1
    #     v_des = (1.0 - sigma_speed*self.pp_turn_slow) * self.pp_vmax
    #     v_des = max(self.pp_vmin, min(self.pp_vmax, v_des))
        
    #     # Pure Pursuit curvature and omega
    #     # Guard L_d
    #     Ld = max(0.1, self.pp_Ld)
    #     kappa = 2.0 * math.sin(alpha) / Ld
    #     omega_des = v_des * kappa
        
    #     # Stability guard: if turning too hard, optionally suppress linear speed
    #     if abs(omega_des) > self.turn_coeff * self.heading_max:
    #         v_des = 0.0
        
    #     # Slew-rate limit using your existing helper
    #     v_cmd   = self.__slew(self.prev_speed,   v_des,   self.max_speed_slew)
    #     w_cmd   = self.__slew(self.prev_heading, omega_des, self.max_heading_slew)
    #     self.prev_speed, self.prev_heading = v_cmd, w_cmd
        
    #     # Small deadband on alpha -> zero tiny omega
    #     if abs(alpha) < self.heading_db * (self.pp_hfov*0.5):  # reuse your concept of deadband
    #         w_cmd = 0.0
        
    #     # Log at low rate
    #     self.get_logger().info(
    #         f"[PP] u={u:.3f} alpha={alpha:.3f} rad | sigma_turn={sigma_turn:.2f} sigma_dist={sigma_dist:.2f} "
    #         f"| v_des={v_des:.3f} m/s omega={omega_des:.3f} rad/s -> v={v_cmd:.3f}, w={w_cmd:.3f}"
    #     )

    #     return v_cmd, w_cmd

    def __plan_pp(self, cx, bw, bh, dt):
        # earing (PP)
        u = (cx - 0.5*self.frame_width) / max(0.5*self.frame_width, 1e-6)
        # make left target => positive alpha => CCW (left) turn
        alpha = -u * (self.pp_hfov * 0.5)
        kappa = 2.0 * math.sin(alpha) / max(0.1, self.pp_Ld)

        # turn/proximity severities for speed shaping
        sigma_turn = clamp01(abs(alpha) / max(self.pp_hfov*0.5, 1e-6))
        sigma_dist = 0.0
        if self.bw_far is not None and self.bw_near is not None:
            sigma_dist = clamp01((bw - self.bw_far) / max(self.bw_near - self.bw_far, 1e-6))
        sigma_speed = clamp01((1.0 - self.pp_w_dist) * sigma_turn + self.pp_w_dist * sigma_dist)

        # Distance control for linear speed
        bw_ref = max(1.0, self.speed_reference)  # set on first frame
        e = (bw - bw_ref) / bw_ref               # + when too close, - when far
        # Deadband near target size
        if abs(e) < self.pp_stop_db:
            v_des = 0.0
        else:
            # Proportional on distance error; negative e => forward; positive e => reverse
            v_des = - self.pp_kdist * e * self.pp_vmax
            # Slow down further if turning/proximate
            v_des *= (1.0 - 0.6 * sigma_speed)   # 0.6 is conservative; tune 0.4–0.8

        # Clip forward and reverse speeds
        v_fwd_cap = self.pp_vmax
        v_back_cap = self.pp_vback_max if self.pp_reverse_ok else 0.0
        v_des = max(-v_back_cap, min(v_fwd_cap, v_des))

        # “zero linear if turning extremely hard” guard
        if abs(alpha) > 0.9 * (self.pp_hfov * 0.5):
            v_des = 0.0

        # Pure Pursuit steering (decoupled from v sign)
        # Use a nominal speed to scale omega so reversing doesn't flip turn direction.
        # v_for_omega = max(self.pp_vmin, min(self.pp_omega_vnom, abs(v_des)))
        v_for_omega = self.pp_omega_vnom  # always use nominal for steering strength
        omega_des = v_for_omega * kappa

        # Tiny heading deadband (for no jitter)
        if abs(alpha) < self.heading_db * (self.pp_hfov*0.5):
            omega_des = 0.0

        # Apply skew limits
        v_cmd = self.__slew(self.prev_speed, v_des, self.max_speed_slew)
        w_cmd = self.__slew(self.prev_heading, omega_des, self.max_heading_slew)

        self.prev_speed, self.prev_heading = v_cmd, w_cmd

        self.get_logger().info(
            f"[PP] alpha={alpha:.3f} e={e:.3f} "
            f"| v_des={v_des:.3f} (fwd_cap={v_fwd_cap:.2f}, back_cap={v_back_cap:.2f}) "
            f"| kappa={kappa:.3f} omega={omega_des:.3f} -> v={v_cmd:.3f}, w={w_cmd:.3f}"
        )

        return v_cmd, w_cmd

    def _plan(self, cx, bw, bh, dt):
        if self.controller == 'pid':
            return self.__plan_pid(cx, bw, bh, dt)
        elif self.controller == 'pidgs':
            return self.__plan_pidgs(cx, bw, bh, dt)
        elif self.controller == 'pp':
            return self.__plan_pp(cx, bw, bh, dt)
        else:
            return 0.0, 0.0

    def _follow(self, speed, heading):
        cmd_vel = Twist()

        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def _set_state(self, new_state: str, now_sec: float):
        if self.sc.state != new_state:
            self.get_logger().info(f"[SEARCH] {self.sc.state} -> {new_state}")
        self.sc.state = new_state
        self.sc.state_start_sec = now_sec
        self.sc.spin_accum_yaw = 0.0
        self.sc.commit_sec = 0.0

    def _spin_step(self, dt):
        # spin at configured rate (rad/s)
        w = 2 * math.pi * self.spin_rate_hz / (2 * math.pi)  # simplify: ~spin_rate_hz rad/s
        w = max(0.4, min(1.2, self.spin_rate_hz))            # clamp
        return 0.0, np.sign(1.0) * w

    def _search_cmd(self, frame, dt, now_sec):
        """
        Returns (v, w) during search state machine, wrapped with safety.
        Uses synthetic cx to drive chosen controller toward desired heading.
        """
        # Predict hypothesis forward
        self.tgt.predict(dt)

        if self.sc.state == 'TRACK':
            # Should not be here (only when ball lost, we switch); fallthrough
            pass

        # SPIN_SCAN: 360° sweep to seed gaps and chance to see ball
        if self.sc.state == 'SPIN_SCAN':
            # spin until accumulated yaw passes target
            v, w = 0.0, 0.9
            self.sc.spin_accum_yaw += abs(w) * dt
            if self.sc.spin_accum_yaw >= self.sc.spin_target:
                self._set_state('SELECT_GAP', now_sec)
            return self._safety_wrap(v, w)

        # SELECT_GAP: score gaps and choose target
        if self.sc.state == 'SELECT_GAP':
            gaps = self._scan_gaps(self.gap_min_width_m)
            best = self._score_gaps(frame, gaps)
            if best is None:
                self.sc.spin_target = 2 * math.pi
                self._set_state('SPIN_SCAN', now_sec)
                return self._safety_wrap(0.0, 0.8)

            # Normalize desired bearing and set waypoint
            self.sc.goal_angle = wrap_pi(float(best['theta']))
            depth = float(best['depth'])
            gx = self.sc.x + depth * math.cos(self.sc.yaw + self.sc.goal_angle)
            gy = self.sc.y + depth * math.sin(self.sc.yaw + self.sc.goal_angle)
            self.sc.goal_waypoint = (gx, gy)
            self.sc.commit_sec = 0.0

            # NEW: go align first (fast turn-in-place) before GO_TO_GAP
            self._set_state('ALIGN', now_sec)

        # ALIGN: rotate quickly to face the chosen gap (bearing to waypoint NOW)
        if self.sc.state == 'ALIGN':
            wx, wy = self.sc.goal_waypoint
            # Bearing from current odom pose to waypoint, in robot frame
            desired = math.atan2(wy - self.sc.y, wx - self.sc.x) - self.sc.yaw
            berr = wrap_pi(desired)

            aligned = abs(berr) < math.radians(15.0)  # 10–20° works well

            # Fast, pure rotation toward the waypoint
            # Scale a bit with error (proportional clamp) to avoid hunting
            w_mag = 0.8 * max(0.25, min(1.0, abs(berr) / math.radians(60.0)))
            w_cmd = w_mag * (1.0 if berr >= 0.0 else -1.0)
            v_cmd = 0.0 if not aligned else 0.0  # still zero when aligned; GO_TO_GAP handles forward

            if aligned:
                self._set_state('GO_TO_GAP', now_sec)
                return self._safety_wrap(0.0, 0.0)

            return self._safety_wrap(v_cmd, w_cmd)

        if self.sc.state == 'GO_TO_GAP':
            # Re-score occasionally with hysteresis; if a much better angle appears, re-align
            self.sc.replan_ticker += dt
            if self.sc.replan_ticker >= 1.0 / max(0.5, self.replan_hz):
                self.sc.replan_ticker = 0.0
                gaps = self._scan_gaps(self.gap_min_width_m)
                alt = self._score_gaps(frame, gaps)
                if alt is not None:
                    new_theta = wrap_pi(float(alt['theta']))
                    if abs(new_theta) < 0.8 * abs(wrap_pi(new_theta)):  # relaxed check
                        # Update waypoint to the new gap and re-align
                        depth = float(alt['depth'])
                        gx = self.sc.x + depth * math.cos(self.sc.yaw + new_theta)
                        gy = self.sc.y + depth * math.sin(self.sc.yaw + new_theta)
                        self.sc.goal_waypoint = (gx, gy)
                        self._set_state('ALIGN', now_sec)
                        return self._safety_wrap(0.0, 0.0)

            # Compute live bearing error (waypoint - current heading)
            wx, wy = self.sc.goal_waypoint
            desired = math.atan2(wy - self.sc.y, wx - self.sc.x) - self.sc.yaw
            berr = wrap_pi(desired)
            aligned = abs(berr) < math.radians(15.0)

            # Build a synthetic pixel target from the CLAMPED bearing alpha
            edge = 0.85 * (self.pp_hfov * 0.5)
            alpha = max(-edge, min(edge, berr))
            u = -alpha / (self.pp_hfov * 0.5)
            cx_syn = 0.5 * self.frame_width + u * (0.5 * self.frame_width)
            bw_syn = self.bw_far
            bh_syn = bw_syn

            # If aligned, let your controller translate. If not, keep nudging rotation.
            if aligned:
                v_des, w_des = self._plan(cx_syn, bw_syn, bh_syn, dt)
            else:
                # gentle rotate-only nudge
                v_des, w_des = 0.0, 0.4 * (1.0 if berr >= 0.0 else -1.0)

            v_out, w_out = self._safety_wrap(v_des, w_des)

            # Only start commit/arrival logic once aligned (prevents early LOCAL_SCAN flips)
            if aligned:
                self.sc.commit_sec += dt

                # Arrival when close enough to the waypoint
                if math.hypot(wx - self.sc.x, wy - self.sc.y) < max(0.8 * self.gap_min_depth_m, 0.6):
                    self._mark_visited(self.sc.x, self.sc.y)
                    self._set_state('LOCAL_SCAN', now_sec)
                    return self._safety_wrap(0.0, 0.0)

                # If stuck with v≈0 for a while, peek locally
                if abs(v_out) < 1e-2 and self.sc.commit_sec > max(2.0, self.min_commit_s):
                    self._mark_visited(self.sc.x, self.sc.y)
                    self._set_state('LOCAL_SCAN', now_sec)
                    return self._safety_wrap(0.0, 0.0)

            return v_out, w_out

        # LOCAL_SCAN: short scan at waypoint, mark visited, then reselect
        if self.sc.state == 'LOCAL_SCAN':
            # mark visited cell (after we stop)
            self._mark_visited(self.sc.x, self.sc.y)

            # spin quickly for a limited angle (config)
            v, w = 0.0, 1.0
            self.sc.spin_accum_yaw += abs(w) * dt
            target = math.radians(max(90, int(self.local_spin_deg)))
            if self.sc.spin_accum_yaw >= target:
                self._set_state('SELECT_GAP', now_sec)
                return self._safety_wrap(0.0, 0.0)
            return self._safety_wrap(v, w)

        # RECOVER (rare): back-off & reorient
        if self.sc.state == 'RECOVER':
            # short reverse and slight turn
            v, w = -0.08, 0.5
            self.sc.commit_sec += dt
            if self.sc.commit_sec >= 1.0:
                self._set_state('SPIN_SCAN', now_sec)
                return self._safety_wrap(0.0, 0.0)
            return self._safety_wrap(v, w)

        # default
        return self._safety_wrap(0.0, 0.0)

    def listener_callback(self, msg):
        # Logs the received messages from a topic
        # self.get_logger().info(f'Message Type: {type(msg)}')

        # Convert ros to cv image type
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Compute dt
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        dt = now_sec - self.prev_sec if self.prev_sec is not None else 1.0 / 10.0
        self.prev_sec = now_sec

        if self.first_frame:
            self.frame_height, self.frame_width = frame.shape[:2]
            # self.speed_reference = 0.5 * 0.3 * self.frame_width + 0.5 * 0.3 * self.frame_height
            self.speed_reference = 0.08 * self.frame_width
            self.heading_reference = 0.5 * self.frame_width
            self.heading_norm = 0.5 * self.frame_width
            self.bw_far  = 0.05 * self.frame_width   # “comfortably far”
            self.bw_near = 0.18 * self.frame_width   # “quite close”
            self.first_frame = False
            self.get_logger().info('Frame Parameters are set!')

        result = self._detect(frame)
        if result is not None:
            self.get_logger().info('Red Ball Detected!')
            cx, cy, w, h, (x, y, bw, bh) = result
            # Logging pixels from top-left origin
            self.get_logger().info(f'[Object - Red Ball] centroid=({cx:.1f},{cy:.1f}) size=({w:.0f},{h:.0f})')

            # Draw bbox + centroid
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

            # Estimate approximate range from bbox width (rough pinhole proxy)
            # Need a scale; use proportional inverse of width, normalized by frame width.
            # This is heuristic; adequate for biasing search hypothesis.
            norm = max(1.0, float(self.frame_width))
            approx_range = max(0.4, min(3.5, 0.9 * (norm / max(10.0, float(bw)))))  # tuneable clamp

            # Convert pixel bearing to angle in robot frame
            u = (cx - 0.5*self.frame_width) / max(0.5*self.frame_width, 1e-6)
            alpha = -u * (self.pp_hfov * 0.5)

            # Update hypothesis in odom frame
            if self.sc.have_odom:
                hx = self.sc.x + approx_range * math.cos(self.sc.yaw + alpha)
                hy = self.sc.y + approx_range * math.sin(self.sc.yaw + alpha)
                self.tgt.update(hx, hy, max(dt, 1e-3))

            speed, heading = self._plan(cx, bw, bh, dt)

            # reset search state to TRACK
            self._set_state('TRACK', now_sec)
        else:
            self.get_logger().info('No Red Ball Detected!')
            if self.controller in ('pid', 'pidgs'):
                if not self.pid_speed.is_reset():
                    self.pid_speed.reset()
                    self.prev_speed = 0.0
                    self.get_logger().info('Speed PID Reset!')
                if not self.pid_heading.is_reset():
                    self.pid_heading.reset()
                    self.prev_heading = 0.0
                    self.get_logger().info('Heading PID Reset!')
            
            if self.search_mode != 'none':
                # Enter search state if coming from TRACK
                if self.sc.state == 'TRACK':
                    self.sc.spin_target = 2 * math.pi
                    self._set_state('SPIN_SCAN', now_sec)

                # Run search machine
                speed, heading = self._search_cmd(frame, dt, now_sec)
            else:
                speed, heading = 0.0, 0.0

        if self.log_prev_speed != speed and self.log_prev_heading != heading:
            self.get_logger().info(f'[Robot] Speed: {speed}; Heading: {heading}')
            self.log_prev_speed, self.log_prev_heading = speed, heading
        
        self._follow(speed, heading)

        # Show or display frames on a window
        if self.show_video:
            cv.imshow('Red Ball Tracker: Frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.show_video = False
                cv.destroyAllWindows()


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'RedBallTracker' class node
    red_ball_tracker = RedBallTracker()

    # Spin up the created node
    rclpy.spin(red_ball_tracker)

    # Explicit declaration to destroy the node object and shutdown rclpy
    red_ball_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
