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
from task_6.utils.helper import clamp01

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
        self.declare_parameter('search_mode', 'none')   # 'none', 'frontier', 'wall_following'
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
        self.pp_vmax = 0.42  # cap linear speed for PP
        self.pp_vmin = 0.24  # floor linear speed so it moves
        self.pp_turn_slow = 0.8  # how much turning reduces speed (0..1)
        self.pp_w_dist = 0.6  # blend weight of distance vs turn
        self.pp_kdist      = 0.8       # gain on normalized distance error
        self.pp_stop_db    = 0.08     # deadband on |(bw - ref)/ref| ; e.g., ±8% width
        self.pp_vback_max  = 0.42  # max reverse speed (0 to disable backing up)
        self.pp_omega_vnom = 0.18  # nominal speed used to scale omega (steer strength)
        self.pp_reverse_ok = True  # allow reverse when too close

        self.scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        # Search params
        self.search_rotate = True

        # Odometry / yaw tracking
        self.accum_yaw = 0.0        # accumulated rotation magnitude [rad]
        self.odom_yaw = None
        self.prev_odom_yaw = None

        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # Command Velocity Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.speed, self.heading = 0.0, 0.0
    
    def _scan_cb(self, msg: LaserScan):
        self.scan = msg
    
    def _quat_to_yaw(self, q):
        # q: geometry_msgs/Quaternion
        # returns yaw in [-pi, pi]
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _ang_wrap_pi(self, a):
        # wrap to [-pi, pi]
        while a > math.pi:  a -= 2.0 * math.pi
        while a < -math.pi: a += 2.0 * math.pi
        return a

    def _ang_diff(self, a, b):
        # smallest signed diff a-b in [-pi, pi]
        return self._ang_wrap_pi(a - b)

    def _odom_cb(self, msg: Odometry):
        yaw = self._quat_to_yaw(msg.pose.pose.orientation)
        self.odom_yaw = yaw
        if self.prev_odom_yaw is None:
            self.prev_odom_yaw = yaw

        d = self._ang_diff(self.odom_yaw, self.prev_odom_yaw)
        # accumulate absolute rotation regardless of direction
        self.accum_yaw += d  # abs(d)
        self.prev_odom_yaw = self.odom_yaw

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

    def _follow(self):
        cmd_vel = Twist()

        cmd_vel.linear.x = self.speed
        cmd_vel.angular.z = self.heading

        self.cmd_vel_pub.publish(cmd_vel)

    def _wrap_to_span(self, a, amin, amax):
        """Wrap angle a (rad) into [amin, amax] by adding/subtracting 2π."""
        twopi = 2.0 * math.pi
        while a < amin:
            a += twopi
        while a > amax:
            a -= twopi
        return a

    def _scan_value(self, deg):
        """
        Return range at 'deg'
        Mapping: sensor_angle_rad = radians(deg).
        If out of bounds or non-finite, returns None.
        """
        if self.scan is None or not self.scan.ranges:
            return None

        scan = self.scan

        if deg > len(scan.ranges):
            self.get_logger().info(f'Invalid degree provided!: {deg}')
        if deg == 360:
            deg = 0

        r = scan.ranges[deg]
        if r is None or (isinstance(r, float) and not math.isfinite(r)):
            return None
        return float(r)

    # ------------- GAP FINDING (segments of indices and angles) -------------
    def _scan_to_numpy(self):
        """Return (ranges_np, angle_min, angle_inc, angle_max) or (None,...) if no scan."""
        if self.scan is None or not self.scan.ranges:
            return None, None, None, None
        scan = self.scan
        # Use numpy for fast masking; keep NaN/inf as-is for detection
        rng = np.array(scan.ranges, dtype=float)
        self.get_logger().info(f'[SCAN] Range Len: {len(rng)}, Angle Min: {scan.angle_min}, Angle Increment: {scan.angle_increment}, Angle Max: {scan.angle_max}')
        return rng, scan.angle_min, scan.angle_increment, scan.angle_max

    def _mask_to_segments(self, mask: np.ndarray):
        """
        Given a boolean mask over indices, return list of contiguous (start_idx, end_idx) inclusive.
        Handles wrap-around (merge end and start if both True).
        """
        n = mask.size
        if n == 0:
            return []

        # If all False -> no segments
        if not mask.any():
            return []

        # Handle wrap-around by rotating if mask[0] and mask[-1] are True
        # Find runs by diff on padded mask
        m = mask.astype(np.int8)
        # If wrap True at both ends, rotate so a segment doesn't straddle edges
        if m[0] == 1 and m[-1] == 1:
            # find first zero->one transition as a rotation point
            # Build transitions
            dm = np.diff(np.r_[0, m, 0])
            starts = np.where(dm == 1)[0]       # inclusive
            ends   = np.where(dm == -1)[0] - 1  # inclusive
            # choose the longest segment and rotate just after it ends (heuristic),
            # or simply rotate to the first start to break the wrap.
            rot = int(starts[0])  # simple & robust
            m = np.roll(m, -rot)
            rotated = True
        else:
            rot = 0
            rotated = False

        dm = np.diff(np.r_[0, m, 0])
        starts = np.where(dm == 1)[0]
        ends   = np.where(dm == -1)[0] - 1
        segs = list(zip(starts, ends))

        if rotated:
            # rotate indices back
            n = mask.size
            segs = [((s + rot) % n, (e + rot) % n) for (s, e) in segs]
            # After un-rotating, segments are correct; nothing straddles edges anymore.

        return segs

    def _segments_to_angles(self, segs, angle_min, angle_inc):
        """
        For each (start_idx, end_idx) inclusive, compute (ang_start, ang_end) in radians.
        Uses the sensor's native frame: angle = angle_min + idx * angle_inc.
        """
        out = []
        for s, e in segs:
            ang_s = angle_min + s * angle_inc
            ang_e = angle_min + e * angle_inc
            out.append((s, e, ang_s, ang_e))
        return out

    # def _find_gap_segments(self, mode='nan', radius=None, finite_only=True):
    #     """
    #     mode: 'nan'  -> gaps are runs of non-finite (inf/NaN).
    #         'radius' -> gaps are runs where range > radius (optionally also must be finite).
    #     radius: float, used only when mode='radius'.
    #     finite_only (radius mode): if True, only counts finite readings > radius as gaps;
    #                             if False, non-finite are also treated as gaps.
    #     returns: list of tuples (start_idx, end_idx, start_ang, end_ang) inclusive indices,
    #             angles in sensor frame (radians).
    #     """
    #     rng, a_min, a_inc, a_max = self._scan_to_numpy()
    #     if rng is None:
    #         return []

    #     if mode == 'nan':
    #         mask = ~np.isfinite(rng)  # True where inf/NaN
    #     elif mode == 'radius':
    #         if radius is None or radius <= 0:
    #             # Sensible default: slightly less than max range if available
    #             # If max_range is available in LaserScan, you could use self.scan.range_max.
    #             radius = getattr(self.scan, 'range_max', np.nanmax(np.where(np.isfinite(rng), rng, 0.0)) * 0.9) or 1.0
    #         if finite_only:
    #             mask = (np.isfinite(rng)) & (rng > radius)
    #         else:
    #             # treat non-finite also as gaps
    #             mask = (~np.isfinite(rng)) | (rng > radius)
    #     else:
    #         # Unknown mode -> no segments
    #         return []

    #     segs = self._mask_to_segments(mask)
    #     return self._segments_to_angles(segs, a_min, a_inc)

    def _find_gap_segments(self, mode='nan', radius=None, finite_only=True, front_only=True):
        """
        mode: 'nan'     -> gaps are runs of non-finite (inf/NaN)
            'radius'  -> gaps are runs where range > radius
        radius: threshold used when mode='radius'
        finite_only (radius mode): True -> only finite > radius; False -> OR with non-finite
        front_only: if True, restrict detection to sensor angles in [-pi/2, +pi/2]
        returns: list of (start_idx, end_idx, start_ang, end_ang) [angles: sensor frame, rad]
        """
        rng, a_min, a_inc, a_max = self._scan_to_numpy()
        if rng is None:
            return []

        # 1) base mask by gap definition
        if mode == 'nan':
            base = ~np.isfinite(rng)
        elif mode == 'radius':
            if radius is None or radius <= 0:
                radius = getattr(self.scan, 'range_max', None)
                if radius is None or not np.isfinite(radius):
                    # fallback: 90% of finite max or 1.0
                    finite_vals = rng[np.isfinite(rng)]
                    radius = (0.9 * finite_vals.max()) if finite_vals.size else 1.0
            if finite_only:
                base = (np.isfinite(rng)) & (rng > radius)
            else:
                base = (~np.isfinite(rng)) | (rng > radius)
        else:
            return []

        # 2) optional front-only mask: sensor angles in [-pi/2, +pi/2]
        if front_only:
            n = rng.size
            idx = np.arange(n, dtype=np.int64)
            ang = a_min + idx * a_inc
            front_mask = ((ang >= 0) & (ang <= math.pi/2)) | (ang >= 3*math.pi/2)
            mask = base & front_mask
        else:
            mask = base

        segs = self._mask_to_segments(mask)
        return self._segments_to_angles(segs, a_min, a_inc)

    # Optional: convert sensor angle (rad) -> your compass style degrees
    def _sensor_rad_to_deg(self, ang_rad: float) -> float:
        """
        For logging convenience: convert a sensor-frame angle to compass degrees
        """
        # Inverse of what _scan_value does:
        # sensor_angle = radians(deg)  => deg = degrees(sensor_angle)
        return math.degrees(ang_rad)

    def _split_three_segments(self, s: int, e: int):
        """
        Split inclusive index range [s, e] into 3 near-equal inclusive sub-segments.
        Returns [(s1, e1), (s2, e2), (s3, e3)].
        """
        idxs = np.arange(s, e + 1, dtype=int)
        parts = np.array_split(idxs, 3)
        out = []
        for p in parts:
            if p.size == 0:
                out.append((None, None))
            else:
                out.append((int(p[0]), int(p[-1])))
        return out

    def _mid_index(self, s: int, e: int) -> int:
        """Inclusive mid index of [s, e]."""
        return int((s + e) // 2)

    def _idx_to_ang(self, idx: int) -> float:
        """Sensor-frame angle (rad) for index."""
        if self.scan is None:
            return 0.0
        return self.scan.angle_min + idx * self.scan.angle_increment

    def _log_gaps_with_thirds(self, label: str, segments):
        """
        segments: list of (s_idx, e_idx, a_s, a_e).
        For each gap, log 3 sub-segments + mid index/angle for each sub-segment.
        """
        if not segments:
            self.get_logger().info(f"[Gaps {label}] none (front-only)")
            return

        self.get_logger().info(f"[Gaps {label}] count={len(segments)} (front-only)")
        for gi, (s, e, a_s, a_e) in enumerate(segments):
            self.get_logger().info(
                f"  gap#{gi:02d} idx[{s}-{e}] | ang[{a_s:+.3f},{a_e:+.3f}] rad "
                f"| front-deg[{self._sensor_rad_to_deg(a_s):.1f},"
                f"{self._sensor_rad_to_deg(a_e):.1f}]"
            )
            thirds = self._split_three_segments(s, e)
            for ti, (ts, te) in enumerate(thirds):
                if ts is None:
                    self.get_logger().info(f"    seg{ti}: <empty>")
                    continue
                mid = self._mid_index(ts, te)
                ang_ts, ang_te = self._idx_to_ang(ts), self._idx_to_ang(te)
                ang_mid = self._idx_to_ang(mid)
                self.get_logger().info(
                    f"    seg{ti}: idx[{ts}-{te}] mid={mid} | "
                    f"ang[{ang_ts:+.3f},{ang_te:+.3f}] mid={ang_mid:+.3f} rad | "
                    f"front-deg[{self._sensor_rad_to_deg(ang_ts):.1f},"
                    f"{self._sensor_rad_to_deg(ang_te):.1f}] "
                    f"mid={self._sensor_rad_to_deg(ang_mid):.1f}"
                )
    
    def _log_scan(self):
        if self.scan is not None:
            probe_angles = [0, 45, 90, 135, 180, 225, 270, 315]
            readings = []
            for a in probe_angles:
                v = self._scan_value(a)
                readings.append(f"{a}:{('%.2f' % v) if v is not None else 'nan'}")
            self.get_logger().info("[ScanProbe] " + " ".join(readings))
        else:
            self.get_logger().info("[ScanProbe] No /scan data yet!")
    
    def _log_yaw(self):
        self.get_logger().info(f'Accum Yaw: {self.accum_yaw}')

    def _log_gaps(self):
        # --- NEW: find gap segments by (A) non-finite readings ---
        nan_segments = self._find_gap_segments(mode='nan', front_only=True)
        self._log_gaps_with_thirds("non-finite", nan_segments)
        # if nan_segments:
        #     self.get_logger().info(f"[Gaps: non-finite] count={len(nan_segments)}")
        #     for (s, e, a_s, a_e) in nan_segments:
        #         self.get_logger().info(
        #             f"  seg idx[{s}-{e}] | ang[{a_s:+.3f},{a_e:+.3f}] rad "
        #             f"| front-deg[{self._sensor_rad_to_deg(a_s):.1f},"
        #             f"{self._sensor_rad_to_deg(a_e):.1f}]"
        #         )
        # else:
        #     self.get_logger().info("[Gaps: non-finite] none")

        # --- NEW: find gap segments by (B) radius threshold (> radius) ---
        # choose whatever radius makes sense for your environment; e.g., 2.0 m
        RADIUS = 2.0
        rad_segments = self._find_gap_segments(mode='radius', radius=RADIUS, finite_only=True, front_only=True)
        self._log_gaps_with_thirds(f">{RADIUS:.1f}m", rad_segments)
        # if rad_segments:
        #     self.get_logger().info(f"[Gaps: >{RADIUS:.1f}m] count={len(rad_segments)}")
        #     for (s, e, a_s, a_e) in rad_segments:
        #         self.get_logger().info(
        #             f"  seg idx[{s}-{e}] | ang[{a_s:+.3f},{a_e:+.3f}] rad "
        #             f"| front-deg[{self._sensor_rad_to_deg(a_s):.1f},"
        #             f"{self._sensor_rad_to_deg(a_e):.1f}]"
        #         )
        # else:
        #     self.get_logger().info(f"[Gaps: >{RADIUS:.1f}m] none")
    
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

            self.speed, self.heading = self._plan(cx, bw, bh, dt)
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
            
            # Rotate 360
            if self.search_rotate:
                self.speed, self.heading = 0.0, 0.6
                if abs(self.accum_yaw) > 2*math.pi:
                    self.speed, self.heading = 0.0, 0.0
                    self.search_rotate = False

        # self._log_yaw()
            
        # self._log_scan()

        self._log_gaps()

        ## Do a 360 to look for the ball
        ## Find the gaps based on radius mainly and find the first gap on the left and its 2nd segment and its mid angle
        ## Move towards the gap->segment until that angle distance is less than a distance defined (see how to keep track of that pointttttt or angle which was identified in the gap initially, since you will have to turn to head to that point/angle)
        ## During the process, lookout for
        ## ## Obstacles and to avoid it, course correct, by keeping the heading in memory to correct it back to that initial heading
        ## ## Any change in lidar readings on the left, if detected make a left turn and explore
        ## If there are no gaps in front facing, then turn right by 90, check for gaps
        ## Stop every some distance between the bot's initialposition after findig a gap to the gap direction, and perform a 360 to look for the ball.

        if self.log_prev_speed != self.speed and self.log_prev_heading != self.heading:
            self.get_logger().info(f'[Robot] Speed: {self.speed}; Heading: {self.heading}')
            self.log_prev_speed, self.log_prev_heading = self.speed, self.heading
        
        self._follow()

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
