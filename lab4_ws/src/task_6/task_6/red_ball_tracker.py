import math
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from task_6.utils.detect import find_object_hsv, find_object_hsv_triangle, find_object_hsv_circle, LOWER_RED_1, LOWER_RED_2, UPPER_RED_1, UPPER_RED_2
from task_6.utils.pid import PID


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
        self.detector_mode = 'hsv'  # hsv_circle, hsv_triangle or hsv

        # Runtime options/ROS Params
        self.declare_parameter('controller', 'pid')     # 'pid' or 'stanley'
        self.declare_parameter('search_mode', 'none')   # 'none', 'spiral', or 'hybrid'
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

        # Stanley+tanh controller gains
        # self.k_stanley = 0.9
        # self.k_omega   = 1.4
        # self.k_v       = 0.04      # per pixel
        self.v_max     = self.speed_max
        self.eps_v     = 0.1
        # self.e_dead    = 0.03      # deadband on normalized lateral error
        # self.r_dead    = 8         # deadband on size error (px)
        self.k_stanley = 0.8
        self.k_omega   = 1.0
        self.k_v       = 0.03
        self.e_dead    = 0.02
        self.r_dead    = 10

        # Search/reacquire state
        self.last_seen_t   = None
        self.last_seen_cx  = None
        self.search_phase  = 0.0   # spiral phase accumulator
        self.search_omega0 = 0.8   # rad/s at start of spiral
        self.search_decay  = 0.98  # per cycle decay of |omega|
        self.search_v      = 0.08  # m/s forward in reacquire
        self.reacquire_window = 2.0  # seconds to try spiral after loss

        self.scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        # FSM
        self.mode = 'TRACKING'   # TRACKING | DOD | GVH | ISP | PTS
        self.room_visit_count = 0
        self.heading_history = []      # avoid re-trying same gap/door headings
        self.heading_hist_len = 6

        # ISP sweep accumulators
        self.isp_bins        = 24        # number of yaw bins per 360° sweep
        self.isp_bin_time    = 0.22      # seconds spent in each yaw bin
        self.isp_peak_ratio  = 1.2       # (peak / median) threshold for red likelihood
        self.isp_scores = np.zeros(self.isp_bins, dtype=np.float32)
        self.isp_elapsed = 0.0
        self.isp_active = False

        # Door/gap thresholds
        self.door_min_width_m = 0.7      # minimum door opening width (meters)
        self.door_min_depth_m = 1.2      # must be this deep to be considered a doorway
        self.gap_min_width_m  = 0.6      # minimum gap for a valid vantage hop (meters)
        self.hop_distance_m   = 1.7      # forward distance per gap hop (meters)

        # PTS-mini
        self.pts_stripes     = 2         # number of lawnmower stripes
        self.pts_spacing_m   = 0.7       # side offset between stripes
        self.pts_forward_s   = 2.0       # seconds to move forward per stripe leg
        self.pts_state = {'leg': 0, 'dir': 1, 'timer': 0.0}        

        # Command Velocity Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _red_likelihood(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # simple red mask
        m = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        return float(cv.countNonZero(m)) / float(m.size)  # fraction in [0,1]

    def _scan_gaps(self, width_m):
        if self.scan is None: return []
        r = np.array(self.scan.ranges, dtype=np.float32)
        ang0 = self.scan.angle_min
        dang = self.scan.angle_increment
        valid = np.isfinite(r)

        # Consider “open” if valid and not too close; use 1.0 m as a softer floor
        farish = r > 1.0
        open_mask = valid & farish

        gaps = []
        i = 0
        while i < len(open_mask):
            if open_mask[i]:
                j = i
                while j < len(open_mask) and open_mask[j]:
                    j += 1

                seg = r[i:j][np.isfinite(r[i:j])]
                if seg.size == 0:
                    i = j
                    continue

                depth = float(np.quantile(seg, 0.3))  # robust depth
                span  = (j - i) * dang
                gap_width = depth * span

                if gap_width >= width_m:
                    ang_center = ang0 + (i + j - 1) * 0.5 * dang
                    gaps.append((ang_center, gap_width))
                i = j
            else:
                i += 1

        gaps.sort(key=lambda t: t[1], reverse=True)
        return gaps

    def _pick_new_heading(self, candidates):
        """Pick heading not in recent history (within ~15°)."""
        if not candidates: return None
        def far_from_hist(a):
            for h in self.heading_history[-self.heading_hist_len:]:
                if abs((a - h + np.pi) % (2*np.pi) - np.pi) < np.deg2rad(15):
                    return False
            return True
        for a, _ in candidates:
            if far_from_hist(a): return a
        return candidates[0][0]  # fallback

    def _isp_step(self, frame, dt):
        if not self.isp_active:
            self.isp_scores[:] = 0.0
            self.isp_elapsed = 0.0
            self.isp_active = True

        # rotate slowly and accumulate red score into current bin
        omega_cmd = 0.3   # rad/s sweep
        v_cmd     = 0.0

        # which bin are we in?
        # map image center offset -> yaw bin by time (uniform sweep)
        self.isp_elapsed += dt
        sweep_T = self.isp_bins * self.isp_bin_time
        # clamp to [0, sweep_T]
        t = min(self.isp_elapsed, sweep_T - 1e-3)
        b = int(t // self.isp_bin_time)
        b = np.clip(b, 0, self.isp_bins - 1)

        self.isp_scores[b] += self._red_likelihood(frame)

        # done?
        done = self.isp_elapsed >= sweep_T

        return v_cmd, omega_cmd, done

    def _isp_pick_heading(self):
        s = self.isp_scores
        if np.all(s == 0): return None
        total = float(np.sum(s))
        med = float(np.median(s[s > 0])) if np.any(s > 0) else 0.0
        peak_idx = int(np.argmax(s)); peak = float(s[peak_idx])

        # accept if clearly above background OR if overall red is significant
        if (med == 0.0) or (peak / max(med, 1e-6) >= self.isp_peak_ratio) or (total >= 0.10 * self.isp_bins):
            theta = (peak_idx / self.isp_bins) * 2*np.pi - np.pi
            return theta
        return None

    def _pts_step(self, dt):
        """Two lawnmower stripes, time-based. Returns (v, omega), done."""
        v = 0.0; omega = 0.0
        self.pts_state['timer'] += dt
        leg = self.pts_state['leg']

        if leg % 2 == 0:
            # forward stripe
            v = 0.12; omega = 0.0
            if self.pts_state['timer'] >= self.pts_forward_s:
                self.pts_state['timer'] = 0.0
                self.pts_state['leg'] += 1
        else:
            # turn 90° alternating left/right
            omega = (1.0 if ((leg//2) % 2 == 0) else -1.0) * 0.6
            v = 0.0
            if self.pts_state['timer'] >= (np.pi/2) / abs(omega):
                self.pts_state['timer'] = 0.0
                self.pts_state['leg'] += 1

        done = self.pts_state['leg'] >= (self.pts_stripes*2 + 1)
        if done:
            self.pts_state = {'leg': 0, 'dir': 1, 'timer': 0.0}
        return v, omega, done

    def _enter_search(self):
        if self.scan is None:
            # No scan → start with spiral, not DOD
            self.mode = 'PTS'   # or a small spiral routine
            self.search_phase = 0.0
            self.isp_active = False
        else:
            self.mode = 'DOD'
            self.search_phase = 0.0
            self.isp_active = False

    def _search_step(self, frame, dt):
        # DOD → GVH → ISP → PTS → (repeat or stop)
        if self.mode == 'DOD':
            gaps = self._scan_gaps(self.door_min_width_m)
            if gaps:
                theta = self._pick_new_heading(gaps)
                if theta is not None:
                    # PID to face doorway and move in
                    cx = self.heading_reference - theta * (0.5 * self.frame_width)  # pseudo target shift
                    bw = 0                          # keep distance control calm
                    v, w = self.__plan_pid(cx, bw, bw, dt)
                    v = max(v, 0.08)                # commit forward
                    # once we're roughly centered, switch to GVH
                    if abs(theta) < np.deg2rad(10):
                        self.heading_history.append(theta)
                        self.mode = 'GVH'
                    return v, w
            # no door detected → directly try GVH
            self.mode = 'GVH'

        if self.mode == 'GVH':
            gaps = self._scan_gaps(self.gap_min_width_m)
            if gaps:
                theta = self._pick_new_heading(gaps)
                if theta is not None:
                    # short hop toward best gap
                    cx = self.heading_reference - theta * (0.5 * self.frame_width)
                    v, w = self.__plan_pid(cx, 0, 0, dt)
                    v = max(v, 0.10)
                    self.search_phase += dt
                    if self.search_phase >= 1.2:     # ~1.2s hop
                        self.mode = 'ISP'
                        self.isp_active = False
                        self.search_phase = 0.0
                    return v, w
            # no good gap → ISP
            self.mode = 'ISP'
            self.isp_active = False
            self.search_phase = 0.0

        if self.mode == 'ISP':
            v, w, done = self._isp_step(frame, dt)
            if done:
                theta = self._isp_pick_heading()
                if theta is not None:
                    self.heading_history.append(theta)
                    # bias toward that heading for a short commit
                    self.mode = 'GVH'
                    self.search_phase = 0.0
                    return 0.12, np.sign(theta)*0.35
                # nothing promising → PTS
                self.mode = 'PTS'
                self.search_phase = 0.0
            return v, w

        if self.mode == 'PTS':
            v, w, done = self._pts_step(dt)
            if done:
                # done with minimal coverage → try DOD again (move on)
                self.mode = 'DOD'
                self.search_phase = 0.0
            return v, w

        # fallback
        return 0.0, 0.0

    def _detect(self, frame):
        if self.detector_mode == 'hsv_circle':
            return find_object_hsv_circle(frame)
        if self.detector_mode == 'hsv_triangle':
            return find_object_hsv_triangle(frame)
        return find_object_hsv(frame)
    
    # def _search_spiral(self, dt):
    #     # exponentially decay |omega| so radius grows (spiral)
    #     omega_mag = self.search_omega0 * (self.search_decay ** self.search_phase)
    #     omega = omega_mag * (1.0 if (int(self.search_phase) % 2 == 0) else -1.0)
    #     v = self.search_v

    #     # advance phase slowly
    #     self.search_phase += dt / max(dt, 1e-3)

    #     # slew-limit for smoothness
    #     v = self.__slew(self.prev_speed, v, self.max_speed_slew)
    #     omega = self.__slew(self.prev_heading, omega, self.max_heading_slew)

    #     self.prev_speed, self.prev_heading = v, omega

    #     return v, omega

    def _search_spiral(self, dt):
        # grow radius over time: decrease |omega| slowly, keep a small forward v
        self.search_phase += max(dt, 1e-3)
        omega_mag = max(0.15, self.search_omega0 * (self.search_decay ** self.search_phase))
        direction = 1.0 if (int(self.search_phase) // 5) % 2 == 0 else -1.0  # flip every ~5s
        omega = direction * omega_mag
        v = self.search_v

        # advance phase slowly
        self.search_phase += dt / max(dt, 1e-3)

        # slew-limit for smoothness
        v = self.__slew(self.prev_speed, v, self.max_speed_slew)
        omega = self.__slew(self.prev_heading, omega, self.max_heading_slew)

        self.prev_speed, self.prev_heading = v, omega

        return v, omega

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
        heading_err = (self.heading_reference - heading_curr) / max(heading_curr, 1e-6)
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

    # def __plan_stanley(self, cx, bw, dt):
    #     # normalized lateral error in [-1,1]
    #     e = (cx - 0.5 * self.frame_width) / max(0.5 * self.frame_width, 1e-6)

    #     # desired apparent width (pixels)
    #     r_ref = self.speed_reference
    #     r_err = (r_ref - bw)

    #     # smooth speed law (auto-brake & gentle reverse)
    #     v = self.v_max * math.tanh(self.k_v * r_err)

    #     # small deadband around target size
    #     if abs(r_err) < self.r_dead:
    #         v = 0.0

    #     # pseudo heading error and Stanley term
    #     theta_e = math.atan2(e, 1.0)
    #     stanley_correction = math.atan(self.k_stanley * e / (abs(v) + self.eps_v))
    #     delta = theta_e + stanley_correction
    #     omega = self.k_omega * delta

    #     # deadband around center
    #     if abs(e) < self.e_dead:
    #         omega = 0.0

    #     # turn-in-place preference
    #     if abs(omega) > self.turn_coeff * self.heading_max:
    #         v = 0.0

    #     v = self.__slew(self.prev_speed, max(-self.v_max, min(self.v_max, v)), self.max_speed_slew)
    #     omega = self.__slew(self.prev_heading, max(-self.heading_max, min(self.heading_max, omega)), self.max_heading_slew)

    #     self.prev_speed, self.prev_heading = v, omega

    #     return v, omega

    def __plan_stanley(self, cx, bw, dt):
        # lateral error: +e means "ball is LEFT of image center" -> turn CCW (positive omega)
        e = (self.heading_reference - cx) / max(0.5 * self.frame_width, 1e-6)

        # range (apparent width) control: positive when ball is far (too small) -> drive forward
        r_ref = self.speed_reference
        r_err = (r_ref - bw)

        # smooth speed (forward when far, reverse gently when too close)
        v = self.v_max * math.tanh(self.k_v * r_err)

        # small deadband around desired width to hold distance
        if abs(r_err) < self.r_dead:
            v = 0.0

        # Stanley steering: proportional heading + velocity-aware term
        stanley_term = math.atan(self.k_stanley * e / (abs(v) + self.eps_v))
        delta = e + stanley_term            # both terms steer TOWARD center
        omega = self.k_omega * delta

        # deadband around image center
        if abs(e) < self.e_dead:
            omega = 0.0

        # prefer turning in place if turning hard
        if abs(omega) > self.turn_coeff * self.heading_max:
            v = 0.0

        v = self.__slew(self.prev_speed, max(-self.v_max, min(self.v_max, v)), self.max_speed_slew)
        omega = self.__slew(self.prev_heading, max(-self.heading_max, min(self.heading_max, omega)), self.max_heading_slew)

        self.prev_speed, self.prev_heading = v, omega

        return v, omega

    def _plan(self, cx, bw, bh, dt):
        if self.controller == 'pid':
            return self.__plan_pid(cx, bw, bh, dt)
        else:
            return self.__plan_stanley(cx, bw, dt)

    def _follow(self, speed, heading):
        cmd_vel = Twist()

        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

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
            self.first_frame = False
            self.get_logger().info('Frame Parameters are set!')

        result = self._detect(frame)
        if result is not None:
            self.get_logger().info('Red Ball Detected!')
            cx, cy, w, h, (x, y, bw, bh) = result
            # Logging pixels from top-left origin
            self.get_logger().info(f'centroid=({cx:.1f},{cy:.1f}) size=({w:.0f},{h:.0f})')

            self.last_seen_t  = now_sec
            self.last_seen_cx = cx
            self.search_phase = 0.0

            self.mode = 'TRACKING'
            self.isp_active = False

            # Draw bbox + centroid
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

            speed, heading = self._plan(cx, bw, bh, dt)
        else:
            self.get_logger().info('No Red Ball Detected!')
            # Lost target
            if self.controller == 'pid':
                if not self.pid_speed.is_reset():
                    self.pid_speed.reset()
                    self.prev_speed = 0.0
                    self.get_logger().info('Speed PID Reset!')
                if not self.pid_heading.is_reset():
                    self.pid_heading.reset()
                    self.prev_heading = 0.0
                    self.get_logger().info('Heading PID Reset!')

            if self.search_mode == 'spiral':
                speed, heading = self._search_spiral(dt)

            elif self.search_mode == 'hybrid':
                if self.mode == 'TRACKING':
                    self._enter_search()
                speed, heading = self._search_step(frame, dt)

            else:  # 'none'
                self.mode = 'TRACKING'
                speed, heading = 0.0, 0.0

        self.get_logger().info(f'Speed: {speed}; Heading: {heading}')
        
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
