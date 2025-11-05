class PID:
    """ PID """
    def __init__(self, kp, ki, kd, i_limit=1.0, out_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0
        self.prev_e = 0.0
        self.first = True
        self.i_limit = i_limit
        self.out_limit = out_limit  # (min,max) or None

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def step(self, e, dt):
        if dt <= 0.0:
            return 0.0
        # integrate with clamp
        self.i += e * dt
        self.i = max(-self.i_limit, min(self.i_limit, self.i))
        # derivative
        d = 0.0 if self.first else (e - self.prev_e) / dt
        self.first = False
        self.prev_e = e
        u = self.kp*e + self.ki*self.i + self.kd*d
        if self.out_limit:
            u = max(self.out_limit[0], min(self.out_limit[1], u))
        return u
