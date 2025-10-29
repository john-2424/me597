class PID:
    """ Normal PID """
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

class PIDStar:
    """ PID with two modes: Optimum and Fast """
    def __init__(self, skp, ski, skd, fkp, fki, fkd, si_limit=1.0, sout_limit=None, fi_limit=1.0, fout_limit=None):
        self.skp, self.ski, self.skd, self.fkp, self.fki, self.fkd  = skp, ski, skd, fkp, fki, fkd
        self.si_limit, self.fi_limit = si_limit, fi_limit
        self.sout_limit, self.fout_limit = sout_limit, fout_limit  # (min,max) or None

        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def step(self, e, dt, mode='opt'):
        if mode == 'fast':
            kp, ki, kd, i_limit, out_limit = self.fkp, self.fki, self.fkd, self.fi_limit, self.fout_limit
        else:
            kp, ki, kd, i_limit, out_limit = self.skp, self.ski, self.skd, self.si_limit, self.sout_limit
        
        if dt <= 0.0:
            return 0.0
        
        # integrate with clamp
        self.i += e * dt
        self.i = max(-i_limit, min(i_limit, self.i))
        # derivative
        d = 0.0 if self.first else (e - self.prev_e) / dt
        self.first = False
        self.prev_e = e

        val = kp*e + ki*self.i + kd*d
        
        if out_limit:
            val = max(out_limit[0], min(out_limit[1], val))
        
        return val
