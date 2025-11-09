import math


def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x

def wrap_pi(ang: float) -> float:
    return math.atan2(math.sin(ang), math.cos(ang))
