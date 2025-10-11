import numpy as np
from math import atan2, sqrt

def yaw_from_quat(q):
    siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return atan2(siny_cosp, cosy_cosp)

def wrap_pi(a):
    a = (a + np.pi) % (2*np.pi) - np.pi
    return a

def dist2d(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
