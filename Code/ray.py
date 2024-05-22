import numba as nb
import numpy as np

@nb.experimental.jitclass([
    ("origin", nb.float64[:]),
    ("direction", nb.float64[:]),
    ("tmin", nb.float64),
    ("tmax", nb.float54),
])

class Ray():
    def __init__(self, origin, direction, tmin=0.001, tmax=np.inf):
        self.origin = origin
        self.direction = direction
        self.tmin = tmin
        self.tmax = tmax

    def P(self, t):
        return self.origin + t*self.direction 