import numba as nb
from numba import literal_unroll
import numpy as np

from ray import Ray

@nb.experimental.jitclass([
    ("position", nb.float64[:]),
    ("radius", nb.float64),
])
class Sphere():
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def normal(self, P):
        return (P - self.position)/self.radius
    
    def intersect(self, ray):
        a = np.dot(ray.direction, ray.direction)
        b = 2*np.dot(ray.origin-self.position, ray.direction)
        c = np.dot(ray.origin-self.position, ray.origin-self.position) - self.radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return [np.inf], [np.array([0, 0, 0], dtype=np.float64)]
        else:
            t1 = (-b - np.sqrt(discriminant))/(2*a)
            t2 = (-b + np.sqrt(discriminant))/(2*a)
            return [t1, t2], [self.normal(ray.P(t)) for t in [t1, t2]]