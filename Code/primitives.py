import numba as nb
from numba import literal_unroll
import numpy as np

from ray import Ray
from mesh import *

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
        
@nb.njit()
def get_node_children(node_index):
    left_index = 2*node_index+1
    right_index = 2*node_index+2
    return left_index, right_index

@nb.njit()
def intersect_tree(tree, ray):
    stack = [(0, 0)]    #start with the root and depth 0
    nearest = -1

    while stack:
        idx, depth = stack.pop()

        if idx>=len(tree) or np.all(tree[idx]==np.zeros(3)-1):
            continue

        axis = 0

        next_idx = None
        opposite_idx = None

        left_idx, right_idx = get_node_children(idx)

        if ray.origin[axis] < tree[idx][axis]:
            next_idx = left_idx
            opposite_idx = right_idx
        else:
            next_idx = right_idx
            opposite_idx = right_idx

        if nearest == -1 or np.linalg.norm(tree[nearest] - ray.origin) > np.linalg.norm(tree[idx] - ray.origin):
            nearest = idx

        stack.append((next_idx, depth+1))
    
    return nearest

@nb.njit()
def _intersect(tree, traingles, ray):
    traingle_index = intersect_tree(tree, ray)

    if traingle_index == -1:
        return [np.inf], [np.array([0, 0, 0], dtype=np.float64)]
    
    traingle = traingles[traingle_index]

    A,B,C = traingle
    edgeAB = B-A
    edgeAC = C-A
    normal_vector = np.cross(edgeAB, edgeAC)
    ao = ray.origin-A

    determinant = -np.dot(ray.direction, normal_vector)
    invDet = 1/determinant

    t = np.dot(ao, normal_vector)*invDet

    if t<ray.tmin or t>ray.tmax:
        pass

    return [t], [normal_vector]

@nb.njit
def closest_intersection(ray, objects):
    closest_t = np.inf
    closest_index = -1
    closest_normal = np.array([0,0,0], dtype=np.float64)

    i=-1

    object = None 

    for obj in literal_unroll(objects):
        i += 1
        ts, normals = obj.intersect(ray)
        for t, normal in zip(ts, normals):
            if ray.tmin <= t <= ray.tmax:
                if t < closest_t:
                    closest_t = t
                    closest_index = i
                    closest_normal = normal
                    object = obj
    
    return object, closest_t, closest_index, closest_normal

if __name__ == '__main__':
    ss = (Sphere(np.array([0,0,0], np.float64), 1),
          Sphere(np.array([0,0,1], np.float64), 1))
    
    r = Ray(np.array([0,0,-2], np.float64), np.array([0,0,1], np.float64))

    closest_object, closest_t, closest_index, closest_normal = closest_intersection(r, ss)

    P = r.P(closest_t)
    print(closest_object.normal(P))
    print(closest_normal)

    from load_object import load_obj
    obj = load_obj("objects/teapot.obj")
    triangles = np.array(obj.triangles, dtype=np.float64)
    
    r = Ray(np.array([0,0,-2], np.float64), np.array([0,0,1], np.float64))
    ts, normals = obj.intersect(r)

    print(normals)
