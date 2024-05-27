import numba as nb
from primitives import _intersect

@nb.experimental.jitclass([
    ("tree", nb.float64[:,:]),
    ("min_ax", nb.float64[:]),
    ("max_ax", nb.float64[:]),
    ("triangles", nb.float64[:,:,:]),
])
class Mesh():
    def __init__(self, tree, min_ax, max_ax, triangles):
        self.tree = tree
        self.min_ax = min_ax
        self.max_ax = max_ax
        self.triangles = triangles

    def get_node_parent(self, node_index):
        return int((node_index-1)/2)
    
    def intersect(self, ray):
        return _intersect(self.tree, self.triangles, ray)
    