import numpy as np

from mesh import *

def number_of_nodes_in_tree(kd_tree):
    if kd_tree is None:
        return num_nodes
    
    num_nodes = 1

    if kd_tree.left is not None:
        num_nodes += number_of_nodes_in_tree(kd_tree.left)
    if kd_tree.right is not None:
        num_nodes += number_of_nodes_in_tree(kd_tree.right)
    
    return num_nodes

def convert_tree_to_numba(kd_tree):
    n_nodes = number_of_nodes_in_tree(kd_tree)*2
    tree = np.zeros(shape=(n_nodes, 3))-1
    min_ax = np.zeros(n_nodes)-1
    max_ax = np.zeros(n_nodes)-1
    triangles = np.zeros(shape=(n_nodes, 3, 3))-1

    def convert_node_to_array(node, index):
        tree[index] = node.center
        min_ax[index] = node.min_ax
        max_ax[index] = node.max_ax
        triangles[index] = node.triangle

        if node.left is not None:
            convert_node_to_array(node.left, 2*index+1)
        if node.right is not None:
            convert_node_to_array(node.right, 2*index+2)

    convert_node_to_array(kd_tree, 0)

    return tree, min_ax, max_ax, triangles

class Node():
    def __init__(self, triangle, center, min_ax, max_ax, axis, left=None, right=None):
        self.triangle = triangle
        self.center = center
        self.min_ax = min_ax
        self.max_ax = max_ax
        self.axis = axis
        self.left = left
        self.right = right

def get_triangle_center(triangle):
    return (triangle[0] + triangle[1] + triangle[2]) / 3

def construct_kd_tree(triangles, axis=0):
    if len(triangles) == 0:
        return None
    
    vals = list(sorted(triangles, key=lambda x: get_triangle_center(x)[axis]))

    median = len(triangles) // 2

    left = construct_kd_tree(vals[:median])
    right = construct_kd_tree(vals[median+1:])

    return Node(vals[median],
                get_triangle_center(vals[median]),
                np.min(vals[0][:,axis]),
                np.max(vals[-1][:,axis]),
                axis,
                left=left,
                right=right)

def get_triangles(vertices, faces):
    triangles = []
    for face in faces:
        triangles.append([vertices[face[0]-1], vertices[face[1]-1], vertices[face[2]-1]])
    return triangles

def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith("f "):
                line_data = line.split(" ")
                faces.append([int(x) for x in line_data[1:]])
            if line.startswith("v "):
                line_data = line.split(" ")
                vertices.append([float(x) for x in line_data[1:]])

    triangles = get_triangles(vertices, faces)
    triangles = np.array(triangles)

    kd_tree = construct_kd_tree(triangles)

    tree, min_ax, max_ax, triangles = convert_tree_to_numba(kd_tree)

    return Mesh(tree, min_ax, max_ax, triangles)

def intersect_tree(node, ray, depth=0):
    if node is None:
        return None
    
    axis = 0
    next_branch = None
    opposite_branch = None

    if ray.origin[axis] < node.center[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left

    nearest = intersect_tree(next_branch, ray, depth+1)

    if nearest is None or np.linalg.norm(nearest.center - ray.origin) > np.linalg.norm(node.center - ray.origin):
        nearest = node

    return nearest


if __name__ == "__main__":
    object = load_obj("objects/teapot.obj")

    from ray import Ray
    ray = Ray(np.array([0,0,0], dtype=np.float64), np.array([1,1,1], dtype=np.float64))

    print(object.intersect(ray))
    