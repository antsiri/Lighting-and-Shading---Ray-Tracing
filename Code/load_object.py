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

def convert_node_to_array(tree, min_ax, max_ax, triangles, node, index):
    tree[index] = node.center
    min_ax[index] = node.min_ax
    max_ax[index] = node.max_ax
    triangles[index] = node.triangle

    if node.left is not None:
        convert_node_to_array(node.left, 2*index+1)
    if node.right is not None:
        convert_node_to_array(node.right, 2*index+2)

def convert_tree_to_numba(kd_tree):
    n_nodes = number_of_nodes_in_tree(kd_tree)*2
    tree = np.zeros(shape=(n_nodes, 3))-1
    min_ax = np.zeros(n_nodes)-1
    max_ax = np.zeros(n_nodes)-1
    triangles = np.zeros(shape=(n_nodes, 3, 3))-1

    convert_node_to_array(tree, min_ax, max_ax, triangles, kd_tree, 0)

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

#def get_triangles


