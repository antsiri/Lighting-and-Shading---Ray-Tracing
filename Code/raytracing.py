from PIL import Image
import numpy as np

w=400
h=300

def normalize(x):
    x /= np.linalg.norm(x)
    return x

