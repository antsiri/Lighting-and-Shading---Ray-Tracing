import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.animation as animation

from tqdm import tqdm


def normalization(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def ray_tracing(camera_position):
    width = 500
    height = 300

    max_depth = 3

    camera = camera_position
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)

    light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

    objects = [
        { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5}
    ]

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalization(pixel - origin)

            color = np.zeros((3))
            reflection = 1

            for k in range(max_depth):

                # check for intersections
                nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
                if nearest_object is None:
                    break

                # compute intersection point between ray and nearest object
                intersection = origin + min_distance * direction

                normal_to_surface = normalization(intersection - nearest_object['center'])
                shifted_point = intersection + 1e-5 * normal_to_surface
                intersection_to_light = normalization(light['position'] - shifted_point)

                _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
                is_shadowed = min_distance < intersection_to_light_distance
                
                if is_shadowed:
                    break

                # RGB
                illumination = np.zeros((3))

                # ambiant
                illumination += nearest_object['ambient'] * light['ambient']

                # diffuse
                illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

                # specular
                intersection_to_camera = normalization(camera - intersection)
                H = normalization(intersection_to_light + intersection_to_camera)
                illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

                #reflection
                color += reflection * illumination
                reflection *= nearest_object['reflection']

                #new ray origin and direction
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)


            image[i, j] = np.clip(color, 0, 1)

        #print("\tprogress: %d/%d" % (i+1, height))

    return image


def frame_generator():
    camera_position = np.array([0, 0, 1])


    max_frames_num = 50

    frames = []

    for iter in tqdm(range(max_frames_num), position=0):
        
        image = ray_tracing(camera_position)

        frames.append([plt.imshow(image, cmap=cm.Greys_r,animated=True)])

        camera_position = camera_position + np.array([0.01, 0, 0.01])
    
    #print(len(frames))
    #print(camera_position)

    return frames

def video_generator():
    fig = plt.figure()
    plt.axis('off')

    frames = frame_generator()

    video = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

    video.save('video.mp4')

video_generator()