import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# settings for producing images
IMG_SIZE = 7     # image size is 7x7
N_COLORS = 5
N_SHAPES = 5
N_TASKS = 2      # color and shape
N_TRIALS_TRAIN = 10000  # generate 10,000 training samples
N_TRIALS_TEST = 2000    # generate 2,000 testing samples

def make_shape_masks(size=IMG_SIZE):
    masks = {}
    y,x = np.mgrid[0:size, 0:size]
    mid = size // 2                  # 0 1 2 3 4 5 6 so for size=7, mid=3
    # circle
    r = size / 2 - 0.5               # r=3
    masks['circle'] = ((x - mid) ** 2 + (y - mid) ** 2 <= r ** 2).astype(np.float32)
    # square
    masks['square'] = np.ones((size, size), dtype=np.float32)
    # triangle
    tri = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        width = i + 1
        start = (size - width) // 2
        tri[i, start:start + width] = 1
    masks['triangle'] = tri
    # cross
    cross = np.zeros((size, size), dtype=np.float32)
    cross[mid, :] = 1
    cross[:, mid] = 1
    masks['cross'] = cross
    # diamond
    diamond = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if abs(i - mid) + abs(j - mid) <= mid:
                diamond[i, j] = 1
    masks['diamond'] = diamond
    return masks

SHAPE_NAMES = ['circle', 'square', 'triangle', 'cross', 'diamond']
COLOR_NAMES = ['red', 'green', 'blue', 'yellow', 'purple']
COLOR_RGB = {
    'red':    [1.0, 0.0, 0.0],
    'green':  [0.0, 1.0, 0.0],
    'blue':   [0.0, 0.0, 1.0],
    'yellow': [1.0, 1.0, 0.0],
    'purple': [1.0, 0.0, 1.0],
}
MASKS = make_shape_masks()

def make_image(shape_name, color_name):
    mask = MASKS[shape_name]
    rgb = COLOR_RGB[color_name]
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = mask * rgb[c]
    return img

def generate_dataset(n_trials):
    X_img = np.zeros((n_trials, IMG_SIZE * IMG_SIZE * 3), dtype=np.float32)
    X_task = np.zeros((n_trials, 2), dtype=np.float32)
    Y = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        c_idx = np.random.randint(N_COLORS)
        s_idx = np.random.randint(N_SHAPES)
        t_idx = np.random.randint(N_TASKS)
        img = make_image(SHAPE_NAMES[s_idx], COLOR_NAMES[c_idx])
        X_img[i] = img.flatten()
        X_task[i, t_idx] = 1.0
        if t_idx == 0:
            Y[i] = c_idx
        else:
            Y[i] = s_idx
    return X_img, X_task, Y


# so we get dataset like this:
# X_img:  (N, 7*7*3) = (N, 147)
# X_task: (N, 2)
# Y:      (N,)
