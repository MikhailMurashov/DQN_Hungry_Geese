import numpy as np

import config


def get_epsilon_for_iteration(iter_num):
    if iter_num is 0:
        return config.initial_epsilon
    elif iter_num > config.final_iteration_num:
        return config.final_epsilon
    else:
        # return -0.0168 * iter_num**2 + 3.0867 * iter_num + 41.3745
        return -0.0001 * iter_num + 1.0001


def rgb_to_gray(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


color = {
    0: rgb_to_gray(1, 0, 0),  # red
    1: rgb_to_gray(0, 1, 0),  # green
    2: rgb_to_gray(0, 0, 1),  # blue
    3: rgb_to_gray(1, 1, 0),  # yellow
    4: rgb_to_gray(1, 1, 1)   # white
}


def get_image_from_obs(obs):
    img = np.zeros((7 * 11), dtype=np.float32)

    # whole body position
    for p, pos_list in enumerate(obs['geese']):
        for pos in pos_list:
            img[pos] = color[p]

    # food
    for pos in obs['food']:
        img[pos] = color[4]

    return img.reshape(7, 11)
