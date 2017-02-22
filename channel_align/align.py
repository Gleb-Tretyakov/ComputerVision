from skimage import io, data
import numpy as np
import math
from numpy import array, dstack, roll
from skimage.transform import rescale, resize
from skimage.util import crop

def get_val(A, B, dx, dy):
    A = A[:A.shape[0] - abs(dx), :A.shape[1] - abs(dy)]
    B = B[:B.shape[0] - abs(dx), :B.shape[1] - abs(dy)]
    return np.sum((A - B) ** 2) * 1.0 / ((A.shape[0] - abs(dx)) * (A.shape[1] - abs(dy)))

def get_shift(A, B, val):
    arr_shift = np.zeros((2))
    min_val = get_val(A, B, 0, 0)
    for dx in range(-val, val + 1):
        for dy in range(-val, val + 1):
            B1 = roll(B, dx, axis = 0)
            B1 = roll(B1, dy, axis = 1)
            now = get_val(A, B1, dx, dy)
            if (now < min_val):
                min_val = now
                arr_shift = np.array([dx, dy])
    return arr_shift

def level_pyramid(A, B, level):
    if (level == 0):
        res = get_shift(A, B, 4)
        return res
    else:
        A1 = rescale(A, 0.5)
        B1 = rescale(B, 0.5)
        res = level_pyramid(A1, B1, level - 1) * 2
        res = res.astype(int)
        B = roll(B, res[0], axis = 0)
        B = roll(B, res[1], axis = 1)
        res += get_shift(A, B, 2).astype(int)
        return res

def align(image):
    img = np.array(image)
    imgs = np.array_split(img, 3)
    for index in range(3):
        imgs[index] = crop(imgs[index], (0.05 * len(img[1])))
    h_min = min(imgs[0].shape[0], imgs[1].shape[0], imgs[2].shape[0])
    for index in range(3):
        while (imgs[index].shape[0] != h_min):
            imgs[index] = np.delete(imgs[index], 0, 0)
    h, w = imgs[0].shape[0], imgs[0].shape[1]
    dep = 1
    while (h >= 200 or w >= 200):
        h /= 2
        w /= 2
        dep += 1
    shift1 = level_pyramid(imgs[2], imgs[0], dep)
    imgs[0] = roll(imgs[0], shift1[0], axis = 0)
    imgs[0] = roll(imgs[0], shift1[1], axis = 1)
    shift2 = level_pyramid(imgs[2], imgs[1], 5)
    imgs[1] = roll(imgs[1], shift2[0], axis = 0)
    imgs[1] = roll(imgs[1], shift2[1], axis = 1)
    return dstack((imgs[2], imgs[1], imgs[0]))
