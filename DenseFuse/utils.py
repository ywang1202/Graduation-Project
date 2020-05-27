# Utility

import numpy as np

from os import listdir, makedirs, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize
import tensorflow as tf
import h5py
import random
from functools import reduce


def list_images(directory):
    """
    Get the path of all images.
    :param directory: the path of trainset
    :return: For train dataset, output data would be ['.../01.bmp', '.../02.bmp', ...]
    """
    images = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))

    return images


def get_train_images(paths):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, mode='L')
        image = image.reshape((256,256,1))
        images.append(image)
    images = np.stack(images, axis=-1)
    return images


def read_test_image(path, mod_type='L', height=None, width=None):

    image = imread(path, mode=mod_type)
    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')

    if mod_type=='L':
        d = image.shape
        image = np.reshape(image, [d[0], d[1], 1])

    return image


def save_images(paths, datas, save_path, prefix=None):
    if isinstance(paths, str):
        paths = [paths]

    t1 = len(paths)
    t2 = len(datas)
    assert(len(paths) == len(datas))

    if not exists(save_path):
        makedirs(save_path)

    if prefix is None:
        prefix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        # print('data ==>>\n', data)
        if data.shape[2] == 1:
            data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', data)

        name, ext = splitext(path)
        name = name.split(sep)[-1]
        
        path = join(save_path, prefix + ext)
        print('data path==>>', path)


        # new_im = Image.fromarray(data)
        # new_im.show()

        imsave(path, data)


def gradient(input):
    filter = tf.reshape(tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tf.float32), [3, 3, 1, 1])
    d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

    return d
