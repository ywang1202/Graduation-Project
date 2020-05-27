import numpy as np
import random
from os import listdir, path, mkdir
from scipy.misc import imread, imsave, imresize

original_path = "./train2017"

if not path.exists("./trainset"):
    mkdir("./trainset")

img_list = listdir(original_path)
random.shuffle(img_list)

for img_path in img_list:
    image_path = path.join(original_path, img_path)
    image = imread(image_path, mode='L')
    image = imresize(image, [256, 256], interp='bilinear')

    save_path = path.join("./trainset", img_path)
    imsave(save_path, image)
