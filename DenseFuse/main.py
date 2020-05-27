# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function
# import sys
# sys.path.append('/home/aistudio/external-libraries')
import time

from train_recons import train_recons
from generate import generate
from utils import list_images
import os


# True for training phase
IS_TRAINING = False
# True for RGB images
is_RGB = False

BATCH_SIZE = 4
EPOCHES = 10
nModel = 1
MODEL_SAVE_PATH = './models/densefuse/densefuse_model.ckpt' + '-' + str(nModel)

# In testing process, 'model_pre_path' is set to None
# The "model_pre_path" in "main.py" is just a pre-train model and not necessary for training and testing.
# It is set as None when you want to train your own model.
# If you already train a model, you can set it as your model for initialize weights.
model_pre_path = None


def main():

    if IS_TRAINING:

        original_imgs_path = list_images('./trainset/')

        print('Begin to train the network ...')
        train_recons(original_imgs_path, MODEL_SAVE_PATH, model_pre_path,
                     EPOCHES, BATCH_SIZE, debug=True, logging_period=10)
        print('Successfully! Done training...')
    else:
        model_path = MODEL_SAVE_PATH
        print('Begin to generate pictures ...')

        from os import listdir
        img_path = './IV_images/'
        images = sorted(listdir(img_path))
        # ir_images = images[::2]
        # vi_images = images[1::2]
        ir_images = []
        vi_images = []
        for img in images:
            if 'IR' in img:
                ir_images.append(img)
            else:
                vi_images.append(img)

        for ir, vi in zip(ir_images, vi_images):
            index = ir[2:4]

            infrared = img_path + ir
            visible = img_path + vi

            # choose fusion layer
            # fusion_type = 'CBF'
            fusion_type = 'wt'
            # fusion_type = 'addition'
            output_save_path = os.path.join('./outputs', fusion_type, str(nModel))

            generate(infrared, visible, model_path, model_pre_path,
                     index, type = fusion_type, output_path = output_save_path)


if __name__ == '__main__':
    main()
