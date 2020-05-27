# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from datetime import datetime
from densefuse_net import DenseFuseNet
from utils import save_images, read_test_image


def generate(infrared_path, visible_path, model_path, model_pre_path, index, type='addition', output_path=None):
    if type == 'addition':
        print('addition')
        _handler(infrared_path, visible_path, model_path, model_pre_path, index, output_path=output_path)
    # elif type == 'CBF':
    #     print('CBF')
    #     _handler_cbf(infrared_path, visible_path, model_path, model_pre_path, index, output_path=output_path)
    # elif type == 'l1':
    #     _handler_l1(infrared_path, visible_path, model_path, model_pre_path, index, output_path=output_path)
    elif type == 'wt':
        _handler_wt(infrared_path, visible_path, model_path, model_pre_path, index, output_path=output_path)


def _handler_wt(ir_path, vi_path, model_path, model_pre_path, index, output_path=None):
    ir_img = read_test_image(ir_path)
    vi_img = read_test_image(vi_path)
    dimension = ir_img.shape

    ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
    vi_img = vi_img.reshape([1, dimension[0], dimension[1], dimension[2]])

    print('img shape final:', ir_img.shape)

    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
        infrared_field = tf.placeholder(tf.float32, shape=ir_img.shape, name='image_ir')
        visible_field = tf.placeholder(tf.float32, shape=ir_img.shape, name='image_vi')

        dfn = DenseFuseNet(model_pre_path)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        output_image = dfn.transform_wt(infrared_field, visible_field)

        output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vi_img})

        save_images(ir_path, output, output_path, prefix='Dense_fuse_' + str(index).zfill(2))


def _handler(ir_path, vi_path, model_path, model_pre_path, index, output_path=None):
    ir_img = read_test_image(ir_path)
    vi_img = read_test_image(vi_path)
    dimension = ir_img.shape

    ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
    vi_img = vi_img.reshape([1, dimension[0], dimension[1], dimension[2]])

    print('img shape final:', ir_img.shape)

    with tf.Graph().as_default(), tf.Session() as sess:
        infrared_field = tf.placeholder(
            tf.float32, shape=ir_img.shape, name='content')
        visible_field = tf.placeholder(
            tf.float32, shape=ir_img.shape, name='style')

        dfn = DenseFuseNet(model_pre_path)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        output_image = dfn.transform_addition(infrared_field, visible_field)
        # output_image = dfn.transform_cbf(infrared_field, visible_field)

        output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vi_img})

        save_images(ir_path, output, output_path, prefix='Dense_fuse_' + str(index).zfill(2))


# def _handler_cbf(ir_path, vi_path, model_path, model_pre_path, index, output_path=None):
#     ir_img = read_test_image(ir_path)
#     vi_img = read_test_image(vi_path)
#     dimension = ir_img.shape
#
#     ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
#     vi_img = vi_img.reshape([1, dimension[0], dimension[1], dimension[2]])
#
#     print('img shape final:', ir_img.shape)
#
#     with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
#         infrared_field = tf.placeholder(tf.float32, shape=ir_img.shape, name='image_ir')
#         visible_field = tf.placeholder(tf.float32, shape=ir_img.shape, name='image_vi')
#
#         dfn = DenseFuseNet(model_pre_path)
#
#         # restore the trained model and run the style transferring
#         # saver = tf.train.Saver()
#         # saver.restore(sess, model_path)
#
#         output_image = dfn.transform_cbf(infrared_field, visible_field)
#
#         output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vi_img})
#
#         save_images(ir_path, output, output_path, prefix='Dense_fuse_' + str(index).zfill(2))