import numpy as np
import tensorflow as tf

def feature_wt(enc_ir, enc_vi):
    array_ir = enc_ir
    array_vi = enc_vi

    tmp_array_ir = tf.abs(array_ir)
    tmp_array_vi = tf.abs(array_vi)

    l1_ir = tf.reduce_sum(tmp_array_ir, 3, keepdims=True)
    l1_vi = tf.reduce_sum(tmp_array_vi, 3, keepdims=True)

    l1_ir_padded = tf.pad(l1_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    l1_vi_padded = tf.pad(l1_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    kernel = tf.ones([3, 3, 1, 1])
    activity_level_map_ir = tf.nn.conv2d(l1_ir_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    activity_level_map_vi = tf.nn.conv2d(l1_vi_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')

    activity_level_map_ir = activity_level_map_ir / (2 * 1 + 1) ** 2
    activity_level_map_vi = activity_level_map_vi / (2 * 1 + 1) ** 2

    wt_ir = activity_level_map_ir / (activity_level_map_ir + activity_level_map_vi)
    wt_vi = activity_level_map_vi / (activity_level_map_ir + activity_level_map_vi)

    fuse_map = enc_ir * wt_ir + enc_vi * wt_vi

    return fuse_map