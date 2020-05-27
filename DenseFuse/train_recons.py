# -*- coding: utf-8 -*-
# Train the DenseFuse Net

from __future__ import print_function

import os
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from densefuse_net import DenseFuseNet
from utils import get_train_images


HEIGHT = 256
WIDTH = 256
CHANNELS = 1 # gray scale, default

LEARNING_RATE = 1e-4
EPSILON = 1e-5


def train_recons(original_imgs_path, save_path, model_pre_path, EPOCHES_set, BATCH_SIZE_set, debug=False, logging_period=1):
    from datetime import datetime
    if debug:
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    BATCH_SIZE = BATCH_SIZE_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)

    num_imgs = len(original_imgs_path)
    mod = num_imgs % BATCH_SIZE

    print('Train images number {}.'.format(num_imgs))
    print('Train images samples {}.'.format(num_imgs // BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed {} samples...'.format(mod))
        original_imgs_path = original_imgs_path[:-mod]

    # get the traing image shape
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
        with tf.compat.v1.name_scope('Input'):
            original = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='original')
            source = original

        print('source :', source.shape)
        print('original :', original.shape)

        # create the deepfuse net (encoder and decoder)
        dfn = DenseFuseNet(model_pre_path)
        generated_img = dfn.transform_recons(source)
        print('generate:', generated_img.shape)

        epsilon_1 = tf.reduce_mean(tf.square(generated_img - original))
        epsilon_2 = 1 - tf.reduce_mean(tf.image.ssim(generated_img, original, max_val=1.0))
        total_loss = epsilon_1 + 1000 * epsilon_2

        tf.compat.v1.summary.scalar('epsilon_1', epsilon_1)
        tf.compat.v1.summary.scalar('epsilon_2', epsilon_2)
        tf.compat.v1.summary.scalar('total_loss', total_loss)

        train_op = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)

        summary_op = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter('./models/log', sess.graph, flush_secs=60)
        train_writer.add_graph(sess.graph)

        sess.run(tf.compat.v1.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.compat.v1.train.Saver(max_to_keep=20)

        # ** Start Training **
        step = 0
        n_batches = int(len(original_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Elapsed time for preprocessing before actually train the model: {}'.format(elapsed_time))
            print('Now begin to train the model...')
            start_time = datetime.now()

        Loss_1 = []
        Loss_2 = []
        Loss_all = []
        for epoch in range(EPOCHS):
            for batch in range(n_batches):
                # retrive a batch of infrared and visiable images
                original_path = original_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                original_batch = get_train_images(original_path)
                # print(original_batch.shape)
                original_batch = original_batch.transpose((3, 0, 1, 2))
                # run the training step
                step += 1
                _, summary_str, _epsilon_1, _epsilon_2, _total_loss = sess.run([train_op, summary_op, epsilon_1, epsilon_2, total_loss], feed_dict={original: original_batch})

                train_writer.add_summary(summary_str, step)
                Loss_1.append(_epsilon_1)
                Loss_2.append(_epsilon_2)
                Loss_all.append(_total_loss)

                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        print('epoch:{:>2}/{}, step:{:>4}, total loss: {:.4f}, elapsed time: {}'.format(epoch + 1, EPOCHS, step, _total_loss, elapsed_time))
                        print('epsilon_1: {}, epsilon_2: {}\n'.format(_epsilon_1, _epsilon_2))

            # ** Done Training & Save the model **
            saver.save(sess, save_path, global_step=epoch+1)
        
            if not os.path.exists('./models/loss/'):
                os.mkdir('./models/loss/')

            scio.savemat('./models/loss/TotalLoss_'+str(epoch+1)+'.mat', {'total_loss': Loss_all})
            scio.savemat('./models/loss/Epsilon1_'+str(epoch+1)+'.mat', {'epsilon_1': Loss_1})
            scio.savemat('./models/loss/Epsilon2_'+str(epoch+1)+'.mat', {'epsilon_2': Loss_2})

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: {}'.format(elapsed_time))
            print('Model is saved to: {}'.format(save_path))
