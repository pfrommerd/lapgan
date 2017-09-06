import utils

import os

import numpy as np

import tensorflow as tf

PARAMS_L1 = {'layer_num': 0,
          'output_dir': './output/cifar_ae_l1',
          'coarse_shape': (8, 8, 3),
          'fine_shape': (16, 16, 3),
          'num_planes': 128,
          'latent_dim': 1024,
 
          'data_dir': './data/cifar',
          'num_classes': 10,
          'noise': 1, # std deviations of noise
          'initial_epoch': 0,
          'epochs': 300,
          'steps_per_epoch': 391, #~50000 images (782 * 64)
          'batch_size': 128,
          'test_batch_size': 128,
          'sample_img_num': 16,
          'preprocess_threads': 2,
          'use_voronoi': False}

PARAMS_L2 = {'layer_num': 1,
          'output_dir': './output/cifar_ae_l2',
          'coarse_shape': (16, 16, 3),
          'fine_shape': (32, 32, 3),
          'num_planes': 128,
          'latent_dim': 512,

          'data_dir': './data/cifar',
          'num_classes': 10,
          'noise': 1, # std deviations of noise
          'initial_epoch': 0,
          'epochs': 300,
          'steps_per_epoch': 391, #~50000 images (782 * 64)
          'batch_size': 128,
          'test_batch_size': 128,
          'sample_img_num': 16,
          'preprocess_threads': 4,
          'use_voronoi': False}
    
def get_params(layer_num):
    return PARAMS_L1 if layer_num == 0 else PARAMS_L2

def build_model(params, inputs):
    # Inputs
    base_img = inputs['base_img']
    diff_real = inputs['diff_real']
    class_cond = inputs['class_cond']
    keep_prob = inputs['keep_prob']
    noise = inputs['noise']

    with tf.variable_scope('autoencoder'):
        gen_output = _build_autoencoder(inputs, params['fine_shape'], params['num_planes'], params['latent_dim'])

    diff_gen = gen_output['diff_img']

    gen_loss = tf.reduce_sum(tf.square(diff_gen - diff_real))
    loss = gen_loss

    fake_diff_summary = utils.diff_summary('fake_diff', gen_output['diff_img'])
    real_diff_summary = utils.diff_summary('real_diff', diff_real)

    fake_img_summary = tf.summary.image('fake_img', base_img + gen_output['diff_img'])
    real_img_summary = tf.summary.image('real_img', base_img + diff_real)

    base_img_summary = tf.summary.image('input_img', base_img)

    gen_loss_test_summary = tf.summary.scalar('gen_test_loss', gen_loss)
    loss_test_summary = tf.summary.scalar('test_loss', loss)

    test_summaries_op = tf.summary.merge_all()

    gen_loss_summary = tf.summary.scalar('gen_loss', gen_loss)
    loss_summary = tf.summary.scalar('loss', loss)

    train_summaries_op = tf.summary.merge([gen_loss_summary, loss_summary])

    gen_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder')
    gen_opt = tf.train.AdamOptimizer(1e-4).minimize(gen_loss, var_list=gen_weights)

    def train(iteration, sess, writer):
        _, train_summary = sess.run([gen_opt, train_summaries_op]) 
        writer.add_summary(train_summary, iteration)

    def test(iteration, sess, writer):
        test_summary = sess.run(test_summaries_op) 
        writer.add_summary(test_summary, iteration)
        writer.flush()

    return diff_gen, train, test

def _build_autoencoder(inputs, data_shape, num_planes, latent_dim):
    base_img = inputs['base_img']
    class_cond = inputs['class_cond']

    # We don't actually need a bias here as we are learning a bitplane per class anyways
    class_weights = tf.get_variable('class_weights', [10, data_shape[0] * data_shape[1]])
    class_vec = utils.dense(class_weights)(class_cond)
    class_plane = tf.nn.relu(tf.reshape(class_vec, [-1, data_shape[0], data_shape[1], 1]))

    # Now concatenate the tensors
    stacked_input = tf.concat([base_img, class_plane], axis=3)

    c1_weights = tf.get_variable('d1_weights', [7, 7, data_shape[2] + 1, num_planes])
    c1_bias = tf.get_variable('d1_bias', [num_planes])
    c1 = utils.conv2d(c1_weights, bias=c1_bias)(stacked_input)

    c2_weights = tf.get_variable('d2_weights', [7, 7, num_planes, num_planes])
    c2_bias = tf.get_variable('d2_bias', [num_planes])
    c2 = utils.conv2d(c2_weights, bias=c2_bias)(c1)

    c2_dropout = tf.nn.dropout(c2, inputs['keep_prob'])

    c3_weights = tf.get_variable('d3_weights', [7, 7, num_planes, num_planes])
    c3_bias = tf.get_variable('d3_bias', [num_planes])
    c3 = utils.conv2d(c3_weights, bias=c3_bias)(c2_dropout)

    c4_weights = tf.get_variable('d4_weights', [7, 7, num_planes, num_planes])
    c4_bias = tf.get_variable('d4_bias', [num_planes])
    c4 = utils.conv2d(c4_weights, bias=c4_bias)(c3)

    c4_dropout = tf.nn.dropout(c4, inputs['keep_prob'])

    c5_weights = tf.get_variable('d5_weights', [7, 7, num_planes, 3])
    c5_bias = tf.get_variable('d5_bias', [3])
    c5 = utils.conv2d(c5_weights, bias=c5_bias)(c4_dropout)

    diff_img = c5

    return {'diff_img': diff_img}
