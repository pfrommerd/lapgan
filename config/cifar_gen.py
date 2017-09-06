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
    class_cond = inputs['class_cond']
    base_img = inputs['base_img']
    keep_prob = inputs['keep_prob']

    with tf.variable_scope('gen') as scope:
        gen_diff = _build_generator(inputs, params['fine_shape'], params['num_planes'])['diff_img']

        gen_image_summary = tf.summary.image('result', base_img + gen_diff)
        gen_diff_summary = utils.diff_summary('diff', gen_diff)

        test_summaries_op = tf.summary.merge([gen_image_summary, gen_diff_summary])

        gen_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')

        def train(iteration, sess, writer):
            pass

        def test(iteration, sess, writer):
            test_summary = sess.run(test_summaries_op) 
            writer.add_summary(test_summary, iteration)
            writer.flush()

        return gen_diff, gen_weights, test

def _build_generator(inputs, data_shape, nplanes):
    class_cond = inputs['class_cond']
    base_img = inputs['base_img']
    noise = inputs['noise']

    class_weights = tf.get_variable('class_weights', [10, data_shape[0] * data_shape[1]])

    # We don't actually need a bias here as we are learning a bitplane per class anyways
    class_vec = utils.dense(class_weights)(class_cond)
    class_plane = tf.nn.relu(tf.reshape(class_vec, [-1, data_shape[0], data_shape[1], 1]))

    # Reshape the noise
    noise_plane = tf.reshape(noise, [-1, data_shape[0], data_shape[1], 1])

    # Now concatenate the tensors
    stacked_input = tf.concat([base_img, noise_plane, class_plane], axis=3)

    g1_weights = tf.get_variable('g1_weights', [7, 7, data_shape[2] + 2, nplanes])
    g1_bias = tf.get_variable('g1_bias', [nplanes])
    g1 = tf.nn.relu(utils.conv2d(g1_weights, bias=g1_bias)(stacked_input))

    g2_weights = tf.get_variable('g2_weights', [7, 7, nplanes, nplanes])
    g2_bias = tf.get_variable('g2_bias', [nplanes])
    g2 = tf.nn.relu(utils.conv2d(g2_weights, bias=g2_bias)(g1))

    g3_weights = tf.get_variable('g3_weights', [5, 5, nplanes, data_shape[2]])
    g3_bias = tf.get_variable('g3_bias', data_shape[2])
    g3 = tf.nn.tanh(utils.conv2d(g3_weights, bias=g3_bias, name='diff_img')(g2))

    return {'diff_img': g3}
