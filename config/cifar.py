import dataio
import utils

import os

import numpy as np

import tensorflow as tf

PARAMS_L1 = {'layer_num': 0,
          'output_dir': './output/cifar_l1',
          'coarse_shape': (8, 8, 3),
          'fine_shape': (16, 16, 3),
          'num_planes': 64,
 
          'data_dir': './data/cifar',
          'num_classes': 10,
          'noise': 1, # std deviations of noise
          'initial_epoch': 0,
          'epochs': 300,
          'steps_per_epoch': 391, #~50000 images (782 * 64)
          'validation_steps': 20,
          'batch_size': 128,
          'test_batch_size': 128,
          'sample_img_num': 16,
          'use_voronoi': False}

PARAMS_L2 = {'layer_num': 1,
          'output_dir': './output/cifar_l2',
          'coarse_shape': (16, 16, 3),
          'fine_shape': (32, 32, 3),
          'num_planes': 128,

          'data_dir': './data/cifar',
          'num_classes': 10,
          'noise': 1, # std deviations of noise
          'initial_epoch': 0,
          'epochs': 300,
          'steps_per_epoch': 391, #~50000 images (782 * 64)
          'validation_steps': 20,
          'batch_size': 128,
          'test_batch_size': 128,
          'sample_img_num': 16,
          'use_voronoi': False}
    
def get_params(layer_num):
    return PARAMS_L1 if layer_num == 0 else PARAMS_L2

# Will be called by main script
from dataloaders.cifar import build_data_pipeline

def format_batches(params, data):
    # Pretty much here we just format the data
    # as it needs to be fed into the model
    # ([gen_fake_cond, gen_fake_class, disc_fake_class, disc_real_input, disc_real_class],
    #       [gen_fake_target..., gen_real_targets..., disc_fake_targets..., disc_real_targets...])
    def input_formatter(pyramid):
        for batch in pyramid:
            class_input = batch[3]
            cond_image_input = batch[params['layer_num']]
            gen_image_target = batch[params['layer_num'] + 1]
            # Upsize the input to match the generation target
            cond_image_input = dataio.images_resize(cond_image_input, (gen_image_target.shape[1], gen_image_target.shape[2])) 

            diff_real = gen_image_target - cond_image_input # The real output we want our generator to produce

            yield (cond_image_input, class_input, diff_real)

    return input_formatter(data)

def build_model(params):
    # First generator 8x8 --> 16x16
    gen = None
    disc = None
    # Inputs
    img_cond = tf.placeholder(tf.float32, name='img_cond', shape=[None, params['fine_shape'][0], params['fine_shape'][1], params['fine_shape'][2]])
    diff_real = tf.placeholder(tf.float32, name='diff_real', shape=[None, params['fine_shape'][0], params['fine_shape'][1], params['fine_shape'][2]])
    class_cond = tf.placeholder(tf.float32, name='class_cond', shape=[None, params['num_classes']])

    keep_prob = tf.placeholder(tf.float32, name='keep_prob', shape=())

    noise = tf.random_normal(tf.shape(img_cond)[:-1], stddev=params['noise'])

    gen = None
    with tf.variable_scope('generator') as scope:
        gen = _make_generator(noise, img_cond, class_cond, 
                              data_shape=params['fine_shape'], 
                              nplanes=params['num_planes'])

    yreal = None
    yreal_logits = None
    yfake = None
    yfake_logits = None
    with tf.variable_scope('discriminator') as scope:
        yfake, yfake_logits = _make_discriminator(gen, class_cond, keep_prob,
                    data_shape=params['fine_shape'], nplanes=params['num_planes']) 
        scope.reuse_variables()
        yreal, yreal_logits =  _make_discriminator(diff_real, class_cond, keep_prob,
                    data_shape=params['fine_shape'], nplanes=params['num_planes']) 

    # The variable weights
    gen_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')
    disc_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc')
    
    return ((gen_weights, disc_weights), (img_cond, class_cond, diff_real, keep_prob), (yfake, yreal, yfake_logits, yreal_logits, gen))

def _make_generator(noise, img_cond, class_cond, data_shape, nplanes=128):
    class_weights = tf.get_variable('class_weights', [10, data_shape[0] * data_shape[1]])
    # We don't actually need a bias here as we are learning a bitplane per class anyways
    class_vec = utils.dense(class_weights)(class_cond)
    class_plane = tf.reshape(class_vec, [-1, data_shape[0], data_shape[1], 1])

    # Reshape the noise
    noise_plane = tf.reshape(noise, [-1, data_shape[0], data_shape[1], 1])

    # Now concatenate the tensors
    stacked_input = tf.concat([img_cond, noise_plane, class_plane], axis=3)

    g1_weights = tf.get_variable('g1_weights', [7, 7, data_shape[2] + 2, nplanes])
    g1_bias = tf.get_variable('g1_bias', [nplanes])
    g1 = tf.nn.relu(utils.conv2d(g1_weights, bias=g1_bias)(stacked_input))

    g2_weights = tf.get_variable('g2_weights', [7, 7, nplanes, nplanes])
    g2_bias = tf.get_variable('g2_bias', [nplanes])
    g2 = tf.nn.relu(utils.conv2d(g2_weights, bias=g2_bias)(g1))

    g3_weights = tf.get_variable('g3_weights', [5, 5, nplanes, data_shape[2]])
    g3 = tf.nn.relu(utils.conv2d(g3_weights)(g2))

    return g3

def _make_discriminator(gen_input, class_cond, keep_prob, data_shape, nplanes=128):
    class_weights = tf.get_variable('class_weights', [10, data_shape[0] * data_shape[1]])
    # We don't actually need a bias here as we are learning a bitplane per class anyways
    class_vec = utils.dense(class_weights)(class_cond)
    class_plane = tf.reshape(class_vec, [-1, data_shape[0], data_shape[1], 1])

    # Now concatenate the tensors
    stacked_input = tf.concat([gen_input, class_plane], axis=3)

    g1_weights = tf.get_variable('g1_weights', [5, 5, data_shape[2] + 1, nplanes])
    g1_bias = tf.get_variable('g1_bias', [nplanes])
    g1 = tf.nn.relu(utils.conv2d(g1_weights, bias=g1_bias)(stacked_input))

    g2_weights = tf.get_variable('g2_weights', [5, 5, nplanes, nplanes])
    g2_bias = tf.get_variable('g2_bias', [nplanes])
    g2 = tf.nn.relu(utils.conv2d(g2_weights, bias=g2_bias)(g1))

    flattened = tf.reshape(g2, [-1, data_shape[0] * data_shape[1] * nplanes])
    dropout = tf.nn.dropout(flattened, keep_prob)

    dense_weights = tf.get_variable('dense_weights', [data_shape[0] * data_shape[1] * nplanes, 1])
    dense_biases = tf.get_variable('dense_bias', [1])
    y_logits = utils.dense(dense_weights, bias=dense_biases)(dropout)
    y = tf.nn.sigmoid(y_logits)

    return y, y_logits
