import dataio

import os

import numpy as np
import utils

import tensorflow as tf

PARAMS_L1 = {'layer_num': 1,
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
            cond_image_input = utils.images_resize(cond_image_input, (gen_image_target.shape[1], gen_image_target.shape[2])) 

            diff_real = gen_image_target - cond_image_input # The real output we want our generator to produce

            ones = np.ones((batch[3].shape[0], 1))
            zeros = np.zeros((batch[3].shape[0], 1))

            gen_fake_target = ones # Generator should maximize the fake-pass rate
            gen_real_target = zeros # Doesn't really matter as the generator isn't included in the gradient
            disc_fake_target = zeros # Discriminator wants to keep out the fakes
            disc_real_target = ones # And include the reals
            yield ([cond_image_input, class_input, class_input, diff_real, class_input],
                   [gen_fake_target, gen_real_target, disc_fake_target, disc_real_target])

    return input_formatter(data)

def build_model(params):
    # First generator 8x8 --> 16x16
    gen = None
    disc = None
    # The variable weight
    # scopes so we can reuse the variables
    gen_weights = tf.variable_scope('gen')
    disc_weights = tf.variable_scope('disc')

    # Inputs
    img_cond = tf.placeholder(tf.float32, shape=[None, params['fine_shape'][0], params['fine_shape'][1], params['fine_shape'][2]])

    class_cond = tf.placeholder(tf.float32, shape=[None, params['num_classes']])
    noise = tf.random_normal(tf.shape(img_cond)[:-1], stddev=params['noise'])

    gen = _make_generator(gen_weights, noise, img_cond, class_cond, 
                            data_shape=params['fine_shape'], 
                            nplanes=params['num_planes'])
    #disc = _make_discriminator(disc_weights, data_shape=params['fine_shape'], num_classes=params['num_classes'],
    #                            nplanes=params['num_planes'])

def _make_generator(weights, noise, img_cond, class_cond, data_shape, nplanes=128):
    pass


