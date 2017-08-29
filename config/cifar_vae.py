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

# Will be called by main script
from dataloaders.cifar import build_data_pipeline

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
    encoding_loss = tf.reduce_sum(tf.abs(gen_output['encoded']))
    loss = gen_loss + 0.05 * encoding_loss

    fake_diff_summary = utils.diff_summary('fake_diff', gen_output['diff_img'])
    real_diff_summary = utils.diff_summary('real_diff', diff_real)

    fake_img_summary = tf.summary.image('fake_img', base_img + gen_output['diff_img'])
    real_img_summary = tf.summary.image('real_img', base_img + diff_real)

    base_img_summary = tf.summary.image('input_img', base_img)

    encoding_loss_test_summary = tf.summary.scalar('encoding_test_loss', encoding_loss)
    gen_loss_test_summary = tf.summary.scalar('gen_test_loss', gen_loss)
    loss_test_summary = tf.summary.scalar('test_loss', loss)

    test_summaries_op = tf.summary.merge_all()

    encoding_loss_summary = tf.summary.scalar('encoding_loss', encoding_loss)
    gen_loss_summary = tf.summary.scalar('gen_loss', gen_loss)
    loss_summary = tf.summary.scalar('loss', loss)

    train_summaries_op = tf.summary.merge([gen_loss_summary, encoding_loss_summary, loss_summary])

    gen_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder')
    gen_opt = tf.train.AdamOptimizer(1e-4).minimize(gen_loss, var_list=gen_weights)

    def train(iteration, sess, writer):
        _, train_summary = sess.run([gen_opt, train_summaries_op]) 
        writer.add_summary(train_summary, iteration)

    def test(iteration, sess, writer):
        test_summary = sess.run(test_summaries_op) 
        writer.add_summary(test_summary, iteration)
        writer.flush()

    return train, test

def _build_autoencoder(inputs, data_shape, num_planes, latent_dim):
    base_img = inputs['base_img']



    d1_weights = tf.get_variable('d1_weights', [7, 7, data_shape[2], num_planes])
    d1_bias = tf.get_variable('d1_bias', [num_planes])
    d1 = utils.conv2d(d1_weights, bias=d1_bias)(base_img)

    d2_weights = tf.get_variable('d2_weights', [7, 7, num_planes, num_planes])
    d2_bias = tf.get_variable('d2_bias', [num_planes])
    d2 = utils.conv2d(d2_weights, bias=d2_bias)(d1)

    d2_flattened = tf.reshape(d2, (-1, data_shape[0]*data_shape[1]*num_planes))

    d3_weights = tf.get_variable('d3_weights', [data_shape[0]*data_shape[1]*num_planes, latent_dim])
    d3_bias = tf.get_variable('d3_bias', [latent_dim])
    d3 = utils.dense(d3_weights, bias=d3_bias)(d2_flattened)

    # Add the class data to the representation
    h = tf.concat([d3, inputs['class_cond']], axis=1)

    e1_weights = tf.get_variable('e1_weights', [latent_dim + 10, data_shape[0]*data_shape[1]*num_planes])
    e1_bias = tf.get_variable('e1_bias', [data_shape[0]*data_shape[1]*num_planes])
    e1 = utils.dense(e1_weights, bias=e1_bias)(h)

    e1_reshaped = tf.reshape(e1, (-1, data_shape[0], data_shape[1], num_planes))

    e2_weights = tf.get_variable('e2_weights', [7, 7, num_planes, num_planes])
    e2_bias = tf.get_variable('e2_bias', [num_planes])
    e2 = utils.conv2d_transpose(e2_weights, (data_shape[0], data_shape[1], num_planes), bias=e2_bias)(e1_reshaped)

    e3_weights = tf.get_variable('e3_weights', [7, 7, num_planes, data_shape[2]])
    e3_bias = tf.get_variable('e3_bias', [data_shape[2]])
    e3 = utils.conv2d_transpose(e3_weights, (data_shape[0], data_shape[1], data_shape[2]), bias=e3_bias)(e2)

    diff_img = e3

    return {'encoded': h, 'reconst_img': e1, 'diff_img': diff_img}
