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


def build_model(params, inputs, reuse=False, use_weights=None):
    # Inputs
    diff_img = inputs['diff_img']
    keep_prob = inputs['keep_prob']

    name = inputs['name']
    target_prob = inputs['target_prob']

    disc_output = None
    with tf.variable_scope('disc', reuse=reuse) as scope:
        disc_output = _build_discriminator(inputs, params['fine_shape'], params['num_planes'])

        prob_logits = disc_output['prob_logits']
        prob = disc_output['prob']

        loss = tf.reduce_mean(utils.logits_sigmoid_cross_entropy(prob_logits, target_prob))

        prob_summary = tf.summary.scalar('%s_prob' % name, tf.reduce_mean(prob))
        loss_summary = tf.summary.scalar('%s_loss' % name, loss)

        train_summaries_op = tf.summary.merge([prob_summary, loss_summary])

        prob_test_summary = tf.summary.scalar('%s_test_prob' % name, tf.reduce_mean(prob))
        loss_test_summary = tf.summary.scalar('%s_test_loss' % name, loss)

        test_summaries_op = tf.summary.merge([prob_test_summary, loss_test_summary])


    if use_weights is not None:
        weights = use_weights
        opt = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=use_weights)
    else:
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc')
        with tf.variable_scope('disc', reuse=reuse) as scope:
            opt = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=weights)

    def train(iteration, sess, writer):
        _, train_summary = sess.run([opt, train_summaries_op]) 
        writer.add_summary(train_summary, iteration)

    def test(iteration, sess, writer):
        test_summary = sess.run(test_summaries_op) 
        writer.add_summary(test_summary, iteration)
        writer.flush()

    return prob, train, test, train_summaries_op, loss, weights

def _build_discriminator(inputs, data_shape, nplanes):
    diff_input = inputs['diff_img']
    keep_prob = inputs['keep_prob']

    g1_weights = tf.get_variable('g1_weights', [5, 5, 3, nplanes])
    g1_bias = tf.get_variable('g1_bias', [nplanes])
    g1 = tf.nn.relu(utils.conv2d(g1_weights, bias=g1_bias)(diff_input))

    g2_weights = tf.get_variable('g2_weights', [5, 5, nplanes, nplanes])
    g2_bias = tf.get_variable('g2_bias', [nplanes])
    g2 = tf.nn.relu(utils.conv2d(g2_weights, bias=g2_bias)(g1))

    flattened = tf.reshape(g2, [-1, data_shape[0] * data_shape[1] * nplanes])
    dropout = tf.nn.dropout(flattened, keep_prob)

    prob_weights = tf.get_variable('prob_weights', [data_shape[0] * data_shape[1] * nplanes, 1])
    prob_biases = tf.get_variable('prob_bias', [1])
    y_logits = utils.dense(prob_weights, bias=prob_biases)(dropout)
    y = tf.nn.sigmoid(y_logits, name='prob_real')

    #class_weights = tf.get_variable('class_weights', [data_shape[0] * data_shape[1] * nplanes, 10])
    #class_biases = tf.get_variable('class_bias', [1])
    #class_logits = utils.dense(class_weights, bias=class_biases)(dropout)
    #class_prob = tf.nn.softmax(class_logits, name='logits_class')

    return {'prob': y, 'prob_logits': y_logits}#, 'prob_class': class_prob, 'logits_class': class_logits}
