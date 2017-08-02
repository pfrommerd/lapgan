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
    # Inputs
    base_img = tf.placeholder(tf.float32, name='base_img', shape=[None, params['fine_shape'][0], params['fine_shape'][1], params['fine_shape'][2]])
    diff_real = tf.placeholder(tf.float32, name='diff_real', shape=[None, params['fine_shape'][0], params['fine_shape'][1], params['fine_shape'][2]])
    class_cond = tf.placeholder(tf.float32, name='class_cond', shape=[None, params['num_classes']])

    keep_prob = tf.placeholder(tf.float32, name='keep_prob', shape=())

    noise = tf.random_normal(tf.shape(base_img)[:-1], stddev=params['noise'], name='noise')

    gen_output = None
    with tf.variable_scope('generator') as scope:
        gen_output = _make_generator({'class_cond': class_cond, 'base_img': base_img, 'noise': noise},
                              data_shape=params['fine_shape'], 
                              nplanes=params['num_planes'])

    # Add the keep_prob to the disc
    # input from the gen_output
    gen_output['keep_prob'] = keep_prob

    fake_output = None
    real_output = None
    with tf.variable_scope('disc') as scope:
        with tf.name_scope('fake'):
            fake_output = _make_discriminator(gen_output, data_shape=params['fine_shape'], nplanes=params['num_planes'])

        scope.reuse_variables()

        with tf.name_scope('real'):
            real_output = _make_discriminator({'diff_img': diff_real, 'keep_prob': keep_prob}, data_shape=params['fine_shape'], nplanes=params['num_planes'])

    real_prob = fake_output['prob_real']
    fake_prob = fake_output['prob_real']

    real_logits = fake_output['logits_real']
    fake_logits = fake_output['logits_real']
    
    # The variable weights
    gen_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')
    disc_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc')

    # 
    ones = tf.ones(tf.stack([tf.shape(fake_logits)[0], 1]))
    zeros = tf.zeros(tf.stack([tf.shape(fake_logits)[0], 1]))

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=zeros) + \
                                    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=ones))

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=ones))

    fake_diff_summary = tf.summary.image('fake_diff', gen_output['diff_img'])
    real_diff_summary = tf.summary.image('real_diff', diff_real)

    fake_img_summary = tf.summary.image('fake_img', base_img + gen_output['diff_img'])
    real_img_summary = tf.summary.image('real_img', base_img + diff_real)

    base_img_summary = tf.summary.image('input_img', base_img)


    disc_loss_test_summary = tf.summary.scalar('disc_test_loss', disc_loss)
    gen_loss_test_summary = tf.summary.scalar('gen_test_loss', gen_loss)

    real_test_summary = tf.summary.scalar('yreal_test', tf.reduce_mean(real_prob))
    fake_test_summary = tf.summary.scalar('yfake_test', tf.reduce_mean(fake_prob))

    test_summaries_op = tf.summary.merge_all()

    disc_loss_summary = tf.summary.scalar('disc_loss', disc_loss)
    gen_loss_summary = tf.summary.scalar('gen_loss', gen_loss)

    real_summary = tf.summary.scalar('yreal', tf.reduce_mean(real_prob))
    fake_summary = tf.summary.scalar('yfake', tf.reduce_mean(fake_prob))

    train_summaries_op = tf.summary.merge([real_summary, fake_summary, disc_loss_summary, gen_loss_summary])

    disc_opt = tf.train.AdamOptimizer(1e-4).minimize(disc_loss, var_list=disc_weights)
    gen_opt = tf.train.AdamOptimizer(1e-4).minimize(gen_loss, var_list=disc_weights)

    def train(iteration, sess, writer, feed_dict):
        _, _, train_summary = sess.run([disc_opt, gen_opt, train_summaries_op], feed_dict=feed_dict) 
        writer.add_summary(train_summary, iteration)

    def test(iteration, sess, writer, feed_dict):
        test_summary = sess.run(test_summaries_op, feed_dict=feed_dict) 
        writer.add_summary(test_summary, iteration)
        writer.flush()

    return train, test

def _make_generator(inputs, data_shape, nplanes=128):
    class_cond = inputs['class_cond']
    base_img = inputs['base_img']
    noise = inputs['noise']

    class_weights = tf.get_variable('class_weights', [10, data_shape[0] * data_shape[1]])

    tf.summary.image('gen_class_weights', tf.reshape(class_weights, [10, data_shape[0], data_shape[1], 1]), max_outputs=10)

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

def _make_discriminator(inputs, data_shape, nplanes=128):
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

    dense_weights = tf.get_variable('dense_weights', [data_shape[0] * data_shape[1] * nplanes, 1])
    dense_biases = tf.get_variable('dense_bias', [1])
    y_logits = utils.dense(dense_weights, bias=dense_biases)(dropout)
    y = tf.nn.sigmoid(y_logits, name='prob_real')

    return {'prob_real': y, 'logits_real': y_logits}
