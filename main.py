import argparse
import importlib

import tensorflow as tf

# parse the arguments
parser = argparse.ArgumentParser(description='Train a network on the CIFAR10 dataset')
parser.add_argument('--layernum', default=0,
                    help='The layer number to train') 
parser.add_argument('--config', default='config.cifar',
                    help='The name of the config to use')
args = parser.parse_args()

layer_num = int(args.layernum)

configModule = args.config
config = importlib.import_module(configModule)

params = config.get_params(layer_num)

# Get the data
print('Reading data...')
(training_data, test_data, sample_data) = config.build_data_pipeline(params)

training_batches = config.format_batches(params, training_data)

print('Building model...')
(gen_weights, disc_weights), (img_cond, class_cond, diff_real, keep_prob), (yfake, yreal) = config.build_model(params)

# Create the loss functions for yfake, yreal
disc_loss = -tf.log(yreal) - tf.log(1 - yfake)
gen_loss = -tf.log(yfake)

# Create the optimizers
disc_opt = tf.train.AdamOptimizer(1e-4).minimize(disc_loss, var_list=disc_weights)
gen_opt = tf.train.AdamOptimizer(1e-4).minimize(gen_loss, var_list=gen_weights)

print('Training...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(params['epochs']):
        print('Epoch %d/%d' % (epoch + 1, params['epochs'])) 
        for i, (cond_img, cond_class, real_diff) in zip(range(params['steps_per_epoch']), training_batches):
            print('Batch %d/%d' % (i + 1, params['steps_per_epoch']), end='\r')
            feed_dict = {
                 img_cond: cond_img,
                class_cond: cond_class,
                diff_real: real_diff,
                keep_prob: 0.5
            }
            # Get the gradients for each
            disc_opt.run(feed_dict)
            gen_opt.run(feed_dict)
        print()
