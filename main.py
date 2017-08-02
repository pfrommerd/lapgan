import argparse
import importlib
import os

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
(training_data, test_data) = config.build_data_pipeline(params)
test_batch = next(config.format_batches(params,test_data)) # We will only use the first batch for sampling validations

training_batches = config.format_batches(params, training_data)

print('Building model...')
train, test = config.build_model(params)

writer = tf.summary.FileWriter(os.path.join(params['output_dir'], 'logs'), graph=tf.get_default_graph())

print('Training...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    test_feed_dict = {
            'base_img:0': test_batch[0],
            'class_cond:0': test_batch[1],
            'diff_real:0': test_batch[2],
            'keep_prob:0': 1
    }

    for epoch in range(params['epochs']):
        print('Epoch %d/%d' % (epoch + 1, params['epochs'])) 
        for i, (cond_img, cond_class, real_diff) in zip(range(params['steps_per_epoch']), training_batches):
            print('Batch %d/%d' % (i + 1, params['steps_per_epoch']), end='\r')
            feed_dict = {
                    'base_img:0': cond_img,
                    'class_cond:0': cond_class,
                    'diff_real:0': real_diff,
                    'keep_prob:0': 0.5
            }
            train(epoch*params['steps_per_epoch'] + i, sess, writer, feed_dict)

        print()
        test(epoch, sess, writer, test_feed_dict)
