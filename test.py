import argparse
import importlib
import os
import time

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
data_pipeline = config.build_data_pipeline(params, test=True)

print('Building model...')
_, test = config.build_model(params, data_pipeline)

writer = tf.summary.FileWriter(os.path.join(params['output_dir'], 'logs'))

saver = tf.train.Saver()

print('Training...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for epoch in range(1, params['epochs'] + 1):
        print('Waiting for Epoch %d' % epoch)
        checkpoint_path = params['output_dir'] + '/checkpoints/e_%04d.ckpt' % epoch 
        while not os.path.exists(checkpoint_path + '.index'):
            time.sleep(15)
        time.sleep(1)
        print('Evaluating Epoch %d/%d' % (epoch, params['epochs'])) 
        saver.restore(sess, checkpoint_path)
        test(epoch, sess, writer)

    coord.request_stop()
    coord.join(threads)
