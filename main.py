import argparse
import importlib
import os

import tensorflow as tf

# parse the arguments
parser = argparse.ArgumentParser(description='Train a network on the CIFAR10 dataset')
parser.add_argument('--layernum', default=0,
                    help='The layer number to train') 
parser.add_argument('--config', default='config.cifar_vae',
                    help='The name of the config to use')
args = parser.parse_args()

layer_num = int(args.layernum)

configModule = args.config
config = importlib.import_module(configModule)

params = config.get_params(layer_num)

with tf.Session() as sess:
    # Get the data
    print('Reading data...')
    data_pipeline = None
    with tf.name_scope('data_pipline'):
        data_pipeline = config.build_data_pipeline(params, preload=True, test=False)

    print('Building model...')
    train, _ = config.build_model(params, data_pipeline)

    writer = tf.summary.FileWriter(os.path.join(params['output_dir'], 'logs'), graph=tf.get_default_graph())

    saver = tf.train.Saver(max_to_keep=0)

    print('Training...')
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for epoch in range(params['epochs']):
        print('Epoch %d/%d' % (epoch + 1, params['epochs'])) 
        for i in range(params['steps_per_epoch']):
            print('Batch %d/%d' % (i + 1, params['steps_per_epoch']), end='\r')
#            print()
#            print(sess.run('data_pipline/io_size:0'))
            train(epoch*params['steps_per_epoch'] + i, sess, writer)
        print()
        saver.save(sess, (params['output_dir'] + '/checkpoints/e_%04d.ckpt') % (epoch + 1))

    coord.request_stop()
    coord.join(threads)
