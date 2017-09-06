import argparse
import importlib
import os
import time

import tensorflow as tf

# parse the arguments
parser = argparse.ArgumentParser(description='Train a network on the CIFAR10 dataset')
parser.add_argument('--layernum', default=0,
                    help='The layer number to train') 
parser.add_argument('--data', default='dataloaders.cifar',
                    help='The discriminator config to use')
parser.add_argument('--aux', default='config.cifar_vae',
                    help='The auxiliary generator config to use')
parser.add_argument('--gen', default='config.cifar_gen',
                    help='The generator config to use')
parser.add_argument('--disc', default='config.cifar_disc',
                    help='The discriminator config to use')
args = parser.parse_args()

layer_num = int(args.layernum)

aux = importlib.import_module(args.aux)
gen = importlib.import_module(args.gen)
disc = importlib.import_module(args.disc)
data = importlib.import_module(args.data)

params = aux.get_params(layer_num)

with tf.Session() as sess:
    # Get the data
    print('Reading data...')
    data_pipeline = None
    with tf.name_scope('data_pipline'):
        data_pipeline = data.build_data_pipeline(params, preload=True, test=False)

    print('Building auxiliary model...')
    aux_gen, aux_train, aux_test = aux.build_model(params, data_pipeline)

    print('Building generator model...')
    gen_gen, gen_weights, gen_test = gen.build_model(params, data_pipeline)

    writer = tf.summary.FileWriter(os.path.join(params['output_dir'], 'logs'))

    saver = tf.train.Saver()

    print('Testing...')
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
        aux_test(epoch, sess, writer)
        gen_test(epoch, sess, writer)

    coord.request_stop()
    coord.join(threads)
