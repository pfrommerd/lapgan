import argparse
import importlib
import os

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
    aux_gen, aux_train, _ = aux.build_model(params, data_pipeline)

    print('Building generator model...')
    gen_gen, gen_weights, _ = gen.build_model(params, data_pipeline)

    print('Building discriminator model...')
    disc_real_inputs = {'base_img':data_pipeline['base_img'], 'diff_img':data_pipeline['diff_real'],
            'keep_prob':data_pipeline['keep_prob'], 'name': 'real', 'target_prob':1}
    # Target for the generator side training: 
    disc_gen_inputs = {'base_img':data_pipeline['base_img'], 'diff_img':gen_gen,
            'keep_prob':data_pipeline['keep_prob'], 'name': 'gen', 'target_prob':1}
    disc_aux_inputs = {'base_img':data_pipeline['base_img'], 'diff_img':aux_gen,
            'keep_prob':data_pipeline['keep_prob'], 'name': 'aux', 'target_prob':0}
    
    _, _, _, real_summary, disc_real_loss, real_weights = disc.build_model(params, disc_real_inputs)
    _, _, _, aux_summary, disc_aux_loss, aux_weights = disc.build_model(params, disc_aux_inputs, reuse=True)
    _, _, _, gen_summary, disc_gen_loss, gen_weights = disc.build_model(params, disc_gen_inputs, reuse=True, use_weights=gen_weights)

    combined_loss = disc_real_loss + disc_aux_loss + disc_gen_loss
    combined_summary = tf.summary.merge([real_summary, aux_summary, gen_summary])
    opt = tf.train.AdamOptimizer(1e-4).minimize(combined_loss, var_list=[real_weights, aux_weights, gen_weights])

    def train(iteration, sess, writer):
        _, train_summary = sess.run([opt, combined_summary])
        writer.add_summary(train_summary, iteration)


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
            aux_train(epoch*params['steps_per_epoch'] + i, sess, writer)
            train(epoch*params['steps_per_epoch'] + i, sess, writer)
            #disc_aux_train(epoch*params['steps_per_epoch'] + i, sess, writer)
            #disc_real_train(epoch*params['steps_per_epoch'] + i, sess, writer)
            #disc_gen_train(epoch*params['steps_per_epoch'] + i, sess, writer)
        print()
        saver.save(sess, (params['output_dir'] + '/checkpoints/e_%04d.ckpt') % (epoch + 1))

    coord.request_stop()
    coord.join(threads)
