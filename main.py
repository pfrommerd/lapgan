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
(gen_weights, disc_weights), (img_cond, class_cond, diff_real, keep_prob), (yfake, yreal, gen_out) = config.build_model(params)

# Create the loss functions for yfake, yreal
disc_loss = -tf.log(yreal) - tf.log(1 - yfake)
gen_loss = -tf.log(yfake)

# Make the tensorboard visualizations
train_disc_loss = tf.summary.scalar('disc_loss', tf.reduce_mean(disc_loss))
train_gen_loss = tf.summary.scalar('gen_loss', tf.reduce_mean(gen_loss))

train_summary_op = tf.summary.merge([train_disc_loss, train_gen_loss])

# And the test summaries

test_disc_loss = tf.summary.scalar('test_disc_loss', tf.reduce_mean(disc_loss))
test_gen_loss = tf.summary.scalar('test_gen_loss', tf.reduce_mean(gen_loss))

test_gen_diff = tf.summary.image('test_gen_diff', 0.5 * (gen_out + 1), max_outputs=params['sample_img_num'])
test_gen_img = tf.summary.image('test_gen_img', gen_out + img_cond, max_outputs=params['sample_img_num'])

test_summary_op = tf.summary.merge([test_disc_loss, test_gen_loss, test_gen_diff, test_gen_img])

gt_gen_diff = tf.summary.image('gt_gen_diff', diff_real, max_outputs=params['sample_img_num'])
gt_gen_input = tf.summary.image('gt_gen_img', img_cond + diff_real, max_outputs=params['sample_img_num'])

gt_summary_op = tf.summary.merge([gt_gen_diff, gt_gen_input])

writer = tf.summary.FileWriter(os.path.join(params['output_dir'], 'logs'), graph=tf.get_default_graph())

# Create the optimizers
disc_opt = tf.train.AdamOptimizer(1e-4).minimize(disc_loss, var_list=disc_weights)
gen_opt = tf.train.AdamOptimizer(1e-4).minimize(gen_loss, var_list=gen_weights)

print('Training...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    test_feed_dict = {
        img_cond: test_batch[0],
        class_cond: test_batch[1],
        diff_real: test_batch[2],
        keep_prob: 1
    }

    # Write the ground truth images
    gt_summary = sess.run(gt_summary_op, feed_dict=test_feed_dict)
    writer.add_summary(gt_summary, 0)
    writer.flush()

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
            # Update everything
            train_summary, _, _ = sess.run([train_summary_op, disc_opt, gen_opt], feed_dict=feed_dict)
            # Log to tensorboard
            writer.add_summary(train_summary, epoch * params['steps_per_epoch'] + i)

        # now sample the validation losses and images
        test_summary = sess.run(test_summary_op, feed_dict=test_feed_dict)
        writer.add_summary(test_summary, epoch * params['steps_per_epoch'])
        writer.flush()
        print()
