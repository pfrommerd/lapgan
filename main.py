import argparse
import importlib
import itertools
import sys
import os

import callbacks

import keras.backend as K
from keras.callbacks import TensorBoard

# parse the arguments
parser = argparse.ArgumentParser(description='Train a network on the CIFAR10 dataset')
parser.add_argument('--layernum', default=0,
                    help='The layer number to train') 
parser.add_argument('--config', default='config.cifar',
                    help='The name of the config to use')
args = parser.parse_args()

layer_num = args.layernum

configModule = args.config
config = importlib.import_module(configModule)

# Get the data
print('Reading data...')
(training_data, test_data, sample_data) = config.read_data()


print('Building model...')
(model, generator, discriminator) = config.build_model_layer(layer_num)
#(training_xy, test_xy, sample_x) = config.build_targets_layer(layer_num, training_data, test_data, sample_data)

"""
if K.backend() == "tensorflow":
    tensorboard = TensorBoard(log_dir=
                              os.path.join(params['output-dir'],'logs'),
                              histogram_freq=0, write_graph=True)
    imager = callbacks.TensorImageCallback(sampler, sample_data, tensorboard)
    calls.append(imager)
    calls.append(tensorboard)


history = model.fit_generator(generator=training_data,
                              steps_per_epoch=params['steps-per-epoch'],
                              epochs=params['epochs'],
                              callbacks=calls,
                              validation_data=test_data,
                              validation_steps=params['validation-steps'],
                              initial_epoch=params['initial-epoch'])
"""
