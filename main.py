import argparse
import importlib
import itertools
import sys
import os

import lapgan
import keras.backend as K
from keras.callbacks import TensorBoard

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

params = config.setup_params(layer_num)

# Get the data
print('Reading data...')
(training_data, test_data, sample_data) = config.read_data()

print('Building model...')
model, image_sampler, callbacks = config.build_model_layer(layer_num)
training_xy = config.build_batches_layer(layer_num, training_data)
test_xy = config.build_batches_layer(layer_num, test_data)

if K.backend() == "tensorflow":
    tensorboard = TensorBoard(log_dir=
                              os.path.join(params['output-dir'],'logs'),
                              histogram_freq=0, write_graph=True)
    imager = lapgan.TensorImageCallback(image_sampler, sample_data, tensorboard)
    callbacks.append(imager)
    callbacks.append(tensorboard)

history = model.fit_generator(generator=training_xy,
                              steps_per_epoch=params['steps-per-epoch'],
                              epochs=params['epochs'],
                              callbacks=callbacks,
                              validation_data=test_xy,
                              validation_steps=params['validation-steps'],
                              initial_epoch=params['initial-epoch'])
