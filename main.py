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
parser.add_argument('--config', nargs=1,
                    help='The name of the config to use')
args = parser.parse_args()

config = None
if args.config is not None:
    configModule = args.config[0]
    #try:
    config = importlib.import_module(configModule)
    #except ImportError:
    #    print('Could not find config module %s' % configModule)
    #    sys.exit(1)
else:
    import config.cifar as config
# Get params
params = config.get_config_params(args)

# Get the data
print('Reading data...')
(training_data, test_data, sample_data) = config.read_data(params)


print('Building model...')
model, saver, sampler = config.build_model(params)

# Make the callback list
calls = [callbacks.ModelSaver(saver)]

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

