import argparse
import importlib
import itertools
import sys
import os

import matplotlib.pyplot as plt
# Disable keras so we don't load tensorflow
sys.modules['keras'] = None

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

print(sample_data[3].shape)
print(sample_data[2].shape)

plt.figure(0)
for i in range(0, 64):
    for n in range(0, sample_data[2].shape[0], 64):
        plt.imshow(sample_data[2][i + n])
        plt.show()
