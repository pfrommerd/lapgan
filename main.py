import argparse
import importlib
import itertools
import sys

# parse the arguments
parser = argparse.ArgumentParser(description='Train a network on the CIFAR10 dataset')
parser.add_argument('config', nargs=1,
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

# Get params
params = config.get_config_params(args)

# Get the data
(training_data, test_data) = config.read_data(params)

import matplotlib.pyplot as plt

imgs = next(training_data)
print(imgs[0].shape)

plt.imshow(imgs[0])
plt.show()
