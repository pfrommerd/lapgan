import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import os

import keras
import keras.backend as K

from keras.callbacks import TensorBoard

from keras.layers import Input, Reshape, Dense, Flatten, Activation, LeakyReLU
from keras.models import Sequential, Model
from keras import regularizers

from keras.optimizers import Adam

from lapgan import build_lapgan, make_lapgan_targets, make_mmapped_gaussian_pyramid

from keras_adversarial import AdversarialModel, AdversarialOptimizerAlternating, normal_latent_sampling

from TensorImageCallback import TensorImageCallback

def make_generator(layer_num, output_shape,
                   latent_dim=100, hidden_dim=2048,
                   name="generator", reg=lambda: regularizers.l1_l2(1e-5, 1e-5)):
    # The conditional_input has been upsampled already
    conditional_input = Input(name="g%d_conditional_input" % layer_num, shape=output_shape)
    latent_input = Input(name="g%d_latent_input" % layer_num, shape=(latent_dim,))

    conditional_flatten = Flatten(name="g%d_flatten" % layer_num)(conditional_input)
    combined = keras.layers.concatenate([latent_input, conditional_flatten])

    x = Dense(hidden_dim // 4, name="g%d_h1" % layer_num, kernel_regularizer=reg())(combined)
    a = LeakyReLU(0.2)(x)
    x = Dense(hidden_dim // 2, name="g%d_h2" % layer_num, kernel_regularizer=reg())(a)
    a = LeakyReLU(0.2)(x)
    x = Dense(hidden_dim, name="g%d_h3" % layer_num, kernel_regularizer=reg())(a)
    a = LeakyReLU(0.2)(x)
    x = Dense(np.prod(output_shape), name="g%d_x_flat" % layer_num, kernel_regularizer=reg())(a)
    a = Activation('sigmoid')(x)
    r = Reshape(output_shape, name="g%d_x" % layer_num)(a)

    model = Model([latent_input, conditional_input], [r], name=name)
    return model

def make_discriminator(layer_num, input_shape, hidden_dim=1024, reg=lambda: regularizers.l1_l2(1e-5, 1e-5), name="discriminator"):
    input = Input(name="d%d_input" % layer_num, shape=input_shape)
    flatten = Flatten(name="d%d_flatten" % layer_num)(input)
    x = Dense(hidden_dim, name="d%d_h1" % layer_num, kernel_regularizer=reg())(flatten)
    x = LeakyReLU(0.2)(x)
    x = Dense(hidden_dim // 2, name="d%d_h2" % layer_num, kernel_regularizer=reg())(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(hidden_dim // 4, name="d%d_h3" % layer_num, kernel_regularizer=reg())(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(1, name="d%d_y" % layer_num, kernel_regularizer=reg())(x)
    result = Activation("sigmoid")(x)

    return Model([input], [result], name=name)

latent_dim = 100

# ------------ Build the Model -------------

# First generator 24x24 --> 48x48
g1 = make_generator(0,output_shape=(48,48,3), latent_dim=latent_dim, name="g1")
# Second generator 48x48 --> 96x96
g2 = make_generator(1, output_shape=(96,96,3), latent_dim=latent_dim, name="g2")

d1 = make_discriminator(0, input_shape=(48,48,3), name="d1")
d2 = make_discriminator(1, input_shape=(96,96,3), name="d2")

player_params = [g1.trainable_weights, d1.trainable_weights, g2.trainable_weights, d2.trainable_weights]
# Player order must be generator/discriminator pairs
# due to how make_lapgan_targets data is formatted
player_names = ["generator_1", "discriminator_1", "generator_2", "discriminator_2"]

lapgan_training, lapgan_generative = build_lapgan([g1, g2], [d1, d2],
                                                  normal_latent_sampling((latent_dim,)), True, (24,24,3))

model = AdversarialModel(base_model=lapgan_training, player_params=player_params, player_names=player_names)

model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerAlternating(),
                          player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4),
                                             Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                          loss='binary_crossentropy')

# ----------------- Data ------------------
num_samples=10000

print("Importing stl10")

import stl10_reader
stl10_reader.download_and_extract(convert_numpy=True) # Make sure the data is there
print('Reading data')

# Check to see if the unlabeled laplacian pyramid data exists

# We can now read the arrays directly as they have been converted
# and saved as numpy arrays --> memory map those arrays to save memory
xtrain = np.load(stl10_reader.UNLABELED_DATA_PATH_NPY, mmap_mode='r')
xtest = np.load(stl10_reader.TEST_DATA_PATH_NPY, mmap_mode='r')

print("Building gaussian pyramid")
# Returns a 5-dimensional numpy array
#(an array of pyramids, which are arrays of 3 channel, 2d images)
xtrain_pyramid_paths = ['./data/stl10_binary/train_X_1.npy','./data/stl10_binary/train_X_2.npy' ]
xtest_pyramid_paths = ['./data/stl10_binary/test_X_1.npy','./data/stl10_binary/test_X_2.npy' ]

# Reverse the gaussian pyramids so that the smallest layer is first
xtrain_pyramid = list(reversed(make_mmapped_gaussian_pyramid(xtrain, xtrain_pyramid_paths, 2, limit_samples=num_samples)))
xtest_pyramid = list(reversed(make_mmapped_gaussian_pyramid(xtest, xtest_pyramid_paths, 2, limit_samples=512)))

ytrain = make_lapgan_targets(num_layers=2, num_samples=num_samples)
ytest = make_lapgan_targets(num_layers=2, num_samples=xtest_pyramid[0].shape[0])

# -------------- Callbacks -----------------

output_dir='output/stl'

num_gen_images = 10

# Make a callback to periodically generate images after each epoch
zsamples1 = np.random.normal(size=(num_gen_images, latent_dim))
zsamples2 = np.random.normal(size=(num_gen_images, latent_dim))
base_imgs = xtest_pyramid[0][0:100]

def image_sampler():
    results = lapgan_generative.predict([zsamples1, zsamples2, base_imgs])
    # results contains the base input, the output after layer 1, output after layer 2
    images = [ base_imgs,
               results[0].reshape(num_gen_images, 48, 48, 3),
               results[1].reshape(num_gen_images, 96, 96, 3) ]
    return images

callbacks = []
if K.backend() == "tensorflow":
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, 'logs'), histogram_freq=0, write_graph=True)
    imager = TensorImageCallback(['input', 'generated_1', 'generated_2'],
                                 num_gen_images,
                                 image_sampler, tensorboard)
    callbacks.append(imager)
    callbacks.append(tensorboard)


# -------------- Train ---------------

nb_epoch = 40

history = model.fit(x=xtrain_pyramid, y=ytrain, validation_data=(xtest_pyramid, ytest),
                    callbacks=callbacks, epochs=nb_epoch,
                    batch_size=32)

df = pd.DataFrame(history.history)
df.to_csv(os.path.join(output_dir, 'history.csv'))

# save models
g1.save(os.path.join(output_dir, "generator_1.h5"))
g2.save(os.path.join(output_dir, "generator_2.h5"))
d1.save(os.path.join(output_dir, "discriminator_1.h5"))
d2.save(os.path.join(output_dir, "discrimiantor_2.h5"))

