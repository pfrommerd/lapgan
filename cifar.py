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

from lapgan import build_lapgan, make_lapgan_targets, make_gaussian_pyramid, make_laplacian_pyramid

from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, AdversarialOptimizerAlternating, normal_latent_sampling

def make_generator(layer_num, input_shape, output_shape,
                   latent_dim=100, hidden_dim=2048,
                   name="generator", reg=lambda: regularizers.l1_l2(1e-5, 1e-5)):
    conditional_input = Input(name="g%d_conditional_input" % layer_num, shape=input_shape)
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

g1 = make_generator(0, input_shape=(8, 8, 3), output_shape=(16,16,3), latent_dim=latent_dim, name="g1")
g2 = make_generator(1, input_shape=(16, 16, 3), output_shape=(32,32,3), latent_dim=latent_dim, name="g2")

d1 = make_discriminator(0, input_shape=(16,16,3), name="d1")
d2 = make_discriminator(1, input_shape=(32,32,3), name="d2")

player_params = [g1.trainable_weights, d1.trainable_weights, g2.trainable_weights, d2.trainable_weights]
# Player order must be generator/discriminator pairs
# due to how make_lapgan_targets data is formatted
player_names = ["generator_1", "discriminator_1", "generator_2", "discriminator_2"]

lapgan_training, lapgan_generative = build_lapgan([g1, g2], [d1, d2],
                                                  normal_latent_sampling((latent_dim,)), False)

model = AdversarialModel(base_model=lapgan_training, player_params=player_params, player_names=player_names)

model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerAlternating(),
                          player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4),
                                             Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                          loss='binary_crossentropy')

# ----------------- Data ------------------

print("Importing cifar10")
from keras.datasets import cifar10
(xtrain, ytrain_cat), (xtest, ytest_cat) = cifar10.load_data()

print("Formatting data")
# Process the data
xtrain = xtrain.astype(np.float32) / 255.0
xtest = xtest.astype(np.float32) / 255.0

print("Building pyramid")
# Returns a 5-dimensional numpy array
#(an array of pyramids, which are arrays of 3 channel, 2d images)
xtrain_pyramid = make_laplacian_pyramid(make_gaussian_pyramid(xtrain, 3, 2))
ytrain = make_lapgan_targets(num_layers=2, num_samples=xtrain.shape[0])

xtest_pyramid = make_laplacian_pyramid(make_gaussian_pyramid(xtest, 3, 2))
ytest = make_lapgan_targets(num_layers=2, num_samples=xtest.shape[0])

#xtest_pyramid = lapgan_make_pyramid(xtest, 3, 2)

# -------------- Callbacks -----------------

output_dir='output/lapgan'

# Make a callback to periodically generate images after each epoch
zsamples1 = np.random.normal(size=(10 * 10, latent_dim))
zsamples2 = np.random.normal(size=(10 * 10, latent_dim))
base_imgs = xtest_pyramid[0][0:100]

def generator_sampler():
    return lapgan_generative.predict([zsamples1, zsamples2, base_imgs])[-1].reshape((10, 10, 32, 32, 3))

generator_cb = ImageGridCallback(os.path.join(output_dir, "epoch-{:03d}.png"), generator_sampler)

callbacks = [generator_cb]
if K.backend() == "tensorflow":
    callbacks.append(
        TensorBoard(log_dir=os.path.join(output_dir, 'logs'), histogram_freq=0, write_graph=True, write_images=True))

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

