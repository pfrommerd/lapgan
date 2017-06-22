import matplotlib.pyplot as plt
import numpy as np

import keras
from keras import Input
from keras.layers import Reshape, Dense, Flatten, Activation, LeakyReLU
from keras.models import Sequential, Model
from keras import regularizers

from lapgan import build_lapgan, make_lapgan_targets, make_gaussian_pyramid, make_laplacian_pyramid

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

# Build the model
g1 = make_generator(0, input_shape=(8, 8, 3), output_shape=(16,16,3), name="g1")
g2 = make_generator(1, input_shape=(16, 16, 3), output_shape=(32,32,3), name="g2")

d1 = make_discriminator(0, input_shape=(16,16,3), name="d1")
d2 = make_discriminator(1, input_shape=(32,32,3), name="d2")


lapgan_training, lapgan_generative = build_lapgan([g1, g2], [d1, d2], False)

# Get the data
print("Importing cifar10")
from keras.datasets import cifar10
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

print("Formatting data")
# Process the data
xtrain = xtrain.astype(np.float32) / 255.0
xtest = xtest.astype(np.float32) / 255.0

print("Building pyramid")
# Returns a 5-dimensional numpy array
#(an array of pyramids, which are arrays of 3 channel, 2d images)
xtrain_pyramid =make_laplacian_pyramid(make_gaussian_pyramid(xtrain, 3, 2))
#xtest_pyramid = lapgan_make_pyramid(xtest, 3, 2)

