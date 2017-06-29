import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import os

import keras
import keras.backend as K

from tensorflow.python.client import timeline
import tensorflow as tf

from keras.callbacks import TensorBoard

from keras.layers import Input, Reshape, Dense, Flatten, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras import regularizers

from keras.optimizers import Adam

from lapgan import build_lapgan, make_lapgan_targets, make_gaussian_pyramid

from keras_adversarial import AdversarialModel, AdversarialOptimizerSimultaneous, normal_latent_sampling

from TensorImageCallback import TensorImageCallback

#'''
def make_generator(layer_num, output_shape,
                   latent_dim, nplanes=128,
                   name="generator", reg=lambda: regularizers.l1_l2(1e-5, 1e-5)):
    # The conditional_input has been upsampled already
    latent_input = Input(name="g%d_latent_input" % layer_num, shape=(latent_dim,))
    latent_reshaped = Reshape((output_shape[0], output_shape[1], 1),
                              name="g%d_reshape" % layer_num)(latent_input)
    
    conditional_input = Input(name="g%d_conditional_input" % layer_num, shape=output_shape)
    
    combined = keras.layers.concatenate([latent_reshaped, conditional_input])
    x = Conv2D(nplanes, (7, 7), padding='same', kernel_regularizer=reg(),
               name='g%d_c1' % layer_num)(combined)
    a = Activation('relu')(x)
    x = Conv2D(nplanes, (7, 7), padding='same', kernel_regularizer=reg(),
               name='g%d_c2' % layer_num)(a)
    a = Activation('relu')(x)
    x = Conv2D(3, (5, 5), padding='same', kernel_regularizer=reg(),
               name='g%d_c3' % layer_num)(a)
    a = Activation('tanh')(x)
    r = Reshape(output_shape, name="g%d_x" % layer_num)(a)

    model = Model([latent_input, conditional_input], [r], name=name)
    return model

def make_discriminator(layer_num, input_shape, nplanes=128,
                       reg=lambda: regularizers.l1_l2(1e-5, 1e-5),
                       name="discriminator"):
    input = Input(name="d%d_input" % layer_num, shape=input_shape)
    x = Conv2D(nplanes, (5, 5), padding='same', kernel_regularizer=reg(),
               name='d%d_c1' % layer_num)(input)
    a = Activation('relu')(x)
    x = Conv2D(nplanes, (5, 5), padding='same', kernel_regularizer=reg(),
               name='d%d_c2' % layer_num)(a)
    a = Activation('relu')(x)
    
    flattened = Flatten(name='d%d_flatten' % layer_num)(a)
    a = Activation('relu')(flattened)

    dropout = Dropout(0.5, name='d%d_dropout' % layer_num)(a)

    y = Dense(1, kernel_regularizer=reg(), name='d%d_y' % layer_num)(dropout)
    
    result = Activation("sigmoid")(y)

    return Model([input], [result], name=name)

# ------------ Build the Model -------------

# First generator 8x8 --> 16x16
g1 = make_generator(0,output_shape=(16,16,3), latent_dim=16*16, nplanes=128, name="g1")
# Second generator 16x16 --> 32x32
g2 = make_generator(1, output_shape=(32,32,3), latent_dim=32*32, nplanes=128, name="g2")

d1 = make_discriminator(0, input_shape=(16,16,3), nplanes=128, name="d1")
d2 = make_discriminator(1, input_shape=(32,32,3), nplanes=128, name="d2")

z1 = normal_latent_sampling((16 * 16,))
z2 = normal_latent_sampling((32 * 32,))

player_params = [g1.trainable_weights, d1.trainable_weights, g2.trainable_weights, d2.trainable_weights]
# Player order must be generator/discriminator pairs
# due to how make_lapgan_targets data is formatted
player_names = ["generator_1", "discriminator_1", "generator_2", "discriminator_2"]

lapgan_training, lapgan_generative = build_lapgan([g1, g2], [d1, d2],
                                                  [z1, z2], True, (8,8,3))

model = AdversarialModel(base_model=lapgan_training, player_params=player_params, player_names=player_names)

model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                          player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4),
                                             Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                          loss='binary_crossentropy')
                          

#print(lapgan_training._function_kwargs)

# ----------------- Data ------------------

print("Importing cifar10")
from keras.datasets import cifar10
(xtrain, ytrain_cat), (xtest, ytest_cat) = cifar10.load_data()

print("Formatting data")
# Process the data
xtrain = xtrain.astype(np.float32) / 255.0
xtest = xtest.astype(np.float32) / 255.0

num_samples=xtrain.shape[0]

# Reverse the gaussian pyramids so that the smallest layer is first
xtrain_pyramid = list(reversed(make_gaussian_pyramid(xtrain, 3, 2)))
xtest_pyramid = list(reversed(make_gaussian_pyramid(xtest, 3, 2)))

ytrain = make_lapgan_targets(num_layers=2, num_samples=num_samples)
ytest = make_lapgan_targets(num_layers=2, num_samples=xtest_pyramid[0].shape[0])


# -------------- Callbacks -----------------

output_dir='output/cifar'

num_gen_images = 32

# Make a callback to periodically generate images after each epoch
zsamples1 = np.random.normal(size=(num_gen_images, 16*16))
zsamples2 = np.random.normal(size=(num_gen_images, 32*32))
base_imgs = xtest_pyramid[0][0:100]

def image_sampler():
    results = lapgan_generative.predict([zsamples1, zsamples2, base_imgs])
    # results contains the base input, the output after layer 1, output after layer 2
    images = [ base_imgs,
               results[0].reshape(num_gen_images, 16, 16, 3),
               results[1].reshape(num_gen_images, 32, 32, 3) ]
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
                    batch_size=64)

df = pd.DataFrame(history.history)
df.to_csv(os.path.join(output_dir, 'history.csv'))

# save models
g1.save(os.path.join(output_dir, "generator_1.h5"))
g2.save(os.path.join(output_dir, "generator_2.h5"))
d1.save(os.path.join(output_dir, "discriminator_1.h5"))
d2.save(os.path.join(output_dir, "discrimiantor_2.h5"))

