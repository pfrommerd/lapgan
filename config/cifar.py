from keras.layers import Input, Reshape, Dense, Flatten, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras import regularizers

from keras.optimizers import Adam
from lapgan import build_lapgan, make_lapgan_targets, make_gaussian_pyramid

from keras_adversarial import AdversarialModel, AdversarialOptimizerSimultaneous, normal_latent_sampling

import dataio as dataio

import os

import numpy as np

def get_config_params(args):
    return {'data-dir': './data', 'batch_size': 32}
    
# Downloads and processes data to a subdirectory in directory
# returns the training data and testing data pyramids as two lists in a tuple
def read_data(params):
    data_directory = params['data-dir']
    
    files = dataio.join_files(data_directory,
                              ['cifar/data_batch_1.bin', 'cifar/data_batch_2.bin',
                               'cifar/data_batch_3.bin', 'cifar/data_batch_4.bin',
                               'cifar/data_batch_5.bin', 'cifar/test_batch.bin'])

    dataio.cond_wget_untar(data_directory,
                           files,
                           'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
                           renameDir=('cifar-10-batches-bin', 'cifar'))

    # Images are 32x32x3 bytes, with an extra byte at the start for the label
    batchSize = params['batch_size']
    chunkSize = batchSize * (32 * 32 * 3 + 1)

    train_chunk_generator = dataio.files_chunk_generator(files, chunkSize)

    def image_processor(chunks):
        for chunk in chunks:
            # Remove the label byte
            chunk = np.delete(chunk, np.arange(0, chunk.size, 32*32*3+1))
            img = np.reshape(chunk, (batchSize, 3, 32, 32))
            img = np.transpose(img, (0, 3, 2, 1))
            yield img
    
    return (image_processor(train_chunk_generator),None)

# Returns a tuple containing a training model and an evaluation model
def make_model():   
    # First generator 8x8 --> 16x16
    g1 = _make_generator(0,output_shape=(16,16,3), latent_dim=16*16, nplanes=64, name="g1")
    # Second generator 16x16 --> 32x32
    g2 = _make_generator(1, output_shape=(32,32,3), latent_dim=32*32, nplanes=128, name="g2")

    d1 = _make_discriminator(0, input_shape=(16,16,3), nplanes=64, name="d1")
    d2 = _make_discriminator(1, input_shape=(32,32,3), nplanes=128, name="d2")

    z1 = normal_latent_sampling((16 * 16,))
    z2 = normal_latent_sampling((32 * 32,))



def _make_generator(layer_num, output_shape,
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
    r = Reshape(output_shape, name="g%d_x" % layer_num)(x)

    model = Model([latent_input, conditional_input], [r], name=name)
    return model

def _make_discriminator(layer_num, input_shape, nplanes=128,
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