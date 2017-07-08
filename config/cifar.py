from keras.layers import Input, Reshape, Dense, Flatten, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
import keras.regularizers
import keras.layers

from keras.optimizers import Adam
from lapgan import build_lapgan, lapgan_targets_generator

from keras_adversarial import AdversarialModel, AdversarialOptimizerSimultaneous, normal_latent_sampling

import dataio

import os

import numpy as np
import utils

TRAIN_FILES = ['data_batch_1.bin', 'data_batch_2.bin',
               'data_batch_3.bin', 'data_batch_4.bin',
               'data_batch_5.bin']
TEST_FILES = ['test_batch.bin']

def get_config_params(args):
    return {'data-dir': './data/cifar',
            'output-dir': './output/cifar',
            'initial-epoch': 0,
            'epochs': 200,
            'steps-per-epoch': 390, #~50000 images (390 * 128)
            'validation-steps': 1,
            'batch-size': 32,
            'use-voronoi': False}
    
# Downloads and processes data to a subdirectory in directory
# returns the training data and testing data pyramids as two lists in a tuple
def read_data(params):
    data_directory = params['data-dir']
    
    train_files = dataio.join_files(data_directory, TRAIN_FILES)
    test_files = dataio.join_files(data_directory, TEST_FILES)

    dataio.cond_wget_untar(data_directory,
                           train_files + test_files,
                           'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
                            moveFiles=zip(dataio.join_files('cifar-10-batches-bin', TRAIN_FILES + TEST_FILES),
                                            dataio.join_files(data_directory, TRAIN_FILES + TEST_FILES)))

    # Images are 32x32x3 bytes, with an extra byte at the start for the label
    batchSize = params['batch-size']
    chunkSize = batchSize * (32 * 32 * 3 + 1)

    train_chunk_generator = dataio.files_chunk_generator(train_files, chunkSize)
    test_chunk_generator = dataio.files_chunk_generator(test_files, chunkSize)

    def image_label_parser(chunks):
        for chunk in chunks:
            # Remove the label byte
            label_indices = chunk[::(32*32*3+1)].copy()
            # Generate a one-hot encoding of the labels
            labels = np.zeros((label_indices.shape[0], 10)) # 10 classes
            labels[np.arange(label_indices.shape[0]), label_indices] = 1 # set 1's
            
            chunk = np.delete(chunk, np.arange(0, chunk.size, 32*32*3+1))
            
            num_images = chunk.size // (32*32*3)

            imgs = np.reshape(chunk, (num_images, 3, 32, 32))

            imgs = np.transpose(imgs, (0, 2, 3, 1)).astype(np.float32) / 255.0
            yield (imgs, labels)
    
    train_images = image_label_parser(train_chunk_generator)
    test_images = image_label_parser(test_chunk_generator)

    def pyramid_generator(image_label):
        if params['use-voronoi']:
            import voronoi
            # x's are the (imgs, labels) tuples
            layer3 = lambda x: x[0] # just returns the original
            layer2 = lambda x: utils.blur_downsample(voronoi.vorify_batch(x[0], [2, 2], 2), 2)
            layer1 = lambda x: utils.blur_downsample(utils.blur_downsample(voronoi.vorify_batch(x[0], [4, 4], 2), 2), 1)
        else:
            layer3 = lambda x: x[0] # just returns the original
            layer2 = lambda x: utils.blur_downsample(x[0], 2)
            layer1 = lambda x: utils.blur_downsample(utils.blur_downsample(x[0], 2), 1)

        labels = lambda x: x[1]
            
        # Build a pyramid
        return utils.list_simultaneous_ops(image_label, [layer1, layer2, layer3, labels])

    train_pyramid = pyramid_generator(train_images)
    test_pyramid = pyramid_generator(test_images)
    sample_data = next(test_pyramid)
    return (lapgan_targets_generator(train_pyramid, 1, 2),
            lapgan_targets_generator(test_pyramid, 1, 2), sample_data)

# Returns a tuple containing a training model and an evaluation model
def build_model(params):   
    # First generator 8x8 --> 16x16
    g1 = _make_generator(0,output_shape=(16,16,3), latent_dim=16*16, num_classes=10,
                         nplanes=64, name="g1")
    # Second generator 16x16 --> 32x32
    g2 = _make_generator(1, output_shape=(32,32,3), latent_dim=32*32, num_classes=10,
                         nplanes=128, name="g2")

    d1 = _make_discriminator(0, input_shape=(16,16,3), nplanes=64, name="d1")
    d2 = _make_discriminator(1, input_shape=(32,32,3), nplanes=128, name="d2")

    z1 = normal_latent_sampling((16 * 16,))
    z2 = normal_latent_sampling((32 * 32,))

    # For the test sample images
    zsamples1 = np.random.normal(size=(params['batch-size'], 16*16))
    zsamples2 = np.random.normal(size=(params['batch-size'], 32*32))


    player_params = [g1.trainable_weights + g2.trainable_weights, d1.trainable_weights + d2.trainable_weights]
    # Player order must be generator/discriminator pairs
    # due to how make_lapgan_targets data is formatted
    player_names = ["generators", "discriminators"]

    lapgan_training, lapgan_generative = build_lapgan([g1, g2], [d1, d2],
                                                      [z1, z2], True, (8,8,3), True, 10)
    model = AdversarialModel(base_model=lapgan_training, player_params=player_params, player_names=player_names)

    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                              player_optimizers=[Adam(1e-3, decay=1e-5), Adam(1e-4, decay=1e-4)],
                              loss='binary_crossentropy')

    # Now make a model saver and an image sampler
    def image_sampler(image_pyramid):
        base_imgs = image_pyramid[0]
        gt1_imgs = image_pyramid[1]
        gt2_imgs = image_pyramid[2]
        class_conditionals = image_pyramid[3]
        num_gen_images = image_pyramid[0].shape[0]
        results = lapgan_generative.predict([zsamples1, zsamples2, base_imgs, class_conditionals])
        # results contains the base input, the output after layer 1, output after layer 2
        images = [ ('input', base_imgs),
                   ('gt1', gt1_imgs),
                   ('gt2', gt2_imgs),
                   ('gen1', results[0].reshape(num_gen_images, 16, 16, 3)),
                   ('gen2', results[1].reshape(num_gen_images, 32, 32, 3)) ]
        return images

    def model_saver(epoch):
        # save models
        g1.save(os.path.join(params['output-dir'], "generator_1.h5"))
        g2.save(os.path.join(params['output-dir'], "generator_2.h5"))
        d1.save(os.path.join(params['output-dir'], "discriminator_1.h5"))
        d2.save(os.path.join(params['output-dir'], "discriminator_2.h5"))

    return model, model_saver, image_sampler

def _make_generator(layer_num, output_shape, num_classes,
                   latent_dim, nplanes=128,
                   name="generator", reg=lambda: keras.regularizers.l1_l2(1e-5, 1e-5)):
    # The conditional_input has been upsampled already
    latent_input = Input(name="g%d_latent_input" % layer_num, shape=(latent_dim,))
    latent_reshaped = Reshape((output_shape[0], output_shape[1], 1),
                              name="g%d_reshape" % layer_num)(latent_input)
    
    conditional_input = Input(name="g%d_conditional_input" % layer_num, shape=output_shape)
    class_input = Input(name="g%d_class_input" % layer_num, shape=(num_classes,))

    # Put the class input through a dense layer
    class_dense = Dense(output_shape[0]*output_shape[1]*1,
                        kernel_regularizer=reg())(class_input)

    class_reshaped=Reshape((output_shape[0], output_shape[1], 1),
                           name="g%d_reshape_class" % layer_num)(class_dense)
    
    combined = keras.layers.concatenate([latent_reshaped, conditional_input, class_reshaped])
    x = Conv2D(nplanes, (7, 7), padding='same', kernel_regularizer=reg(),
               name='g%d_c1' % layer_num)(combined)
    a = Activation('relu')(x)
    x = Conv2D(nplanes, (7, 7), padding='same', kernel_regularizer=reg(),
               name='g%d_c2' % layer_num)(a)
    a = Activation('relu')(x)
    x = Conv2D(3, (5, 5), padding='same', kernel_regularizer=reg(),
               name='g%d_c3' % layer_num)(a)
    r = Reshape(output_shape, name="g%d_x" % layer_num)(x)

    model = Model([latent_input, conditional_input, class_input], [r], name=name)
    return model

def _make_discriminator(layer_num, input_shape, nplanes=128,
                       reg=lambda: keras.regularizers.l1_l2(1e-5, 1e-5),
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
