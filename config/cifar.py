try:
    import keras
    from keras.layers import Input, Reshape, Dense, Flatten, Activation, Dropout, Lambda
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.models import Sequential, Model
    import keras.regularizers
    import keras.layers

    from keras.optimizers import Adam
    from keras_adversarial import AdversarialModel, AdversarialOptimizerAlternating
    from lapgan import build_gan_layer, normal_latent_sampling

except ImportError:
    print("Disabling Keras functionality...")

import dataio

import os
import itertools

import numpy as np
import utils

TRAIN_FILES = ['data_batch_1.bin', 'data_batch_2.bin',
               'data_batch_3.bin', 'data_batch_4.bin',
               'data_batch_5.bin']
TEST_FILES = ['test_batch.bin']

PARAMS = None
PARAMS_L1 = {'data-dir': './data/cifar',
          'output-dir': './output/cifar',
          'initial-epoch': 0,
          'epochs': 300,
          'steps-per-epoch': 391, #~50000 images (782 * 64)
          'validation-steps': 1,
          'batch-size': 128,
          'use-voronoi': False}

PARAMS_L2 = {'data-dir': './data/cifar',
          'output-dir': './output/cifar',
          'initial-epoch': 0,
          'epochs': 300,
          'steps-per-epoch': 391, #~50000 images (782 * 64)
          'validation-steps': 1,
          'batch-size': 128,
          'use-voronoi': False}
    
def setup_params(layer_num):
    global PARAMS
    PARAMS = PARAMS_L1 if layer_num==0 else PARAMS_L2
    return PARAMS

# Downloads and processes data to a subdirectory in directory
# returns the training data and testing data pyramids as two lists in a tuple
def read_data():
    data_directory = PARAMS['data-dir']
    
    train_files = dataio.join_files(data_directory, TRAIN_FILES)
    test_files = dataio.join_files(data_directory, TEST_FILES)

    dataio.cond_wget_untar(data_directory,
                           train_files + test_files,
                           'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
                            moveFiles=zip(dataio.join_files('cifar-10-batches-bin', TRAIN_FILES + TEST_FILES),
                                            dataio.join_files(data_directory, TRAIN_FILES + TEST_FILES)))

    # Images are 32x32x3 bytes, with an extra byte at the start for the label
    batchSize = PARAMS['batch-size']
    entrySize = 32*32*3+1
    chunkSize = batchSize * entrySize

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

    #translations = list(itertools.product(range(-2, 3), range(-2, 3)))
    translations = [(0, 0)]; # No translation replication at all...
    def replicator(image_label_gen):
        for il in image_label_gen:
            yield (utils.repl_images_trans(il[0], translations, 'edge'), np.tile(il[1], (len(translations), 1)))

    train_images = replicator(train_images)

    def pyramid_generator(image_label):
        if PARAMS['use-voronoi']:
            import voronoi
            # x's are the (imgs, labels) tuples
            layer3 = lambda x: x[0]
            layer2 = lambda x: utils.images_resize(voronoi.vorify_batch(layer3(x), (16, 16)))
            layer1 = lambda x: utils.images_resize(voronoi.vorify_batch(layer3(x), (8, 8)))
        else:
            layer3 = lambda x: x[0]
            layer2 = lambda x: utils.images_resize(x[0], (16, 16))
            layer1 = lambda x: utils.images_resize(x[0], (8, 8))

        labels = lambda x: x[1]
            
        # Build a pyramid
        return utils.list_simultaneous_ops(image_label, [layer1, layer2, layer3, labels])

    train_pyramid = pyramid_generator(train_images)
    test_pyramid = pyramid_generator(test_images)
    sample_data = next(test_pyramid)
    # Use only 16 samples
    sample_data[0] = sample_data[0][:16]
    sample_data[1] = sample_data[1][:16]
    sample_data[2] = sample_data[2][:16]
    sample_data[3] = sample_data[3][:16]
    return (train_pyramid, test_pyramid, sample_data)

# Returns a tuple containing a training model and an evaluation model
def build_model_layer(layer_num):
    # First generator 8x8 --> 16x16
    g1 = _make_generator(0,output_shape=(16,16,3), latent_dim=16*16, num_classes=10,
                         nplanes=64, name="g1")
    # Second generator 16x16 --> 32x32
    g2 = _make_generator(1, output_shape=(32,32,3), latent_dim=32*32, num_classes=10,
                         nplanes=128, name="g2")

    d1 = _make_discriminator(0, num_classes=10, input_shape=(16,16,3), nplanes=64, name="d1")
    d2 = _make_discriminator(1, num_classes=10, input_shape=(32,32,3), nplanes=128, name="d2")

    z1 = normal_latent_sampling((16 * 16,))
    z2 = normal_latent_sampling((32 * 32,))

    zsamples1 = np.random.normal(size=(16, 16*16))
    zsamples2 = np.random.normal(size=(16, 32*32))

    zsamples = zsamples1 if layer_num==0 else zsamples2

    gen = g1 if layer_num==0 else g2
    disc = d1 if layer_num==0 else d2
    noise = z1 if layer_num==0 else z2

    model = build_gan_layer(gen, disc, noise)

    adv_model = AdversarialModel(base_model=model, player_params=[gen.trainable_weights, disc.trainable_weights],
                                    player_names=["generator", "discriminator"])

    adv_model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerAlternating(),
                                player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-4, decay=1e-4)], loss='binary_crossentropy')

    # Now make an image sampler
    def image_sampler(image_pyramid):
        base_imgs = image_pyramid[layer_num]
        gt_imgs = image_pyramid[layer_num+1]
        
        base_imgs = utils.images_resize(base_imgs, (gt_imgs[0].shape[0], gt_imgs[0].shape[1])) 
        class_conditionals = image_pyramid[3]
        num_gen_images = image_pyramid[0].shape[0]
        results = gen.predict([zsamples, base_imgs, class_conditionals], batch_size=num_gen_images) 
        # results contains the base input, the output after layer 1, output after layer 2
        images = [ ('input', base_imgs),
                   ('gt', gt_imgs),
                   ('gen', results.reshape(num_gen_images, 16 if layer_num == 0 else 32, 16 if layer_num == 0 else 32, 3))]
        return images

    def model_saver(epoch):
        # save models
        g1.save(os.path.join(PARAMS['output-dir'], "generator_1.h5"))
        g2.save(os.path.join(PARAMS['output-dir'], "generator_2.h5"))
        d1.save(os.path.join(PARAMS['output-dir'], "discriminator_1.h5"))
        d2.save(os.path.join(PARAMS['output-dir'], "discriminator_2.h5"))

    return (adv_model, image_sampler, model_saver)

def build_batches_layer(layer_num, data):
    # Pretty much here we just format the data
    # as it needs to be fed into the model
    # ([gen_fake_cond, gen_fake_class, disc_fake_class, disc_real_input, disc_real_class],
    #       [gen_fake_target..., gen_real_targets..., disc_fake_targets..., disc_real_targets...])
    def input_formatter(pyramid):
        for batch in pyramid:
            class_input = batch[3]
            cond_image_input = batch[0 if layer_num == 0 else 1]
            gen_image_target = batch[1 if layer_num == 0 else 2]
            # Upsize the input to match the generation target
            cond_image_input = utils.images_resize(cond_image_input, (gen_image_target.shape[1], gen_image_target.shape[2])) 

            diff_real = gen_image_target - cond_image_input # The real output we want our generator to produce

            ones = np.ones((batch[3].shape[0], 1))
            zeros = np.zeros((batch[3].shape[0], 1))

            gen_fake_target = ones # Generator should maximize the fake-pass rate
            gen_real_target = zeros # Doesn't really matter as the generator isn't included in the gradient
            disc_fake_target = zeros # Discriminator wants to keep out the fakes
            disc_real_target = ones # And include the reals
            yield ([cond_image_input, class_input, class_input, diff_real, class_input],
                   [gen_fake_target, gen_real_target, disc_fake_target, disc_real_target])

    return input_formatter(data)

def _make_generator(layer_num, output_shape, num_classes,
                   latent_dim, nplanes=128,
                   name="generator", reg=lambda: keras.regularizers.l1_l2(0, 0)):
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
               name='g%d_c4' % layer_num)(a)
    r = Reshape(output_shape, name="g%d_x" % layer_num)(x)

    model = Model([latent_input, conditional_input, class_input], [r], name=name)
    return model

def _make_discriminator(layer_num, input_shape, nplanes=128, num_classes=10,
                       reg=lambda: keras.regularizers.l1_l2(0, 0),
                       name="discriminator"):
    generated_input = Input(name="d%d_input" % layer_num, shape=input_shape)
    class_input = Input(name="g%d_class_input" % layer_num, shape=(num_classes,))

    # Put the class input through a dense layer
    class_dense = Dense(input_shape[0]*input_shape[1]*1,
                        kernel_regularizer=reg())(class_input)

    class_reshaped=Reshape((input_shape[0], input_shape[1], 1),
                           name="g%d_reshape_class" % layer_num)(class_dense)
    
    combined = keras.layers.concatenate([generated_input, class_reshaped])
    x = Conv2D(nplanes, (5, 5), padding='same', kernel_regularizer=reg(),
               name='d%d_c1' % layer_num)(combined)
    a = Activation('relu')(x) 
    x = Conv2D(nplanes, (5, 5), padding='same', kernel_regularizer=reg(),
               name='d%d_c2' % layer_num)(a)
    a = Activation('relu')(x) 
    
    flattened = Flatten(name='d%d_flatten' % layer_num)(a)
    a = Activation('relu')(flattened)
    dropout = Dropout(0.5)(a)
    y = Dense(1, kernel_regularizer=reg(), name='d%d_y' % layer_num)(dropout)
    
    result = Activation("sigmoid")(y)

    return Model([generated_input, class_input], [result], name=name)
