from scipy.ndimage.filters import gaussian_filter
import numpy as np

def lambda_gen(input, func):
    for i in input:
        yield func(i)

def repl_data(input, num_tee):
    for i in input:
        for n in range(num_tee):
            yield i

# Some utils for replicating data
def repl_images_trans(batch, translations, padMode):
    replicated_images = []
    for translation in translations:
        # Take the subcrop of the images
        startX = max(-translation[0], 0)
        startY = max(-translation[1], 0)
        endX = min(batch.shape[2] - translation[0], batch.shape[2])
        endY = min(batch.shape[1] - translation[1], batch.shape[1])
        crop = batch[:,startX:endX, startY:endY,:]
        paddings = [(0, 0), (batch.shape[2] - endX, startX), (batch.shape[1] - endY, startY), (0,0)]
        replicated_images.append(np.pad(crop, paddings, padMode))
    result = np.concatenate(replicated_images)
    return result

def blur(images, sigma):
    return gaussian_filter(images, (0, sigma, sigma, 0))

def downsample(images):
    return images[:,::2,::2,:]

def blur_downsample(images, sigma):
    return downsample(blur(images,sigma))

def list_simultaneous_ops(images, layer_ops):
    for i in images:
        output = []
        for f in layer_ops:
            output.append(f(i))
        yield output

# Returns the targets for the generator and discriminator for yfake (using generator + discriminator)
# and y real (using discriminator + real input)
# Shoud return [generator_yfake0, generator_yreal0, discriminator_yfake0, discriminator_yreal0, ...]
# So this should be [1 , 0 (can be anythign really as the generator is not affected), 0, 1] repeated by the
# number of layers we have
def lapgan_targets_generator(pyramid_generator, num_player_pairs, num_layers):
    for batch in pyramid_generator:
        # Batch is a list of each layer, subtract a layer as 1 is the input
        num_samples = batch[0].shape[0] # Number of samples in a batch
        # For each player we need to give the desired
        # outputs to all layers, even if the player is not affecting
        # other layers
        generator_fake = np.ones((num_samples, 1))
        generator_real = np.zeros((num_samples, 1))
        discriminator_fake = np.zeros((num_samples, 1))
        discriminator_real = np.ones((num_samples, 1))

        # The desired outputs for all layers for a single generator and discriminator
        single_generator = [generator_fake, generator_real] * num_layers
        single_discriminator = [discriminator_fake, discriminator_real] * num_layers

        # We assume that the player order are generator/discriminator pairs
        yield (batch, (single_generator + single_discriminator) * num_player_pairs)

