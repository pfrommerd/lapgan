from scipy.ndimage.filters import gaussian_filter
import numpy as np

def replicate_data(input, num_tee):
    for i in input:
        for n in range(num_tee):
            yield i

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
