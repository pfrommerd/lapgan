import scipy.ndimage.interpolation
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

def images_resize(batch, size, order=1):
    zoom = (1, float(size[0])/batch.shape[1], float(size[1])/batch.shape[2], 1)
    r = np.zeros((batch.shape[0], size[0], size[1], 3), dtype=np.float32)
    scipy.ndimage.interpolation.zoom(batch, zoom, r, order=order, mode='nearest')
    return r
    
def list_simultaneous_ops(images, layer_ops):
    for i in images:
        output = []
        for f in layer_ops:
            output.append(f(i))
        yield output
