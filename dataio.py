import sys
import itertools
import os, sys, tarfile, shutil
import scipy.ndimage.interpolation
import numpy as np
import random

import tempfile

import urllib.request

import random

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

# The chunk cacher will run
# through the data once, build an in-memory numpy array
# and save that to a file
# it will then cycle through the numpy array
# To randomly index batches into the numpy array, set random_chunks
# to true
def mmapped_chunk_cacher(chunk_generator, cache_file, randomize_readback):
    # Check if the files exist
    if not file_exists(cache_file):
        # Cache the pyramid
        chunks = []
        for chunk in chunk_generator:
            chunks.append(chunk) 
            yield chunk
        a = np.array(chunks)
        np.save(cache_file, a)

    array = np.load(cache_file, mmap_mode='r')
    while True:
        for i in range(array.shape[0]):
            n = i
            if randomize_readback:
                # Select random batch
                n = random.randrange(0, array.shape[0])
            
            yield array[n]

def chunk_concat_generator(chunkGenerator, concatLen):
    chunks = []
    for c in chunkGenerator:
        chunks.append(c)
        if len(chunks) >= concatLen:
            r = np.concatenate(chunks, 0)
            yield r
            chunks = []
    yield np.concatenate(chunks, 0)

def subchunk_generator(chunkGenerator, chunkSize):
    for chunk in chunkGenerator:
        subchunks = [chunk[x:x+chunkSize] for x in range(0, len(chunk), chunkSize)]
        for c in subchunks:
            yield c

def array_chunk_generator(array, sliceSize):
    for start,end in zip(range(0,array.shape[0] - sliceSize, sliceSize),
                         range(sliceSize, array.shape[0], sliceSize)):
        yield array[start:end]
        
def files_chunk_generator(files, chunkSize, cycle=True):
    if cycle:
        files = itertools.cycle(files)

    for filename in files:
        with open(filename) as fin:
            while True:
                data = np.fromfile(fin, dtype=np.uint8, count=chunkSize)
                if data.size > 0:
                    yield data
                else:
                    break

def cond_wget_untar(dest_dir, conditional_files, wget_url, moveFiles=()):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Determine if we need to download
    if not files_exist(conditional_files):
        filename = wget_url.split('/')[-1]
        filepath = os.path.join(tempfile.gettempdir(), filename)
        # Download
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(wget_url, filepath,
                                                 reporthook=_progress)
        print()
        print('Downloaded %s, extracting...' % filename)
        tarfile.open(filepath, 'r:gz').extractall(tempfile.gettempdir())

        for src, tgt in moveFiles:
            shutil.move(os.path.join(tempfile.gettempdir(), src), tgt)

def join_files(dir, files):
    return [os.path.join(dir, f) for f in files]

def files_exist(files):
    return all([os.path.isfile(f) for f in files])

def file_exists(f):
    return os.path.isfile(f)
