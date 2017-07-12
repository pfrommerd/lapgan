import sys
import itertools
import os, sys, tarfile, shutil
import numpy as np
import random

import tempfile

import urllib.request

# TODO: Finish
def mmapped_pyramid_cacher(data, pyramid_size,
                           nmap_cache_files):
    # Check if the files exist
    if files_exist(nmap_cache_files):
        return [array_chunk_generator(np.load(file, mmap_mode='r'))
                for file in nmap_cache_files]
    else:
        # Cache the pyramid
        pass

def chunk_shuffler_generator(chunkGenerator):
    for chunk in chunkGenerator:
        random.shuffle(chunk)
        yield chunk

def list_subchunk_generator(chunkGenerator, chunkSize):
    for chunk in chunkGenerator:
        subchunks = [chunk[x:x+chunkSize] for x in range(0, len(chunk), chunkSize)]
        for c in subchunks:
            yield c

def array_chunk_generator(array, sliceSize):
    for start,end in zip(range(0,array.shape[0] - sliceSize, sliceSize),
                         range(sliceSize, array.shape[0], sliceSize)):
        yield array[start:end]
        
def files_chunk_generator(files, chunkSize):
    for filename in itertools.cycle(files):
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
