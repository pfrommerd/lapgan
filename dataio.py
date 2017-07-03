import sys
import itertools
import os, sys, tarfile
import numpy as np

import tempfile

import urllib.request

def files_chunk_generator(files, chunkSize):
    for filename in itertools.cycle(files):
        with open(filename) as fin:
            while True:
                data = np.fromfile(fin, dtype=np.uint8, count=chunkSize)
                if data.size > 0:
                    yield data
                else:
                    break


def cond_wget_untar(dest_dir, conditional_files, wget_url, renameDir=()):
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
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)

        for src, tgt in renameDir:
            os.rename(os.path.join(dest_dir, src), os.path.join(dest_dir, tgt))

def join_files(dir, files):
    return [os.path.join(dir, f) for f in files]
def files_exist(files):
    return all([os.path.isfile(f) for f in files])
