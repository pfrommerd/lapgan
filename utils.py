import tensorflow as tf
import os, sys, tarfile, shutil

def conv2d(filter, stride=[1, 1, 1, 1], padding='SAME', bias=None, name=None):
    if bias:
        return lambda x: tf.nn.conv2d(x, filter, stride, padding, name=name) + bias
    else:
        return lambda x: tf.nn.conv2d(x, filter, stride, padding, name=name)

def conv2d_transpose(filter, output_shape, stride=[1, 1, 1, 1], padding='SAME', bias=None, name=None):
    if bias:
        return lambda x: tf.nn.conv2d(x, filter, stride, padding, name=name) + bias
    else:
        return lambda x: tf.nn.conv2d(x, filter, stride, padding, name=name)

def dense(weights, bias=None):
    if bias:
        return lambda x: tf.matmul(x, weights) + bias
    else:
        return lambda x: tf.matmul(x, weights)

def diff_summary(name, diff, max_outputs=3):
   return tf.summary.image(name, tf.cast(255 * (diff + 1.0) / 2.0, tf.uint8), max_outputs=max_outputs)

def filter_inputs(inputs, shapes, filter_test):
    match = filter_test(inputs)
    def real():
        return [tf.expand_dims(i,0) for i in inputs]
    def fake():
        return [tf.zeros((0,) + s, dtype=i.dtype) for i, s in zip(inputs, shapes)]

    return tf.cond(match, real, fake)


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
