import tensorflow as tf

import dataio
import itertools

TRAIN_FILES = ['data_batch_1.bin', 'data_batch_2.bin',
               'data_batch_3.bin', 'data_batch_4.bin',
               'data_batch_5.bin']
TEST_FILES = ['test_batch.bin']

# Downloads and processes data to a subdirectory in directory
# returns the training data and testing data pyramids as two lists in a tuple
def build_data_pipeline(params, test=False):
    data_directory = params['data_dir']
    
    train_files = dataio.join_files(data_directory, TRAIN_FILES)
    test_files = dataio.join_files(data_directory, TEST_FILES)

    dataio.cond_wget_untar(data_directory,
                           train_files + test_files,
                           'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
                            moveFiles=zip(dataio.join_files('cifar-10-batches-bin', TRAIN_FILES + TEST_FILES),
                                            dataio.join_files(data_directory, TRAIN_FILES + TEST_FILES)))

    # Images are 32x32x3 bytes, with an extra byte at the start for the label
    if test:
        return _file_pipeline(params, dataio.join_files(data_directory, TEST_FILES), test=True)
    else:
        return _file_pipeline(params, dataio.join_files(data_directory, TRAIN_FILES))

def _file_pipeline(params, files, test=False):

    image_bytes = 32 * 32 * 3
    label_bytes = 1
    record_bytes = image_bytes + label_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    _, value = reader.read(tf.train.string_input_producer(files))
    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.one_hot(tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32), 10)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],
                                [label_bytes + image_bytes]),
                                [3, 32, 32])

    label = tf.reshape(label, [10])
    image = tf.transpose(depth_major, [1, 2, 0])

    base_img = tf.image.resize_images(tf.image.resize_images(image, params['coarse_shape'][:2]), params['fine_shape'][:2])
    target_img = tf.image.resize_images(image, params['fine_shape'][:2])
    real_diff = target_img - base_img

    batch_base_img, batch_real_diff, batch_label = \
            tf.train.shuffle_batch([base_img, real_diff, label],
                            batch_size=params['batch_size'],
                            num_threads=params['preprocess_threads'],
                            capacity=100 * params['batch_size'], min_after_dequeue=10 * params['batch_size'])

    noise = tf.random_normal(tf.shape(batch_base_img)[:-1], stddev=params['noise'], name='noise')
    keep_prob = tf.constant(0.5)

    return {'base_img': batch_base_img, 'diff_real': batch_real_diff, 'class_cond': batch_label, 'noise': noise, 'keep_prob': keep_prob}
