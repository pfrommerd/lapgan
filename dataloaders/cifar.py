import dataio

import numpy as np
import itertools

TRAIN_FILES = ['data_batch_1.bin', 'data_batch_2.bin',
               'data_batch_3.bin', 'data_batch_4.bin',
               'data_batch_5.bin']
TEST_FILES = ['test_batch.bin']

# Downloads and processes data to a subdirectory in directory
# returns the training data and testing data pyramids as two lists in a tuple
def build_data_pipeline(params):
    data_directory = params['data_dir']
    
    train_files = dataio.join_files(data_directory, TRAIN_FILES)
    test_files = dataio.join_files(data_directory, TEST_FILES)

    dataio.cond_wget_untar(data_directory,
                           train_files + test_files,
                           'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
                            moveFiles=zip(dataio.join_files('cifar-10-batches-bin', TRAIN_FILES + TEST_FILES),
                                            dataio.join_files(data_directory, TRAIN_FILES + TEST_FILES)))

    # Images are 32x32x3 bytes, with an extra byte at the start for the label
    batchSize = params['batch_size']
    testBatchSize = params['test_batch_size']
    entrySize = 32*32*3+1

    train_chunk_generator = dataio.files_chunk_generator(train_files, entrySize, cycle=False)
    test_chunk_generator = dataio.files_chunk_generator(test_files, testBatchSize*entrySize)

    cached_random = dataio.mmapped_chunk_cacher(train_chunk_generator, params['data_dir'] + '/cifar.npy', True)
    batched = dataio.chunk_concat_generator(cached_random, batchSize * entrySize)

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
    
    train_images = image_label_parser(batched)
    test_images = image_label_parser(test_chunk_generator)


    translations = list(itertools.product([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]))
    #translations = [(0, 0)]; # No translation replication at all...
    def replicator(image_label_gen):
        for il in image_label_gen:
            for img in dataio.repl_images_trans_gen(il[0], translations, 'edge'):
                yield (img, il[1])
    replicated = replicator(train_images)
    # Split into batches again
    train_images = dataio.chunk_concat_generator(replicated, batchSize,
                    lambda x: (np.concatenate([e[0] for e in x], 0), np.concatenate([e[1] for e in x], 0)),
                    lambda x: x[0].shape[0],
                    lambda x,s,e: (x[0][s:e], x[1][s:e]))

    def pyramid_generator(image_label):
        if params['use_voronoi']:
            import voronoi
            # x's are the (imgs, labels) tuples
            layer3 = lambda x: x[0]
            layer2 = lambda x: dataio.images_resize(voronoi.vorify_batch(layer3(x), (16, 16)))
            layer1 = lambda x: dataio.images_resize(voronoi.vorify_batch(layer3(x), (8, 8)))
        else:
            layer3 = lambda x: x[0]
            layer2 = lambda x: dataio.images_resize(x[0], (16, 16))
            layer1 = lambda x: dataio.images_resize(x[0], (8, 8))

        labels = lambda x: x[1]
            
        # Build a pyramid
        return dataio.list_simultaneous_ops(image_label, [layer1, layer2, layer3, labels])

    train_pyramid = pyramid_generator(train_images)
    test_pyramid = pyramid_generator(test_images)
    return (train_pyramid, test_pyramid)

