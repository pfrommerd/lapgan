from keras.callbacks import Callback

import tensorflow as tf


class ModelSaver(Callback):
    def __init__(self, saver):
        self.saver = saver
        
    def on_epoch_end(self, epoch, logs=None):
        self.saver(epoch)

class TensorImageCallback(Callback):
    def __init__(self,
                 generator, input_data, tensorboard):
        self.generator = generator
        self.tensorboard = tensorboard
        self.input_data = input_data
        self.img_inputs = []
        self.summary = None
        

    def initialize_summary(self, names, num_images):
        if self.summary is None:
            summaries = []
            for name in names:
                img_input = tf.placeholder(tf.float32)
                summary = tf.summary.image(name, img_input, max_outputs=num_images)
                summaries.append(summary)
                self.img_inputs.append(img_input)

            # Merge all the summaries
            self.summary = tf.summary.merge(summaries)
        
    def on_epoch_end(self, epoch, logs={}):
        output = self.generator(self.input_data)

        # Initialize the inputs if they haven't been
        names, images = zip(*output)

        num_images = images[0].shape[0]

        self.initialize_summary(names, num_images)
        
        img_inputs = dict(zip(self.img_inputs, images)) 
        summary_str = self.tensorboard.sess.run(self.summary,
                                    feed_dict=img_inputs)
        self.tensorboard.writer.add_summary(summary_str, epoch)
        self.tensorboard.writer.flush()
