from keras.callbacks import Callback

import tensorflow as tf

class TensorImageCallback(Callback):
    def __init__(self, image_names, num_images,
                 generator, tensorboard, cmap='gray'):
        self.generator = generator
        self.cmap = cmap
        self.tensorboard = tensorboard
        self.img_inputs = []
        
        summaries = []
        for name in image_names:
            img_input = tf.placeholder(tf.float32)
            summary = tf.summary.image(name, img_input, max_outputs=num_images)
            summaries.append(summary)
            self.img_inputs.append(img_input)

        # Merge all the summaries
        self.summary = tf.summary.merge(summaries)
        
    def on_epoch_end(self, epoch, logs={}):
        images = self.generator()

        img_inputs = dict(zip(self.img_inputs, images)) 
        summary_str = self.tensorboard.sess.run(self.summary,
                                    feed_dict=img_inputs)
        self.tensorboard.writer.add_summary(summary_str, epoch)
        self.tensorboard.writer.flush()
