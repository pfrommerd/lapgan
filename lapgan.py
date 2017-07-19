from keras.layers import Input, Lambda, Activation
from keras.models import Model
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K

# Builds the model for a single layer of a Lapgan
# The function assumes that the noise input is the
# first input to the generator
# any other inputs to the generator or discriminator
# get combined, with the generator inputs coming before the discriminator
# In other words the model takes two models:
# generator: (z, aux_inputs...) --> (generated content)
# discriminator: (input content, aux_inputs) --> (probability, aux_outputs)
# and combines them into:
#   (gen_aux_inputs, discriminator_aux_inputs_fake,
#       discriminator_real_content, discriminator_real_aux_input)
#       -->
#           (fake_probability, fake_aux_outputs, real_probability, real_aux_outputs)
def build_gan_layer(generator, discriminator, z_sampler):
    gfake_aux_inputs = replicate_model_inputs(generator.inputs[1:]) 
    dfake_aux_inputs = replicate_model_inputs(discriminator.inputs[1:])
    dreal_inputs = replicate_model_inputs(discriminator.inputs)

    gfake_inputs = [z_sampler(dreal_inputs[0])] # Use discriminator input size
    gfake_inputs.extend(gfake_aux_inputs)
    
    gfake_output = [generator(gfake_inputs)]
    # Add the discriminator aux inputs
    dfake_inputs = gfake_output + dfake_aux_inputs
    dfake_outputs = [Activation('linear', name='fake')(discriminator(dfake_inputs))] 
    dreal_outputs = [Activation('linear', name='real')(discriminator(dreal_inputs))]

    return Model(inputs=(gfake_aux_inputs + dfake_aux_inputs + dreal_inputs), outputs=(dfake_outputs + dreal_outputs))

def replicate_model_inputs(inputs):
    # Should return a list of Input() layers with
    # the same dimensions as each input in inputs
    # There is some weird bug where the shape isn't
    # properly done for inputs
    return [Input(i._keras_shape[1:]) for i in inputs]

def normal_latent_sampling(latent_shape):
    return Lambda(lambda x: K.random_normal((K.shape(x)[0],) + latent_shape),
                  output_shape=lambda x: ((x[0],) + latent_shape))

class AdversarialOptimizerWeighted:
    def __init__(self, weights):
        self.weights = weights

    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        funcs = []
        for loss, param, optimizer, constraint in zip(losses, params, optimizers, constraints):
            updates = optimizer.get_updates(param, constraint, loss)
            funcs.append(K.function(inputs, [], updates=updates, **function_kwargs))
        output_func = K.function(inputs, outputs, updates=model_updates, **function_kwargs)

        def train(_inputs):
            # update each player
            for func, w in zip(funcs, self.weights):
                for i in range(w):
                    func(_inputs)
            # return output
            return output_func(_inputs)
        
        return train

# TODO: Implement
class AdversarialOptimizerHingeTrain(Callback):
    def __init__(self, cutoffs): # Cutoffs is a list of tuples with the player index, the loss name, and the cutoff value
        self.cutoffs = cutoffs
        self.losses = [0] * len(cutoffs)

    def on_batch_end(self, batch, logs=None):
        for i, (p, c, v) in enumerate(self.cutoffs):
            self.losses[i] = logs[c]

    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        funcs = []
        for loss, param, optimizer, constraint in zip(losses, params, optimizers, constraints):
            include = False
            if include:
                updates = optimizer.get_updates(param, constraint, loss)
                funcs.append(K.function(inputs, [], updates=updates, **function_kwargs))
        output_func = K.function(inputs, outputs, updates=model_updates, **function_kwargs)

        def train(_inputs):
            # update each player
            for func in funcs:
                func(_inputs)
            # return output
            return output_func(_inputs)
        
        return train

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
