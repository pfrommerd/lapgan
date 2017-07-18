from keras.layers import Input, Lambda, Activation
from keras.models import Model
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
class AdversarialOptimizerHingeTrain:
    def __init__(self, cutoffs):
        self.cutoffs = cutoffs

    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        funcs = []
        for loss, param, optimizer, constraint, cutoff in zip(losses, params, optimizers, constraints):
            updates = optimizer.get_updates(param, constraint, loss)
            funcs.append(K.function(inputs, [], updates=updates, **function_kwargs))
        output_func = K.function(inputs, outputs, updates=model_updates, **function_kwargs)

        def train(_inputs):
            # update each player
            for func in zip(funcs):
                func(_inputs)
                # return output
                return output_func(_inputs)
        
        return train
