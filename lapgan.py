from keras.layers import Input
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
    dreal_aux_inputs = replicate_model_inputs(discriminator.inputs[1:])

    gfake_inputs = [z_sampler(discriminator.inputs[0])] # Use discriminator input size

    gfake_inputs.extend(gfake_aux_inputs)
    
    gfake_output = generator(gfake_inputs)
    print(gfake_output)
    # Add the discriminator aux inputs
    dfake_inputs = gfake_output + dfake_aux_inputs
    outputs = []

def replicate_model_inputs(inputs):
    # Should return a list of Input() layers with
    # the same dimensions as each input in inputs
    return [Input(i.shape[1:]) for i in inputs]
