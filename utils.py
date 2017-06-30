from keras.layers.merge import _Merge

class Subtract(_Merge):
    """Layer that subtracts 2 inputs
    It takes 2 tensor inputs
    of the same shape and returns the difference
    between the first and the second (input1 - input2)
    """

    def _merge_function(self, inputs):
        if len(inputs) < 2:
            raise ValueError('Subtract layer needs at least 2 inputs')
        output = inputs[0] - inputs[1]
        return output


def subtract(inputs, **kwargs):
    """Functional interface to the `Add` layer.
    # Arguments
        inputs: A list of 2 input tensors
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the sum of the inputs.
    """
    return Subtract(**kwargs)(inputs)
