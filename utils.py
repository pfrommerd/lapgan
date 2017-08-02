import tensorflow as tf

def conv2d(filter, stride=[1, 1, 1, 1], padding='SAME', bias=None, name=None):
    if bias:
        return lambda x: tf.nn.conv2d(x, filter, stride, padding, name=name) + bias
    else:
        return lambda x: tf.nn.conv2d(x, filter, stride, padding, name=name)

def dense(weights, bias=None):
    if bias:
        return lambda x: tf.matmul(x, weights) + bias
    else:
        return lambda x: tf.matmul(x, weights)
