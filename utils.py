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

def diff_summary(name, diff):
   return tf.summary.image(name, tf.cast(255 * (diff + 1.0) / 2.0, tf.uint8))

def filter_inputs(inputs, shapes, filter_test):
    match = filter_test(inputs)
    def real():
        return [tf.expand_dims(i,0) for i in inputs]
    def fake():
        return [tf.zeros((0,) + s, dtype=i.dtype) for i, s in zip(inputs, shapes)]

    return tf.cond(match, real, fake)


