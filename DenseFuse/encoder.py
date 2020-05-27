import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


WEIGHT_INIT_STDDEV = 0.1
DENSE_layers = 3
DECAY = .9
EPSILON = 1e-8


class Encoder(object):
    def __init__(self, model_pre_path):
        self.weight_vars = []
        self.model_pre_path = model_pre_path

        with tf.compat.v1.variable_scope('encoder'):
            self.weight_vars.append(self._create_variables( 1, 16, 5, scope='conv1_1'))

            self.weight_vars.append(self._create_variables(16, 16, 3, scope='dense_block_conv1'))
            self.weight_vars.append(self._create_variables(32, 16, 3, scope='dense_block_conv2'))
            self.weight_vars.append(self._create_variables(48, 16, 3, scope='dense_block_conv3'))

            # self.weight_vars.append(self._create_variables(64,  8, 5, scope='conv1_2'))


    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        if self.model_pre_path:
            reader = pywrap_tensorflow.NewCheckpointReader(self.model_pre_path)
            with tf.compat.v1.variable_scope(scope):
                kernel = tf.Variable(reader.get_tensor('encoder/' + scope + '/kernel'), name='kernel')
                bias = tf.Variable(reader.get_tensor('encoder/' + scope + '/bias'), name='bias')
        else:
            with tf.compat.v1.variable_scope(scope):
                kernel = tf.Variable(tf.random.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)


    def encode(self, image):
        dense_indices = (1, 2, 3)

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i in dense_indices:
                out = conv2d_dense(out, kernel, bias, use_leaky_relu=True)
            else:
                out = conv2d(out, kernel, bias, use_leaky_relu=True)

        return out


def conv2d(x, kernel, bias, use_leaky_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.leaky_relu(out)
    # out = tf.nn.relu(out)
    return out


def conv2d_dense(x, kernel, bias, use_leaky_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.leaky_relu(out)
    # out = tf.nn.relu(out)
    # concatenate
    out = tf.concat([out, x], 3)

    return out