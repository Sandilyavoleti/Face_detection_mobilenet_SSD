import tensorflow as tf
from tensorflow.keras.layers import Layer, DepthwiseConv2D, Conv2D, BatchNormalization, Activation
from tensorflow.keras.activations import relu6

class L2Normalization(Layer):
    '''
    Performs L2 normalization on the input tensor with a learnable scaling parameter
    as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.

    Arguments:
        gamma_init (float): The initial scaling parameter. Defaults to 20 following the
            SSD paper.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `data_format = 'channels_first'`
        or `(batch, height, width, channels)` if `data_format = 'channels_last'`.

    Returns:
        The scaled tensor. Same shape as the input tensor.

    References:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    '''

    def __init__(self, gamma_init=20, axis=3, **kwargs):
        self.gamma_init = gamma_init
        self.axis = axis
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize gamma as a trainable variable
        self.gamma = self.add_weight(
            shape=(input_shape[self.axis],),
            initializer=tf.keras.initializers.Constant(self.gamma_init),
            name='{}_gamma'.format(self.name),
            trainable=True
        )
        super(L2Normalization, self).build(input_shape)

    def call(self, x):
        # Perform L2 normalization and scale by gamma
        output = tf.linalg.l2_normalize(x, axis=self.axis)
        output *= self.gamma
        return output