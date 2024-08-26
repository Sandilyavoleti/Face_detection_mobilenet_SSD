import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils

from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization

def relu6(x):
    return K.relu(x, max_value=6)

class DepthwiseConv2D(layers.Layer):
    def __init__(self, kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1,
                 activation=None, use_bias=True, depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None,
                 depthwise_constraint=None, bias_constraint=None, **kwargs):
        super(DepthwiseConv2D, self).__init__(**kwargs)
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = padding.lower()
        self.depth_multiplier = depth_multiplier
        self.activation = layers.Activation(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        channel_axis = -1
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

    def call(self, inputs):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format='channels_last'
        )

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format='channels_last')

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:3]
        new_space = []
        for i in range(2):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i]
            )
            new_space.append(new_dim)
        return (input_shape[0], new_space[0], new_space[1], input_shape[3] * self.depth_multiplier)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'depth_multiplier': self.depth_multiplier,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'depthwise_initializer': initializers.serialize(self.depthwise_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'depthwise_regularizer': regularizers.serialize(self.depthwise_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'depthwise_constraint': constraints.serialize(self.depthwise_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config
    
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = layers.BatchNormalization(axis=-1, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=-1, name='conv_pw_%d_bn' % block_id)(x)
    return layers.Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

class Scaling(layers.Layer):
    def __init__(self, init_weights=1.0, bias=True, **kwargs):
        super(Scaling, self).__init__(**kwargs)
        self.init_weights = init_weights
        self.has_bias = bias

    def build(self, input_shape):
        size = input_shape[-1]
        self.scaling_factor = self.add_weight(shape=(1, 1, size),
                                              initializer='ones',
                                              trainable=True,
                                              name='scaling_factor')

        if self.has_bias:
            self.bias = self.add_weight(shape=(1, 1, size),
                                        initializer='zeros',
                                        trainable=True,
                                        name='bias')

    def call(self, inputs):
        output = inputs * self.scaling_factor
        if self.has_bias:
            output = output + self.bias
        return output

    def get_config(self):
        config = super(Scaling, self).get_config()
        config.update({
            'init_weights': self.init_weights,
            'has_bias': self.has_bias
        })
        return config

def mn_model(image_size, n_classes, min_scale=0.1, max_scale=0.9, scales=None,
             aspect_ratios_global=None, aspect_ratios_per_layer=None,
             two_boxes_for_ar1=False, limit_boxes=True, variances=[0.1, 0.1, 0.2, 0.2],
             coords='centroids', normalize_coords=False):

    if scales is None:
        scales = np.linspace(min_scale, max_scale, 7)
    
    img_height, img_width, img_channels = image_size

    img_input = layers.Input(shape=(img_height, img_width, img_channels))

    alpha = 1.0
    depth_multiplier = 1

    x = layers.Lambda(lambda z: (z / 255.0 - 0.5) * 2.0, name='preprocess')(img_input)
    
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False, name='conv1')(x)
    x = layers.BatchNormalization(axis=-1, name='conv1_bn')(x)
    x = layers.Activation(relu6, name='conv1_relu')(x)
    
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    for i in range(7, 12):
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=i)
    
    conv4_3 = x
    
    x = _depthwise_conv_block(conv4_3, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    fc7 = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    conv6_1 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='detection_conv6_1')(fc7)
    conv6_2 = _depthwise_conv_block(conv6_1, 512, alpha, depth_multiplier, strides=(2, 2), block_id=14)
    
    conv7_1 = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='detection_conv7_1')(conv6_2)
    conv7_2 = _depthwise_conv_block(conv7_1, 256, alpha, depth_multiplier, strides=(2, 2), block_id=15)
    
    conv8_1 = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='detection_conv8_1')(conv7_2)
    conv8_2 = _depthwise_conv_block(conv8_1, 256, alpha, depth_multiplier, strides=(2, 2), block_id=16)
    
    conv9_1 = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='detection_conv9_1')(conv8_2)
    conv9_2 = _depthwise_conv_block(conv9_1, 256, alpha, depth_multiplier, strides=(2, 2), block_id=17)
    
    conv4_3_norm = L2Normalization(gamma_init=20, name='detection_conv4_3_norm')(conv4_3)
    
    # Define the confidence and localization predictors for each feature map
    def create_mbox_layers(x, n_boxes, n_classes, name):
        mbox_conf = layers.Conv2D(n_boxes * n_classes, kernel_size=(3, 3), padding='same', name=name+'_mbox_conf')(x)
        mbox_loc = layers.Conv2D(n_boxes * 4, kernel_size=(3, 3), padding='same', name=name+'_mbox_loc')(x)
        return mbox_conf, mbox_loc

    # Define the mbox layers for each feature map
    mbox_layers = []
    for feature_map, n_boxes in zip([conv4_3_norm, fc7, conv6_2, conv7_2, conv8_2, conv9_2],
                                    [4, 6, 6, 6, 4, 4]):
        conf, loc = create_mbox_layers(feature_map, n_boxes, n_classes, feature_map.name)
        mbox_layers.append((conf, loc))
    
    # Reshape predictions
    mbox_conf_reshapes = [layers.Reshape((-1, n_classes), name=conf.name+'_reshape')(conf) for conf, loc in mbox_layers]
    mbox_loc_reshapes = [layers.Reshape((-1, 4), name=loc.name+'_reshape')(loc) for conf, loc in mbox_layers]
    
    # Concatenate predictions
    mbox_conf = layers.Concatenate(axis=1, name='mbox_conf')(mbox_conf_reshapes)
    mbox_loc = layers.Concatenate(axis=1, name='mbox_loc')(mbox_loc_reshapes)
    
    mbox_conf_softmax = layers.Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    
    predictions = layers.Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc])
    
    model = models.Model(inputs=img_input, outputs=predictions)
    
    # Extract predictor sizes for external use
    predictor_sizes = np.array([K.int_shape(layer[0])[1:3] for layer in mbox_layers])
    
    # Create a dictionary of all layers by name
    model_layer = {layer.name: layer for layer in model.layers}
    
    return model, model_layer, img_input, predictor_sizes