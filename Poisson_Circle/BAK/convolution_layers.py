from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import layers

from tensorflow.python.layers import base


BATCH_NORM = False

#ACTIVATION = tf.nn.relu
ACTIVATION = tf.nn.leaky_relu
BIAS_SHIFT = 0.01

WT_REG = tf.contrib.layers.l1_regularizer(0.025)
BI_REG = tf.contrib.layers.l1_regularizer(0.025)
#WT_REG = tf.contrib.layers.l2_regularizer(10.0)
#BI_REG = tf.contrib.layers.l2_regularizer(10.0)


class AddCoords(base.Layer):
    """Add coords to a tensor"""
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r
        
    def call(self, input_tensor):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        x_dim = tf.shape(input_tensor)[1]
        y_dim = tf.shape(input_tensor)[2]
        batch_size_tensor = tf.shape(input_tensor)[0]
        xx_ones = tf.ones([batch_size_tensor, x_dim],
                          dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0),
                           [batch_size_tensor, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)
        yy_ones = tf.ones([batch_size_tensor, y_dim],
                          dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0),
                           [batch_size_tensor, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        
        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)
        
        xx_channel = tf.cast(xx_channel, "float32") / tf.cast((x_dim - 1), "float32")
        yy_channel = tf.cast(yy_channel, "float32") / tf.cast((y_dim - 1), "float32")
        xx_channel = xx_channel*2 - 1
        yy_channel = yy_channel*2 - 1
        
        ret = tf.concat([input_tensor,
                         xx_channel,
                         yy_channel], axis=-1)
        
        if self.with_r:
            rr = tf.sqrt( tf.square(xx_channel-0.5)
                          + tf.square(yy_channel-0.5)
            )
            ret = tf.concat([ret, rr], axis=-1)
        return ret

def add_coords(x):
    addcoords = AddCoords()
    x_coords = addcoords(x)
    
    """
    N, H, W, C = x.get_shape().as_list()
    vals = tf.range(0., tf.cast(tf.shape(x)[2], tf.float32), delta=1., dtype=tf.float32)
    vals = tf.expand_dims(vals, 0)
    vals = tf.tile(vals, [tf.shape(x)[1],1])
    vals = tf.divide(vals, tf.cast(tf.shape(x)[2], tf.float32))
    vals = tf.expand_dims(vals, 0)
    vals = tf.expand_dims(vals, 3)
    vals = tf.tile(vals, [tf.shape(x)[0],1,1,1])

    t_vals = tf.range(0., tf.cast(tf.shape(x)[1], tf.float32), delta=1., dtype=tf.float32)
    t_vals = tf.expand_dims(t_vals, 0)
    t_vals = tf.tile(t_vals, [tf.shape(x)[2],1])
    t_vals = tf.divide(t_vals, tf.cast(tf.shape(x)[1], tf.float32))
    t_vals = tf.transpose(t_vals)
    t_vals = tf.expand_dims(t_vals, 0)
    t_vals = tf.expand_dims(t_vals, 3)
    t_vals = tf.tile(t_vals, [tf.shape(x)[0],1,1,1])

    coords = tf.concat([vals, t_vals], 3)
    x_coords = tf.concat([x,coords], 3)
    """
    
    return x_coords


# Define upsampling procedure
def upsample(x, new_res):
    y = tf.image.resize_images(x, [new_res, new_res], method=tf.image.ResizeMethod.BILINEAR)
    #y = tf.image.resize_images(x, [1, new_res], method=tf.image.ResizeMethod.BICUBIC)
    return y
                        
# Define Batch Normalization Layer
def batch_norm_layer(x,training,name=None,reuse=None):
    y = layers.batch_normalization(x,
                                   axis=-1,
                                   momentum=0.99,
                                   epsilon=0.001,
                                   center=True,
                                   scale=True,
                                   beta_initializer=tf.zeros_initializer(),
                                   gamma_initializer=tf.ones_initializer(),
                                   moving_mean_initializer=tf.zeros_initializer(),
                                   moving_variance_initializer=tf.ones_initializer(),
                                   beta_regularizer=None,
                                   gamma_regularizer=None,
                                   training=training,
                                   trainable=True,
                                   name=name,
                                   reuse=reuse,
                                   renorm=False,
                                   renorm_clipping=None,
                                   renorm_momentum=0.99)
    return y


# Define Convolutional Layer
def conv2d_layer(x, n_out, kernel_size, stride=1, activation=ACTIVATION, regularize=False, drop_rate=0.0, batch_norm=BATCH_NORM, training=True, name=None, reuse=None, coordconv=False):

    if coordconv:
        x = add_coords(x)

    wt_init = None
    bi_init = None
    #wt_init = tf.truncated_normal_initializer(stddev=0.1)
    #bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.01)

    if regularize:
        wt_reg = WT_REG
        bi_reg = BI_REG
    else:
        wt_reg = None
        bi_reg = None

    # Apply convolution
    y = layers.conv2d(x,
                      n_out,
                      kernel_size,
                      strides=(stride,stride),
                      padding='same',
                      data_format='channels_last',
                      dilation_rate=(1,1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=wt_init,
                      bias_initializer=bi_init,
                      kernel_regularizer=wt_reg,
                      bias_regularizer=bi_reg,
                      activity_regularizer=None,
                      trainable=True,
                      name=name,
                      reuse=reuse)

    # Apply batch normalization
    if batch_norm:
        if name:
            y = batch_norm_layer(y,training,name=name + '_bn', reuse=reuse)
        else:
            y = batch_norm_layer(y,training,name=name, reuse=reuse)

    # Apply dropout
    y = layers.dropout(y, rate=drop_rate, training=training)

    # Apply activation
    if activation is not None:
        y = activation(y)
    return y





# Define Convolution Transpose Layer
def transpose_conv2d_layer(x, n_out, kernel_size, stride=1, activation=ACTIVATION, add_bias=True, regularize=False, drop_rate=0.0, batch_norm=BATCH_NORM,training=True, name=None, reuse=None, coordconv=False):

    if coordconv:
        x = add_coords(x)
    
    wt_init = None
    bi_init = None
    #wt_init = tf.truncated_normal_initializer(stddev=0.1)
    #bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.01)

    if regularize:
        wt_reg = WT_REG
        bi_reg = BI_REG
    else:
        wt_reg = None
        bi_reg = None
        
    # Apply transpose convolution
    y = layers.conv2d_transpose(x,
                                n_out,
                                kernel_size=[kernel_size,kernel_size],
                                strides=(stride, stride),
                                padding='same',
                                data_format='channels_last',
                                activation=None,
                                use_bias=add_bias,
                                kernel_initializer=wt_init,
                                bias_initializer=bi_init,
                                kernel_regularizer=wt_reg,
                                bias_regularizer=bi_reg,
                                activity_regularizer=None,
                                trainable=True,
                                name=name,
                                reuse=reuse)

    # Apply batch normalization
    if batch_norm:
        if name:
            y = batch_norm_layer(y,training,name=name + '_bn', reuse=reuse)
        else:
            y = batch_norm_layer(y,training,name=name, reuse=reuse)

    # Apply dropout
    y = layers.dropout(y, rate=drop_rate, training=training)

    # Apply activation
    if activation is not None:
        y = activation(y)

    return y



# Define Fully Connected Layer
def dense_layer(x, n_out, activation=ACTIVATION, drop_rate=0.0, reuse=None, name=None, batch_norm=False, regularize=False, training=True):

    wt_init = None
    bi_init = None
    #wt_init = tf.truncated_normal_initializer(stddev=0.15)
    #bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.25)
    #wt_init = tf.truncated_normal_initializer(stddev=0.05)
    #bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.35)
    
    if regularize:
        wt_reg = WT_REG
        bi_reg = BI_REG
    else:
        wt_reg = None
        bi_reg = None

    # Apply dense layer
    y = layers.dense(x,
                     n_out,
                     activation=None,
                     use_bias=True,
                     kernel_initializer=wt_init,
                     bias_initializer=bi_init,
                     kernel_regularizer=wt_reg,
                     bias_regularizer=bi_reg,
                     trainable=True,
                     name=name,
                     reuse=reuse)

    # Apply batch normalization
    if batch_norm:
        if name:
            y = batch_norm_layer(y,training,name=name + '_bn', reuse=reuse)
        else:
            y = batch_norm_layer(y,training,name=name, reuse=reuse)

    # Apply dropout
    y = layers.dropout(y, rate=drop_rate, training=training)

    # Apply activation
    if activation is not None:
        y = activation(y)

    return y



# Defines Inception V3 Layer
# http://arxiv.org/abs/1512.00567
def inception_v3(x, n_out, stride=1, activation=ACTIVATION, regularize=False, drop_rate=0.0, batch_norm=BATCH_NORM, training=True, name=None, reuse=None, omit_activation=False):

    # Store name to use as prefix
    base_name = name

    final_activation = None if omit_activation else activation
    
    ###############################
    """  1x1 CONV  +  3x3 CONV  """
    ###############################
    if name:  name = base_name + '_1a'
    y1 = conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_1b'
    y1 = conv2d_layer(y1, n_out//4, 3, stride=stride, activation=final_activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ############################################
    """  1x1 CONV  +  3x3 CONV  +  3x3 CONV  """
    ############################################
    
    if name:  name = base_name + '_2a'
    y2 = conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2b'
    y2 = conv2d_layer(y2, n_out//4, 3, stride=1, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2c'
    y2 = conv2d_layer(y2, n_out//4, 3, stride=stride, activation=final_activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ###################################
    """  3x3 MAX POOL  +  1x1 CONV  """
    ###################################

    y3 = layers.max_pooling2d(x, 3, stride, padding='same', data_format='channels_last')

    if name:  name = base_name + '_3'
    y3 = conv2d_layer(y3, n_out//4, 1, stride=1, activation=final_activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ##################
    """  1x1 CONV  """
    ##################

    if name:  name = base_name + '_4'
    y4 = conv2d_layer(x, n_out//4, 1, stride=stride, activation=final_activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    y = tf.concat([y1,y2,y3,y4],3)

    return y




# Defines Inception V3 Layer Transpose
# http://arxiv.org/abs/1512.00567
def transpose_inception_v3(x, n_out, stride=1, activation=ACTIVATION, regularize=False, drop_rate=0.0, batch_norm=BATCH_NORM, training=True, name=None, reuse=None):

    # Store name to use as prefix
    base_name = name

    ###############################
    """  1x1 CONV  +  3x3 CONV  """
    ###############################
    if name:  name = base_name + '_1a'
    y1 = transpose_conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_1b'
    y1 = transpose_conv2d_layer(y1, n_out//4, 3, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ############################################
    """  1x1 CONV  +  3x3 CONV  +  3x3 CONV  """
    ############################################
    
    if name:  name = base_name + '_2a'
    y2 = transpose_conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2b'
    y2 = transpose_conv2d_layer(y2, n_out//4, 3, stride=1, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2c'
    y2 = transpose_conv2d_layer(y2, n_out//4, 3, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ###################################
    """  3x3 MAX POOL  +  1x1 CONV  """
    ###################################

    y3 = layers.max_pooling2d(x, 3, 1, padding='same', data_format='channels_last')

    if name:  name = base_name + '_3'
    y3 = transpose_conv2d_layer(y3, n_out//4, 1, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ##################
    """  1x1 CONV  """
    ##################

    if name:  name = base_name + '_4'
    y4 = transpose_conv2d_layer(x, n_out//4, 1, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    y = tf.concat([y1,y2,y3,y4],3)

    return y





# Defines ResNet Layer
def resnet(x, n_out, kernel_size, activation=ACTIVATION, regularize=False, drop_rate=0.0, batch_norm=BATCH_NORM, training=True, name=None, reuse=None):

    # Store name to use as prefix
    base_name = name

    if name:  name = base_name + '_1a'
    y = conv2d_layer(x, n_out, kernel_size, stride=1, activation=activation, regularize=regularize,
                     drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_1b'
    y = conv2d_layer(y, n_out, kernel_size, stride=1, activation=None, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    y = activation(tf.add(x,y))
    return y


# Define factored 5x5 convolution
def factored_conv2d(x, n_out, kernel_size, stride=1, activation=ACTIVATION, regularize=False, drop_rate=0.0, batch_norm=BATCH_NORM, training=True, name=None, reuse=None, omit_activation=False):
    
    final_activation = None if omit_activation else activation
    
    name = name + '_1' if (name is not None) else None
    y = conv2d_layer(x, n_out, kernel_size, stride=stride, activation=activation, regularize=regularize, drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    name = name + '_2' if (name is not None) else None
    y = conv2d_layer(y, n_out, kernel_size, stride=stride, activation=final_activation, regularize=regularize, drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    return y
