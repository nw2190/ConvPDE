import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import layers
import numpy as np
import csv
import sys
import os

# Import utility functions from 'utils.py' file
from utils import checkFolders, show_variables, add_suffix, backup_configs

# Import convolution layer definitions from 'convolution layers.py' file
from convolution_layers import conv2d_layer, inception_v3, transpose_conv2d_layer, transpose_inception_v3, dense_layer, factored_conv2d, upsample

"""
U-Net model with optional KL-divergence (post-activation),
decoder-only batch-normalization, optional upsampling,
and probabilistic loss according to MVN prediction.
"""


# Encoder component of VAE model
def encoder(self, x, training=True, reuse=None, name=None):

    # Unpack data
    data, coeff, mesh, __ = x

    data = tf.concat([data, coeff], axis=3)
    
    if self.use_noise_injection:
        interior_indices = tf.greater(mesh, 0)
        zero_tensor = tf.zeros_like(data)
        noisy_data = tf.distributions.Normal(loc=data, scale=self.noise_level*tf.ones_like(data), name='noisy_data').sample()
        data = tf.cond(training, lambda: noisy_data, lambda: data) 
        data = tf.where(interior_indices, data, zero_tensor)
        
    if not (self.alt_res == 128):
        data = tf.image.resize_images(data, [self.alt_res, self.alt_res])


    # [None, 64, 64, 1]  -->  [None, 32, 32, 16]
    h1 = conv2d_layer(data, 32, kernel_size=5, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='e_conv_1')#, coordconv=self.coordconv)
    h1 = layers.max_pooling2d(h1, 2, 2, padding='same', data_format='channels_last', name='e_pool_1')

    # [None, 32, 32, 16] -->  [None, 16, 16, 32]
    if self.factor:
        h2 = factored_conv2d(h1, 32, kernel_size=3, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='e_conv_2')
        h2 = layers.max_pooling2d(h2, 2, 2, padding='same', data_format='channels_last', name='e_pool_2')
    else:
        h2 = conv2d_layer(h1, 32, kernel_size=3, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='e_conv_2')#, coordconv=self.coordconv)
        h2 = layers.max_pooling2d(h2, 2, 2, padding='same', data_format='channels_last', name='e_pool_2')


    h3 = inception_v3(h2, 64, stride=1, batch_norm=False, training=training, reuse=reuse, name='e_incept_1')

    # [None, 16, 16, 64] --> [None, 8, 8, 64]
    h4 = conv2d_layer(h3, 64, kernel_size=3, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='e_conv_3')#, coordconv=self.coordconv)
    h4 = layers.max_pooling2d(h4, 2, 2, padding='same', data_format='channels_last', name='e_pool_3')

    if self.use_inception:
        h5 = inception_v3(h4,128, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='e_incept_4')
        h5 = layers.max_pooling2d(h5, 2, 2, padding='same', data_format='channels_last', name='e_pool_4')
    else:
        h5 = conv2d_layer(h4, 128, kernel_size=3, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='e_conv_4')#, coordconv=self.coordconv)#, activation=None)
        h5 = layers.max_pooling2d(h5, 2, 2, padding='same', data_format='channels_last', name='e_pool_4')

    chans = 512 if self.use_kl else 256
    omit = True if self.use_kl else False
    h6 = inception_v3(h5, chans, batch_norm=self.use_bn, regularize=self.regularize, training=training, reuse=reuse, name='e_incept_5',
                      omit_activation=omit)
    h6 = layers.max_pooling2d(h6, 2, 2, padding='same', data_format='channels_last', name='e_pool_5')

    if self.coordconv:
        h6 = conv2d_layer(h6, chans, kernel_size=2, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='e_conv_6', coordconv=self.coordconv)

        
    if not self.use_kl:
        h6 = tf.layers.dropout(h6, rate=self.dropout_rate, training=training)
    elif self.use_extra_dropout:
        h6 = tf.layers.dropout(h6, rate=self.dropout_rate, training=training)
        
    return h1, h2, h3, h4, h5, h6


# Decoder component of VAE model
def decoder(self, z, training=True, reuse=None, name=None):

    # Note: h2 and h3 have same resolution
    h1, h2, h3, h4, h5, h6 = z

    # h6 ~ [None, 4, 4, 256]
    h = h6
    h = inception_v3(h, 256, batch_norm=self.use_bn, regularize=self.regularize, training=training, reuse=reuse, name='d_incept_0')

    h = tf.layers.dropout(h, rate=self.dropout_rate, training=training)


    if self.coordconv:
        h = conv2d_layer(h, 256, kernel_size=2, batch_norm=False, regularize=self.regularize, training=training, reuse=reuse, name='d_conv_0', coordconv=self.coordconv)
        
        
    # [None, 4, 4, 256]  -->  [None, 8, 8, 128]
    stride = 1 if self.interpolate else 2
    h = transpose_conv2d_layer(h, 128, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize, stride=stride, training=training, reuse=reuse, name='d_tconv_0')#, coordconv=self.coordconv)
    if self.interpolate:
        h = upsample(h, 4*2)

    h = tf.concat([h, h5],3)
    
    # [None, 8, 8, 64]  -->  [None, 16, 16, 64]
    h = transpose_conv2d_layer(h, 64, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize, stride=stride, training=training, reuse=reuse, name='d_tconv_1')#, coordconv=self.coordconv)
    if self.interpolate:
        h = upsample(h, 4*2*2)

    h = tf.concat([h, h4],3)

    # [None, 16, 16, 64] --> [None, 32, 32, 32]
    if self.symmetric:
        h = transpose_inception_v3(h, 64, stride=stride, batch_norm=self.use_bn, regularize=self.regularize, training=training, reuse=reuse, name='d_tincept_2')
    else:
        h = transpose_inception_v3(h, 32, stride=stride, batch_norm=self.use_bn, regularize=self.regularize, training=training, reuse=reuse, name='d_tincept_2')
        
    if self.interpolate:
        h = upsample(h, 4*2*2*2)

    h = tf.concat([h, h3],3)
        
    # [None, 32, 32, 32] --> [None, 64, 64, 16]
    h_m = transpose_conv2d_layer(h, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize, stride=1, training=training, reuse=reuse, name='d_tconv_2_1')#, coordconv=self.coordconv)
    h_m = transpose_conv2d_layer(h_m, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize, stride=stride, training=training, reuse=reuse, name='d_tconv_2_2')
    h_s = transpose_conv2d_layer(h, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize, stride=1, training=training, reuse=reuse, name='d_tconv_2_1_s')#, coordconv=self.coordconv)
    h_s = transpose_conv2d_layer(h_s, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize, stride=stride, training=training, reuse=reuse, name='d_tconv_2_2_s')
    if self.interpolate:
        h_m = upsample(h_m, 4*2*2*2*2)
        h_s = upsample(h_s, 4*2*2*2*2)

    #h = tf.concat([h, h1],3)

    # [None, 64, 64, 16] --> [None, 128, 128, 1]
    s = transpose_conv2d_layer(h_s, 1, kernel_size=6, batch_norm=False, stride=2, activation=None,
                               add_bias=False, training=training, reuse=reuse, name='d_tconv_3_s')
    h = transpose_conv2d_layer(h_m, 1, kernel_size=6, batch_norm=False, stride=2, activation=None,
                               add_bias=False, training=training, reuse=reuse, name='d_tconv_3_m')

    # Assign name to final output
    return tf.identity(h, name=name), s
    


# Evaluate model on specified batch of data
def evaluate_model(self, data, reuse=None, training=True, suffix=None):

    # Encode input images
    z = self.encoder(self, data, training=training, reuse=reuse, name=add_suffix("encoder", suffix))

    # Sample in latent spaces
    if self.use_kl:
        h1, h2, h3, h4, h5, h6 = z
        m, log_s = tf.split(h6, num_or_size_splits=2, axis=3)
        h6 = self.sampleGaussian(m, log_s, name='latent_sample')
        h6 = tf.layers.dropout(h6, rate=self.dropout_rate, training=training)
        z = [h1, h2, h3, h4, h5, h6]
        #if self.reduce_noise:
        #    # Use KL divergence w.r.t. N(0, 0.1*I)
        #    # by comparing with 10*sigma ~ log(10*sigma) ~ log(10) + log(sigma)
        #    kl_loss = self.kl_wt*tf.reduce_sum([self.compute_kl_loss(m,tf.add(10.0*tf.ones_like(log_s),log_s))])
        #else:
        #    kl_loss = self.kl_wt*tf.reduce_sum([self.compute_kl_loss(m,log_s)])
        kl_loss = self.kl_wt*tf.reduce_sum([self.compute_kl_loss(m,log_s)])
    else:
        h1, h2, h3, h4, h5, h6 = z
        h6 = tf.nn.leaky_relu(h6)
        z = [h1, h2, h3, h4, h5, h6]
        # Compute Kullback–Leibler (KL) divergence
        kl_loss = self.kl_wt


    # Decode latent vector back to original image
    pred = self.decoder(self, z, training=training, reuse=reuse, name=add_suffix("pred", suffix))

    # Compute marginal likelihood loss
    masked_soln, masked_pred, masked_scale, interior_loss, boundary_loss, prob_loss = self.compute_ms_loss(data, pred, name=add_suffix("ms_loss", suffix))

    # Assign names to outputs
    masked_soln = tf.identity(masked_soln, name=add_suffix('masked_soln', suffix))
    masked_pred = tf.identity(masked_pred, name=add_suffix('masked_pred', suffix))
    masked_scale = tf.identity(masked_scale, name=add_suffix('masked_scale', suffix))

    return masked_soln, masked_pred, masked_scale, interior_loss, boundary_loss, kl_loss, prob_loss

