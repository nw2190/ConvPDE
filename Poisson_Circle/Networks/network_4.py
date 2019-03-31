import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import layers
import numpy as np
import csv
import sys
import os

import cv2

# Import MNIST loader and utility functions from 'utils.py' file
from utils import write_tfrecords, checkFolders, show_variables, add_suffix, backup_configs

# Import convolution layer definitions from 'convolution layers.py' file
from convolution_layers import conv2d_layer, inception_v3, transpose_conv2d_layer, transpose_inception_v3, dense_layer, factored_conv2d

"""
U-Net model with optional KL-divergence (pre-activation)
and probabilistic loss according to MVN prediction.
"""


# Encoder component of VAE model
def encoder(self, x, training=True, reuse=None, name=None):

    # Unpack data and mesh
    data, mesh, __ = x

    # Noise injection
    if self.use_noise_injection and training:
        interior_indices = tf.greater(mesh, 0)
        zero_tensor = tf.zeros_like(data)
        data = tf.distributions.Normal(loc=data, scale=self.noise_level*tf.ones_like(data), name='noisy_data').sample()
        data = tf.where(interior_indices, data, zero_tensor)

    # Resize data (default resolution = 128)
    if not (self.alt_res == 128):
        data = tf.image.resize_images(data, [self.alt_res, self.alt_res])

        
    # [None, 64, 64, 1]  -->  [None, 32, 32, 16]
    h1 = conv2d_layer(data, 32, kernel_size=5, batch_norm=False, regularize=self.regularize,
                      training=training, reuse=reuse, name='e_conv_1', activation=None)
    h1_kl = layers.max_pooling2d(h1, 2, 2, padding='same', data_format='channels_last', name='e_pool_1')
    h1 = tf.nn.leaky_relu(h1_kl)

    # [None, 32, 32, 16] -->  [None, 16, 16, 32]
    if self.factor:
        h2 = factored_conv2d(h1, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                             training=training, reuse=reuse, name='e_conv_2', omit_activation=True)
        h2_kl = layers.max_pooling2d(h2, 2, 2, padding='same', data_format='channels_last', name='e_pool_2')
        h2 = tf.nn.leaky_relu(h2_kl)
    else:
        h2 = conv2d_layer(h1, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                          training=training, reuse=reuse, name='e_conv_2', activation=None)
        h2_kl = layers.max_pooling2d(h2, 2, 2, padding='same', data_format='channels_last', name='e_pool_2')
        h2 = tf.nn.leaky_relu(h2_kl)

    h3_kl = inception_v3(h2, 64, stride=1, training=training, reuse=reuse, name='e_incept_1', omit_activation=True)
    h3 = tf.nn.leaky_relu(h3_kl)
    
    # [None, 16, 16, 64] --> [None, 8, 8, 64]
    h4 = conv2d_layer(h3, 64, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                      training=training, reuse=reuse, name='e_conv_3', activation=None)
    h4_kl = layers.max_pooling2d(h4, 2, 2, padding='same', data_format='channels_last', name='e_pool_3')
    h4 = tf.nn.leaky_relu(h4_kl)
    
    if self.use_inception:
        h5 = inception_v3(h4,128, batch_norm=self.use_bn, regularize=self.regularize,
                          training=training, reuse=reuse, name='e_incept_4', omit_activation=True)
        h5_kl = layers.max_pooling2d(h5, 2, 2, padding='same', data_format='channels_last', name='e_pool_4')
        h5 = tf.nn.leaky_relu(h5_kl)
    else:
        h5 = conv2d_layer(h4, 128, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                          training=training, reuse=reuse, name='e_conv_4', activation=None)#, activation=None)
        h5_kl = layers.max_pooling2d(h5, 2, 2, padding='same', data_format='channels_last', name='e_pool_4')
        h5 = tf.nn.leaky_relu(h5_kl)

    h6 = inception_v3(h5, 256, batch_norm=self.use_bn, regularize=self.regularize,
                      training=training, reuse=reuse, name='e_incept_5', omit_activation=True)
    h6_kl = layers.max_pooling2d(h6, 2, 2, padding='same', data_format='channels_last', name='e_pool_5')
    h6 = tf.nn.leaky_relu(h6_kl)

    # Assemble lists of features 'h' and pre-activation values 'h_kl'
    h = [h1, h2, h3, h4, h5, h6]
    h_kl = [h1_kl, h2_kl, h3_kl, h4_kl, h5_kl, h6_kl]
    
    return h, h_kl


# Decoder component of VAE model
def decoder(self, z, training=True, reuse=None, name=None):

    # Note: h2 and h3 have same resolution
    h1, h2, h3, h4, h5, h6 = z

    # h6 ~ [None, 4, 4, 256]
    h = h6
    h = inception_v3(h, 256, batch_norm=self.use_bn, regularize=self.regularize,
                     training=training, reuse=reuse, name='d_incept_0')

    # [None, 4, 4, 256]  -->  [None, 8, 8, 128]
    h = transpose_conv2d_layer(h, 128, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize,
                               stride=2, training=training, reuse=reuse, name='d_tconv_0')

    h = tf.concat([h, h5],3)
    
    # [None, 8, 8, 64]  -->  [None, 16, 16, 64]
    h = transpose_conv2d_layer(h, 64, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize,
                               stride=2, training=training, reuse=reuse, name='d_tconv_1')

    h = tf.concat([h, h4],3)

    # [None, 16, 16, 64] --> [None, 32, 32, 32]
    h = transpose_inception_v3(h, 32, stride=2, batch_norm=self.use_bn, regularize=self.regularize,
                               training=training, reuse=reuse, name='d_tincept_2')

    h = tf.concat([h, h3],3)
        
    # [None, 32, 32, 32] --> [None, 64, 64, 16]
    h = transpose_conv2d_layer(h, 32, kernel_size=3, batch_norm=False, regularize=self.regularize, stride=1,
                               training=training, reuse=reuse, name='d_tconv_2_1')
    h = transpose_conv2d_layer(h, 32, kernel_size=3, batch_norm=False, regularize=self.regularize, stride=2,
                               training=training, reuse=reuse, name='d_tconv_2_2')

    #h = tf.concat([h, h1],3)

    # [None, 64, 64, 16] --> [None, 128, 128, 1]
    s = transpose_conv2d_layer(h, 1, kernel_size=5, batch_norm=False, stride=2, activation=None,
                               add_bias=False, training=training, reuse=reuse, name='d_tconv_3_s')
    h = transpose_conv2d_layer(h, 1, kernel_size=5, batch_norm=False, stride=2, activation=None,
                               add_bias=False, training=training, reuse=reuse, name='d_tconv_3_m')

    # Assign name to final output
    return tf.identity(h, name=name), s
    


# Evaluate model on specified batch of data
def evaluate_model(self, data, reuse=None, training=True, suffix=None):

    # Encode input images
    z, z_kl = self.encoder(self, data, training=training, reuse=reuse, name=add_suffix("encoder", suffix))

    # Sample in latent spaces
    if self.use_kl:
        h_list = []
        l_list = []
        for h in z_kl:
            m, log_s = tf.split(h, num_or_size_splits=2, axis=3)
            h_list.append(self.sampleGaussian(m, log_s, name='latent_sample'))
            l_list.append(self.compute_kl_loss(m,log_s))
        z = h_list
        kl_loss = self.kl_wt*tf.reduce_sum(l_list)
    else:
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

