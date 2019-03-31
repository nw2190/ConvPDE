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
VAE model with convolutional latent space connection
and probabilistic loss according to MVN prediction.
"""


# Encoder component of VAE model
def encoder(self, x, training=True, reuse=None, name=None):

    # Unpack data
    data, _, __ = x

    if not (self.alt_res == 128):
        data = tf.image.resize_images(data, [self.alt_res, self.alt_res])


    # 128 --> 64
    h = conv2d_layer(data, 32, kernel_size=5, batch_norm=False, regularize=self.regularize,
                     training=training, reuse=reuse, name='e_conv_1')
    h = layers.max_pooling2d(h, 2, 2, padding='same', data_format='channels_last', name='e_pool_1')

    # 64 --> 32
    if self.factor:
        h = factored_conv2d(h, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                            training=training, reuse=reuse, name='e_conv_2')
        h = layers.max_pooling2d(h, 2, 2, padding='same', data_format='channels_last', name='e_pool_2')
    else:
        h = conv2d_layer(h, 32, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_conv_2')
        h = layers.max_pooling2d(h, 2, 2, padding='same', data_format='channels_last', name='e_pool_2')

    # 32 --> 32
    h = inception_v3(h, 64, stride=1, training=training, reuse=reuse, name='e_incept_1')

    # 32 --> 16
    h = conv2d_layer(h, 64, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                     training=training, reuse=reuse, name='e_conv_3')
    h = layers.max_pooling2d(h, 2, 2, padding='same', data_format='channels_last', name='e_pool_3')

    # 16 --> 8
    if self.use_inception:
        h = inception_v3(h, 128, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_incept_4')
        h = layers.max_pooling2d(h, 2, 2, padding='same', data_format='channels_last', name='e_pool_4')
    else:
        h = conv2d_layer(h, 128, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_conv_4')#, activation=None)
        h = layers.max_pooling2d(h, 2, 2, padding='same', data_format='channels_last', name='e_pool_4')

    if self.extend:
        # 8 --> 4
        h = inception_v3(h, 256, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_incept_5')
        h = layers.max_pooling2d(h, 2, 2, padding='same', data_format='channels_last', name='e_pool_5')

        # 4 --> 4
        #h = inception_v3(h, 512, batch_norm=self.use_bn, regularize=self.regularize,
        #                 training=training, reuse=reuse, name='e_incept_6')
        h = conv2d_layer(h, 512, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_conv_5')
        # 4 --> 4
        h = conv2d_layer(h, 512, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_conv_6', activation=None)
    else:
        # 8 --> 8
        h = inception_v3(h, 256, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_incept_5')

        # 8 --> 8
        h = conv2d_layer(h, 256, kernel_size=3, batch_norm=self.use_bn, regularize=self.regularize,
                         training=training, reuse=reuse, name='e_conv_5', activation=None)
    
    # Assign names to final outputs
    mean, log_sigma = tf.split(h, num_or_size_splits=2, axis=3)
    mean = tf.identity(mean, name=name+"_mean")
    log_sigma = tf.identity(log_sigma, name=name+"_log_sigma")

    return mean, log_sigma



# Decoder component of VAE model
def decoder(self, z, training=True, reuse=None, name=None):

    # 8 --> 8
    h = inception_v3(z, 256, batch_norm=self.use_bn, regularize=self.regularize,
                     training=training, reuse=reuse, name='d_incept_0')


    # 8 --> 16
    h = transpose_conv2d_layer(h, 128, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize, stride=2,
                               training=training, reuse=reuse, name='d_tconv_0')

    # 16 --> 32
    h = transpose_conv2d_layer(h, 64, kernel_size=2, batch_norm=self.use_bn, regularize=self.regularize, stride=2,
                               training=training, reuse=reuse, name='d_tconv_1')

    # 32 --> 32
    if self.extend:
        h = transpose_inception_v3(h, 32, stride=2, batch_norm=self.use_bn, regularize=self.regularize,
                                   training=training, reuse=reuse, name='d_tincept_2')
    else:
        h = transpose_inception_v3(h, 32, stride=1, batch_norm=self.use_bn, regularize=self.regularize,
                                   training=training, reuse=reuse, name='d_tincept_2')
        
    # 32 --> 64
    h = transpose_conv2d_layer(h, 16, kernel_size=5, batch_norm=False, regularize=self.regularize, stride=2,
                               training=training, reuse=reuse, name='d_tconv_2')

    # 64 --> 128
    s = transpose_conv2d_layer(h, 1, kernel_size=5, batch_norm=False, stride=2, activation=None,
                               add_bias=False, training=training, reuse=reuse, name='d_tconv_3_s')
    h = transpose_conv2d_layer(h, 1, kernel_size=5, batch_norm=False, stride=2, activation=None,
                               add_bias=False, training=training, reuse=reuse, name='d_tconv_3_m')

    # Assign name to final output
    return tf.identity(h, name=name), s


# Evaluate model on specified batch of data
def evaluate_model(self, data, reuse=None, training=True, suffix=None):

    # Encode input images
    mean, log_sigma = self.encoder(self, data, training=training, reuse=reuse, name=add_suffix("encoder", suffix))

    # Sample latent vector
    z_sample = self.sampleGaussian(mean, log_sigma, name=add_suffix("latent_vector", suffix))

    # Decode latent vector back to original image
    pred = self.decoder(self, z_sample, training=training, reuse=reuse, name=add_suffix("pred", suffix))

    # Compute marginal likelihood loss
    masked_soln, masked_pred, masked_scale, interior_loss, boundary_loss, prob_loss = self.compute_ms_loss(data, pred, name=add_suffix("ms_loss", suffix))

    # Compute Kullback–Leibler (KL) divergence
    kl_loss = self.kl_wt*self.compute_kl_loss(mean, log_sigma, name=add_suffix("kl_loss", suffix))

    # Assign names to outputs
    masked_soln = tf.identity(masked_soln, name=add_suffix('masked_soln', suffix))
    masked_pred = tf.identity(masked_pred, name=add_suffix('masked_pred', suffix))
    masked_scale = tf.identity(masked_scale, name=add_suffix('masked_scale', suffix))
    
    return masked_soln, masked_pred, masked_scale, interior_loss, boundary_loss, kl_loss, prob_loss

