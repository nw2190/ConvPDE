from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import tensorflow as tf
import numpy as np
import gzip
import time
import os
import sys
import tensorflow.contrib.slim as slim
from random import shuffle
from shutil import copyfile, copytree

# Import base model for defining early stopping hook
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/session_run_hook.py
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

# Import plotting functions from 'reader.py'
#from reader import *


# Define list of transformations for data augmentation
def get_transformations(rotate=False, flip=False):
    if rotate and flip:
        #transformations = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]]
        transformations = [[0,0], [0,1], [2,0], [2,1]]
    elif rotate:
        #transformations = [[0,0], [1,0], [2,0], [3,0]]
        transformations = [[0,0], [2,0]]
        #transformations = [[0,0], [1,0]]
    elif flip:
        transformations = [[0,0], [0,1]]
    else:
        transformations = [[0,0]]
    return transformations

# Transforms 'example_proto' byte strings into decoded
# onehot label and resized image array 
def _parse_data(example_proto, res=64, transformation=None):
    features = {"data": tf.FixedLenFeature([res,res,1], tf.float32),
                #"mesh": tf.FixedLenFeature([res,res,1], tf.uint8),
                "mesh": tf.FixedLenFeature((), tf.string, default_value=""),
                "soln": tf.FixedLenFeature([res,res,1], tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)

    # Scale solutions back to [-1,1] range (approximately)
    SOLN_SCALING = 100.0

    mesh = tf.decode_raw(parsed_features["mesh"], tf.uint8)
    mesh = tf.cast(tf.reshape(mesh, [res, res, 1]), tf.float32)

    data = parsed_features["data"]
    soln = tf.multiply(parsed_features["soln"], SOLN_SCALING)

    # Apply transformation for data augmentation
    if transformation is not None:
        [rotation, flip] = transformation
        
        # Stacked data
        stacked = tf.stack([data, mesh, soln], 0)

        # Rotate data
        stacked = tf.image.rot90(stacked, k=rotation)

        # Flip data
        """
        if flip == 1:
            stacked = tf.image.flip_left_right(stacked)
        """
        true_fn = lambda: tf.image.flip_left_right(stacked)
        false_fn = lambda: stacked
        stacked = tf.cond(tf.math.equal(flip, 1), true_fn, false_fn)

        # Unstack data
        data, mesh, soln = tf.unstack(stacked)
        
    return data, mesh, soln



# Show all variables in current model
def show_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
# Create folders if they do not already exist
def checkFolders(dir_list):
    for dir in list(dir_list):
        if not os.path.exists(dir):
            os.makedirs(dir)

# Check that fulle MNIST dataset exists in specified directory
def checkData(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Specified data directory '" + data_dir + "' does not exist in filesystem.")
    elif not os.path.exists(os.path.join(data_dir,'Data')):
        raise FileNotFoundError("'Data' not found in data directory.")
    elif not os.path.exists(os.path.join(data_dir,'Meshes')):
        raise FileNotFoundError("'Meshes' not found in data directory.")
    elif not os.path.exists(os.path.join(data_dir,'Solutions')):
        raise FileNotFoundError("'Solutions' not found in data directory.")

# Copy model and flags files for logging
def backup_configs(model_dir):
    checkFolders([model_dir])
    for f in ["main.py", "base_model.py", "utils.py", "flags.py", "convolution_layers.py"]:
        copyfile(f, os.path.join(model_dir, f))
    if not os.path.exists(os.path.join(model_dir,"Networks")):
        copytree("Networks", os.path.join(model_dir,"Networks"))


# Add suffix to end of tensor name
def add_suffix(name, suffix):
    if suffix is not None:
        return name + suffix
    else:
        return name





# Define early stopping hook
class EarlyStoppingHook(session_run_hook.SessionRunHook):
    def __init__(self, loss_name, feed_dict={}, tolerance=0.01, stopping_step=50, start_step=100):
        self.loss_name = loss_name
        self.feed_dict = feed_dict
        self.tolerance = tolerance
        self.stopping_step = stopping_step
        self.start_step = start_step

    # Initialize global and internal step counts
    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use EarlyStoppingHook.")
        self._prev_step = -1
        self._step = 0

    # Evaluate early stopping loss every 1000 steps
    # (avoiding repetition when multiple run calls are made each step)
    def before_run(self, run_context):
        if (self._step % self.stopping_step == 0) and \
           (not self._step == self._prev_step) and (self._step > self.start_step):

            print("\n[ Early Stopping Check ]")
            
            # Get graph from run_context session
            graph = run_context.session.graph

            # Retrieve loss tensor from graph
            loss_tensor = graph.get_tensor_by_name(self.loss_name)

            # Populate feed dictionary with placeholders and values
            fd = {}
            for key, value in self.feed_dict.items():
                placeholder = graph.get_tensor_by_name(key)
                fd[placeholder] = value

            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor,
                                                    'loss': loss_tensor}, feed_dict=fd)
        else:
            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor})
                                                    
    # Check if current loss is below tolerance for early stopping
    def after_run(self, run_context, run_values):
        if (self._step % self.stopping_step == 0) and \
           (not self._step == self._prev_step) and (self._step > self.start_step):
            global_step = run_values.results['step']
            current_loss = run_values.results['loss']
            print("Current stopping loss  =  %.10f\n" %(current_loss))
            
            if current_loss < self.tolerance:
                print("[ Early Stopping Criterion Satisfied ]\n")
                run_context.request_stop()
            self._prev_step = global_step            
        else:
            global_step = run_values.results['step']
            self._step = global_step

