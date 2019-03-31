from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import tensorflow as tf
import numpy as np
import gzip
import os
import sys
import tensorflow.contrib.slim as slim
from random import shuffle
from shutil import copyfile, copytree

# Import plotting functions from 'reader.py'
from reader import *

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
    for f in ["main.py", "Base_Model.py", "reader.py", "utils.py", "misc.py", "flags.py", "convolution_layers.py"]:
        copyfile(f, os.path.join(model_dir, f))
    if not os.path.exists(os.path.join(model_dir,"Networks")):
        copytree("Networks", os.path.join(model_dir,"Networks"))

        
# Add suffix to end of tensor name
def add_suffix(name, suffix):
    if suffix is not None:
        return name + suffix
    else:
        return name
        
# Creates byte feature for storing numpy integer arrays        
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Creates byte feature for storing numpy float arrays
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in value]))

# Define function for creating tfrecords file for dataset
def write_tfrecords(data_dir="./", alt_data_dir="./", data_count=50000, hires=True, examples_per_file=5000):

    # Check that folder contains all dataset files
    checkData(data_dir)

    # Shuffle data and create training and validation sets
    indices = [n for n in range(0,data_count)]
    shuffle(indices)
    t_indices = indices[0 : int(np.floor(0.8 * data_count))]
    v_indices = indices[int(np.floor(0.8 * data_count)) : ]

    
    # Save training dataset in .tfrecords file
    file_count = 0
    step = 0
    print("\n [ Writing Training Dataset ]\n")
    for i in t_indices:
        if (step % examples_per_file == 0) and not (step == 0):
            # Close .tfrecords writer
            writer.close()
            file_count += 1
            train_filename = os.path.join(data_dir, 'hires_training-' + str(file_count) + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(train_filename)
        elif step % examples_per_file == 0:
            file_count += 1
            train_filename = os.path.join(data_dir, 'hires_training-' + str(file_count) + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(train_filename)
            
        data = np.load(alt_data_dir + "Data/hires_data_" + str(i) + ".npy").flatten().astype(np.float32)
        soln = np.load(alt_data_dir + "Solutions/hires_solution_" + str(i) + ".npy").flatten().astype(np.float32)

        
        # Create a feature
        feature = {'data': _floats_feature(data.tolist()),
                   'soln': _floats_feature(soln.tolist())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write to file
        writer.write(example.SerializeToString())

        step += 1
        
        # Display progress
        sys.stdout.write('   Batch {0} of {1}\r'.format(step,len(t_indices)))
        sys.stdout.flush()

    # Close .tfrecords writer
    writer.close()

    # Save validation dataset in .tfrecords file
    val_filename = os.path.join(data_dir, 'hires_validation.tfrecords')
    writer = tf.python_io.TFRecordWriter(val_filename)
    file_count = 0
    step = 0
    print("\n\n [ Writing Validation Dataset ]\n")
    for i in v_indices:
        if (step % examples_per_file == 0) and not (step == 0):
            # Close .tfrecords writer
            writer.close()
            file_count += 1
            val_filename = os.path.join(data_dir, 'hires_validation-' + str(file_count) + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(val_filename)
        elif step % examples_per_file == 0:
            file_count += 1
            val_filename = os.path.join(data_dir, 'hires_validation-' + str(file_count) + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(val_filename)

        data = np.load(alt_data_dir + "Data/hires_data_" + str(i) + ".npy").flatten().astype(np.float32)
        soln = np.load(alt_data_dir + "Solutions/hires_solution_" + str(i) + ".npy").flatten().astype(np.float32)
        
        # Create a feature
        feature = {'data': _floats_feature(data.tolist()),
                   'soln': _floats_feature(soln.tolist())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write to file
        writer.write(example.SerializeToString())

        step += 1
        
        # Display progress
        sys.stdout.write('   Batch {0} of {1}\r'.format(step,len(v_indices)))
        sys.stdout.flush()

    print("\n\n")
    
    # Close .tfrecords writer            
    writer.close()


# Read one example from tfrecords file (used for debugging purposes)
def read_tfrecords(plot_count=1, data_dir="./", res=64, hires=True):
    reader = tf.TFRecordReader()
    filenames = "./hires_training-0.tfrecords"
    filename_queue = tf.train.string_input_producer([filenames])
    _, serialized_example = reader.read(filename_queue)

    feature_set = {'data': tf.FixedLenFeature([128,128,1], tf.float32),
                   'soln': tf.FixedLenFeature([128,128,1], tf.float32)}
        
    features = tf.parse_single_example( serialized_example, features= feature_set )

    data = features['data']
    soln = features['soln']

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(0,plot_count):
            data_vals, soln_vals = sess.run([data, soln])
            plot_data_vals(data_vals[:,:,0])
            plot_data_vals(soln_vals[:,:,0])
            plt.show()
            

