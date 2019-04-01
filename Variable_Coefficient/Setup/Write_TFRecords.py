from __future__ import division
import numpy as np
import os
import sys

import tensorflow as tf
import numpy as np
import multiprocessing
from random import shuffle

# Import flags specifying dataset parameters
from flags import getFlags

# Creates byte feature for storing numpy integer arrays        
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Creates byte feature for storing numpy float arrays
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in value]))

# Define function for creating tfrecords file for dataset
def write_tfrecords(indices, start_file_count, file_count, data_dir="./", tfrecord_dir="./", training=True, examples_per_file=5000, use_hires=False):

    current_indices = indices[(file_count - start_file_count)*examples_per_file : (file_count-start_file_count+1)*examples_per_file]

    if use_hires:
        if training:
            filename = os.path.join(tfrecord_dir, 'hires_training-' + str(file_count) + '.tfrecords')
        else:
            filename = os.path.join(tfrecord_dir, 'hires_validation-' + str(file_count) + '.tfrecords')
    else:
        if training:
            filename = os.path.join(tfrecord_dir, 'training-' + str(file_count) + '.tfrecords')
        else:
            filename = os.path.join(tfrecord_dir, 'validation-' + str(file_count) + '.tfrecords')
            
    writer = tf.python_io.TFRecordWriter(filename)

    # Save dataset in .tfrecords file
    for i in current_indices:

        if use_hires:
            data = np.load(data_dir + "Data/hires_data_" + str(i) + ".npy").flatten().astype(np.float32)
            coeff = np.load(data_dir + "Data/hires_coeff_" + str(i) + ".npy").flatten().astype(np.float32) 
            mesh = np.load(data_dir + "Meshes/hires_mesh_" + str(i) + ".npy").astype(np.uint8)
            soln = np.load(data_dir + "Solutions/hires_solution_" + str(i) + ".npy").flatten().astype(np.float32)
        else:
            data = np.load(data_dir + "Data/data_" + str(i) + ".npy").flatten().astype(np.float32)
            coeff = np.load(data_dir + "Data/coeff_" + str(i) + ".npy").flatten().astype(np.float32)
            mesh = np.load(data_dir + "Meshes/mesh_" + str(i) + ".npy").astype(np.uint8)
            soln = np.load(data_dir + "Solutions/solution_" + str(i) + ".npy").flatten().astype(np.float32)

        # Create a feature
        feature = {'data': _floats_feature(data.tolist()),
                   'coeff': _floats_feature(coeff.tolist()),                   
                   'mesh': _bytes_feature(mesh.tostring()),
                   'soln': _floats_feature(soln.tolist())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write to file
        writer.write(example.SerializeToString())

    # Close .tfrecords writer
    writer.close()


if __name__ == '__main__':
    FLAGS = getFlags()

    # Make folder to store tfrecords files
    if not os.path.exists(FLAGS.tfrecord_dir):
        os.makedirs(FLAGS.tfrecord_dir)
        
    data_count = FLAGS.data_count * FLAGS.cov_count
    examples_per_file = 2500

    use_hires = FLAGS.use_hires
    
    def write(d):
        write_tfrecords(d[0], d[1], d[2], data_dir="./", tfrecord_dir=FLAGS.tfrecord_dir, training=d[3], examples_per_file=examples_per_file, use_hires=use_hires)

    # Shuffle data and create training and validation sets
    indices = [n for n in range(FLAGS.data_start_count,FLAGS.data_start_count + data_count)]
    shuffle(indices)
    t_indices = indices[0 : int(np.floor(0.8 * data_count))]
    v_indices = indices[int(np.floor(0.8 * data_count)) : ]

    if use_hires:
        np.save(os.path.join(FLAGS.tfrecord_dir, "hires_t_indices_" + str(FLAGS.data_start_count)), t_indices)
        np.save(os.path.join(FLAGS.tfrecord_dir, "hires_v_indices_" + str(FLAGS.data_start_count)), v_indices)
    else:
        np.save(os.path.join(FLAGS.tfrecord_dir, "t_indices_" + str(FLAGS.data_start_count)), t_indices)
        np.save(os.path.join(FLAGS.tfrecord_dir, "v_indices_" + str(FLAGS.data_start_count)), v_indices)
        
    t_count = int(np.floor(0.8 * data_count))
    v_count = data_count - t_count
    
    t_indices_list = [t_indices]*t_count
    v_indices_list = [v_indices]*v_count

    t_file_counts = [n for n in range(FLAGS.tfrecord_training_start, FLAGS.tfrecord_training_start + int(np.ceil(t_count/examples_per_file)))]
    v_file_counts = [n for n in range(FLAGS.tfrecord_validation_start, FLAGS.tfrecord_validation_start + int(np.ceil(v_count/examples_per_file)))]

    t_start_file = [FLAGS.tfrecord_training_start]*t_count
    v_start_file = [FLAGS.tfrecord_validation_start]*v_count
    
    t_training_list = [True]*t_count
    v_training_list = [False]*v_count


    # Zip writer parameters together
    t_zip = [d for d in zip(*[t_indices_list, t_start_file, t_file_counts, t_training_list])]
    v_zip = [d for d in zip(*[v_indices_list, v_start_file, v_file_counts, v_training_list])]

    # Merge training and validation parameters
    zipped_parameters = t_zip + v_zip

    
    # Create multiprocessing pool
    NumProcesses = FLAGS.cpu_count
    pool = multiprocessing.Pool(processes=NumProcesses)

    print('\n [ Writing TFRecords ]\n')
    num_tasks = int(np.ceil(t_count/examples_per_file)) + int(np.ceil(v_count/examples_per_file))
    #for i, _ in enumerate(pool.imap_unordered(write, [d for d in zipped_parameters]), 1)
    for i, _ in enumerate(pool.imap_unordered(write, zipped_parameters), 1):
        sys.stdout.write('\r  Progress:  {0:.1%}'.format(i/num_tasks))
        sys.stdout.flush()
    print('\n')

