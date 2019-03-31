import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import os
import sys
import time

import csv

batch_size = 256
batches = 8

data_dir = "Data/"
mesh_dir = "Meshes/"
soln_dir = "Solutions/"

t_indices_file = "./Setup/DATA/t_indices_0.npy"
v_indices_file = "./Setup/DATA/v_indices_0.npy"

# Load graph from frozen .pb file
def load_graph(frozen_model_folder):
    #frozen_graph_filename = frozen_model_folder + "frozen_model.pb"
    frozen_graph_filename = frozen_model_folder + "optimized_frozen_model.pb"
    
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.Graph().as_default() as graph:

        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            producer_op_list=None
        )
    return graph


# Evaluate network on specified input data and plot prediction
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../Model/", type=str, help="Model folder to export")
    parser.add_argument("--DATA_dir", default="./Setup/", type=str, help="Folder containing dataset subdirectories")
    parser.add_argument("--default_res", default=128, type=int, help="Resolution of data")
    parser.add_argument("--ID", default=0, type=int, help="ID to plot")
    parser.add_argument("--slice_plot", default=False, action="store_true", help="Plot a slice of the prediction/solution")
    parser.add_argument("--show_error", default=False, action="store_true", help="Plot the error between the prediction and solution")
    parser.add_argument("--use_hires", default=False, action="store_true", help="Option to use high resolution data")
    args = parser.parse_args()
    default_res = args.default_res
    DATA_dir = args.DATA_dir
    slice_plot = args.slice_plot
    show_error = args.show_error
    graph = load_graph(args.model_dir)
    ID = args.ID
    USE_HIRES = args.use_hires
    
    # Display operators defined in graph
    #for op in graph.get_operations():
        #print(op.name)

    # Define input and output nodes
    data = graph.get_tensor_by_name('prefix/data_test:0')
    mesh = graph.get_tensor_by_name('prefix/mesh_test:0')
    soln = graph.get_tensor_by_name('prefix/soln_test:0')
    y_pred = graph.get_tensor_by_name('prefix/masked_pred_test:0')
    y_scale = graph.get_tensor_by_name('prefix/masked_scale_test:0')


    with tf.Session(graph=graph) as sess:

        # Run initial session to remove graph loading time

        # Read mesh and data files
        source = read_data(0, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES)
        data_batch = np.expand_dims(np.transpose(source, (1, 2, 0)),0)

        mesh_data = read_mesh(0, os.path.join(DATA_dir,mesh_dir), USE_HIRES=USE_HIRES)
        mesh_batch = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)

        y_data = read_soln(0, os.path.join(DATA_dir,soln_dir), USE_HIRES=USE_HIRES)
        soln_batch = np.expand_dims(np.transpose(y_data, (1, 2, 0)),0)

        # Compute network prediction
        y_out = sess.run(y_pred, feed_dict={
            data: data_batch,
            mesh: mesh_batch,
            soln: soln_batch
        })


        def get_batch(n, bs, indices):
            batch_indices = [k for k in range(n*bs, (n+1)*bs)]
            batch_IDs = indices[batch_indices]

            data_list = []
            mesh_list = []
            soln_list = []
            domain_list = []
            
            for ID in batch_IDs:
                # Read mesh and data files
                source = read_data(ID, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES)
                data_array = np.expand_dims(np.transpose(source, (1, 2, 0)),0)
                data_list.append(data_array)
                
                mesh_data = read_mesh(ID, os.path.join(DATA_dir,mesh_dir), USE_HIRES=USE_HIRES)
                mesh_array = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)
                mesh_list.append(mesh_array)
                
                y_data = read_soln(ID, os.path.join(DATA_dir,soln_dir), USE_HIRES=USE_HIRES)
                soln_array = np.expand_dims(np.transpose(y_data, (1, 2, 0)),0)
                soln_list.append(soln_array)

            data_batch = np.concatenate(data_list, axis=0)
            mesh_batch = np.concatenate(mesh_list, axis=0)
            soln_batch = np.concatenate(soln_list, axis=0)
            
            return data_batch, mesh_batch, soln_batch

        print("\n [ Loading Data ] \n")
        v_indices = np.load(v_indices_file)
        indices = np.array([n for n in range(0,v_indices.size)])
        data_batches = []
        mesh_batches = []
        soln_batches = []
        start = time.perf_counter()
        for n in range(0,batches):
            sys.stdout.write("   Batch %d of %d\r" %(n, batches))
            sys.stdout.flush()
            data_batch, mesh_batch, soln_batch = get_batch(n, batch_size, indices)
            data_batches.append(data_batch)
            mesh_batches.append(mesh_batch)
            soln_batches.append(soln_batch)
        end = time.perf_counter()

        load_time = end - start
        print("\n\nLoad Time: %.5f seconds" %(load_time))        
        print("\n")
        
        print("\n [ Evaluating Network ] \n")
        start = time.perf_counter()
        count = 1
        #for data_batch, mesh_batch, soln_batch in data:
        for n in range(0, batches):
            data_batch = data_batches[n]
            mesh_batch = mesh_batches[n]
            soln_batch = soln_batches[n]

            #print(data_batch.shape)
            #print(mesh_batch.shape)
            #print(soln_batch.shape)
            
            sys.stdout.write("   Batch %d of %d\r" %(count, batches))
            sys.stdout.flush()
            count += 1
            
            # Compute network prediction
            y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                data: data_batch,
                mesh: mesh_batch,
                soln: soln_batch
            })
        end = time.perf_counter()

        run_time = end - start
        batch_time = run_time / batches
        average_time = batch_time / batch_size
        print("\nTotal Time: %.5f seconds" %(run_time))
        print("\nBatch Time: %.5f seconds" %(batch_time))
        print("\nAverage Time: %.5f seconds" %(average_time))


        


