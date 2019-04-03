import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import os
import sys
import time

import csv

cov_count = 20
data_count = 5000
batch_size = 500


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


        # Load training and validation indices
        t_indices = np.load(t_indices_file)
        v_indices = np.load(v_indices_file)
        indices = np.array([n for n in range(0, int(cov_count*data_count))])

        DATA_COUNT = int(t_indices.size + v_indices.size)
        BATCH_SIZE = 128
        BATCH_COUNT = int(np.floor(DATA_COUNT/BATCH_SIZE))
        TRAIN_BATCHES = int(np.floor(t_indices.size/BATCH_SIZE))
        VAL_BATCHES = int(np.floor(v_indices.size/BATCH_SIZE))

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

                domain_count = mesh_array[mesh_array > 0.0].size
                domain_list.append(domain_count)
                
            data_batch = np.concatenate(data_list, axis=0)
            mesh_batch = np.concatenate(mesh_list, axis=0)
            soln_batch = np.concatenate(soln_list, axis=0)
            domain_batch = np.array(domain_list)
            
            return data_batch, mesh_batch, soln_batch, domain_batch




        def eval_class(count, start_ID, end_ID, batch_size):
            print("\n")
            print(" [ COMPUTING LOSS: Class %d ]\n" %(count))

            batches = int(np.floor((end_ID-start_ID)/batch_size))
            l1_errors = np.zeros([batches])
            mse_errors = np.zeros([batches])

            class_indices = np.array([k for k in range(start_ID, end_ID)])
                            
            for n in range(0,batches):
                sys.stdout.write("    Batch %d of %d\r" %(n+1, batches))
                sys.stdout.flush()

                #batch_start = start_ID + n*batch_size
                batch_count = n
                data_batch, mesh_batch, soln_batch, domain_batch = get_batch(batch_count, batch_size, class_indices)

                # Compute network prediction
                y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                    data: data_batch,
                    mesh: mesh_batch,
                    soln: soln_batch
                })

                #l1_error = np.sum( np.sum(np.abs(y_out - soln_batch), axis=(1,2,3)) / domain_batch , axis=0)
                #mse_error = np.sum( np.sum(np.power(y_out - soln_batch, 2), axis=(1,2,3)) / domain_batch, axis=0)
                #l1_error = np.mean(np.sum(np.abs(y_out - soln_batch), axis=(1,2,3)) / domain_batch , axis=0)
                #mse_error = np.mean(np.sum(np.power(y_out - soln_batch, 2), axis=(1,2,3)) / domain_batch, axis=0)
                l1_error = np.mean(np.sum(np.abs(y_out - soln_batch), axis=(1,2,3)) / np.sum(np.abs(soln_batch), axis=(1,2,3)) , axis=0)
                mse_error = np.mean(np.sum(np.power(y_out - soln_batch, 2), axis=(1,2,3)) / np.sum(np.power(soln_batch, 2), axis=(1,2,3)), axis=0)
                l1_errors[n] = l1_error
                mse_errors[n] = mse_error

            l1_mean = np.mean(l1_errors)
            l1_std = np.std(l1_errors)
            mse_mean = np.mean(mse_errors)
            mse_std = np.std(mse_errors)
            
            return l1_mean, l1_std, mse_mean, mse_std
        

        for c in range(0,cov_count):
            start_ID = c*data_count
            end_ID = (c+1)*data_count
            l1_mean, l1_std, mse_mean, mse_std = eval_class(c, start_ID, end_ID, batch_size)

            """
            with open('Class_Losses.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                #rows = ["Class %d - L_1 Error  =  %.8f  (%.8f)" %(c+1, l1_mean, l1_std),
                #        "Class %d - MSE Error  =  %.8f  (%.8f)" %(c+1, mse_mean, mse_std)]
                #rows = ["Class %d - L_1 Relative Error  =  %.8f  (%.8f)" %(c+1, l1_mean, l1_std),
                #        "Class %d - MSE Relative Error  =  %.8f  (%.8f)" %(c+1, mse_mean, mse_std)]
                rows = ["Class %d:  L_1  %.8f  (%.8f)   MSE  %.8f  (%.8f)" %(c+1, l1_mean, l1_std, mse_mean, mse_std)]

                for row in rows:
                    #csvwriter.writerow(row)
                    csvfile.write(row+'\n')
            """
            
            #with open('class_losses.csv', 'a') as csvfile:
            with open('noprob_class_losses.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                rows = ["%d %.8f %.8f %.8f %.8f" %(c+1, l1_mean, l1_std, mse_mean, mse_std)]

                for row in rows:
                    csvfile.write(row+'\n')
                    

        print("\n")


