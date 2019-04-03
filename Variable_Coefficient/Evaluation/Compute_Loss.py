import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import os
import sys
import time

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


        print("\n")
        print(" [ COMPUTING TRAINING LOSS ]\n")
        
        # Loop through training batches
        l1_train = np.zeros([TRAIN_BATCHES])
        mse_train = np.zeros([TRAIN_BATCHES])
        for n in range(0,TRAIN_BATCHES):

            
            sys.stdout.write("    Batch %d of %d\r" %(n+1, TRAIN_BATCHES))
            sys.stdout.flush()
            
            data_batch, mesh_batch, soln_batch, domain_batch = get_batch(n, BATCH_SIZE, t_indices)

            # Compute network prediction
            y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                data: data_batch,
                mesh: mesh_batch,
                soln: soln_batch
            })

            #print(y_out.shape)
            #print(soln_batch.shape)
            #print(len(domain_batch))
            #print((np.sum(np.abs(y_out - soln_batch), axis=(1,2,3))).shape)
            #print((np.sum(np.abs(y_out - soln_batch), axis=(1,2,3)))/domain_batch)
            l1_error = np.sum( np.sum(np.abs(y_out - soln_batch), axis=(1,2,3)) / domain_batch , axis=0)
            mse_error = np.sum( np.sum(np.power(y_out - soln_batch, 2), axis=(1,2,3)) / domain_batch, axis=0)

            l1_train[n] = l1_error
            mse_train[n] = mse_error
            

        print("\n\n")
        print(" [ COMPUTING VALIDATION LOSS ]\n")
        
        # Loop through validation batches
        l1_val = np.zeros([VAL_BATCHES])
        mse_val = np.zeros([VAL_BATCHES])
        for n in range(0,VAL_BATCHES):
            
            sys.stdout.write("    Batch %d of %d\r" %(n+1, VAL_BATCHES))
            sys.stdout.flush()

            data_batch, mesh_batch, soln_batch, domain_batch = get_batch(n, BATCH_SIZE, v_indices)

            # Compute network prediction
            y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                data: data_batch,
                mesh: mesh_batch,
                soln: soln_batch
            })

            l1_error = np.sum( np.sum(np.abs(y_out - soln_batch), axis=(1,2,3)) / domain_batch , axis=0)
            mse_error = np.sum( np.sum(np.power(y_out - soln_batch, 2), axis=(1,2,3)) / domain_batch, axis=0)

            l1_val[n] = l1_error
            mse_val[n] = mse_error

        print("\n\n")
        
        # Compute loss statistics
        t_l1_loss_mean = np.mean(l1_train)
        t_l1_loss_std = np.std(l1_train)
        t_mse_loss_mean = np.mean(mse_train)
        t_mse_loss_std = np.std(mse_train)

        v_l1_loss_mean = np.mean(l1_val)
        v_l1_loss_std = np.std(l1_val)
        v_mse_loss_mean = np.mean(mse_val)
        v_mse_loss_std = np.std(mse_val)

        with open('Final_Losses.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            rows = ["Training Losses:",
                    "  MSE mean  =  %.8f" %(t_mse_loss_mean),
                    "  MSE std   =  %.8f" %(t_mse_loss_std),
                    "  L1 mean   =  %.8f" %(t_l1_loss_mean),
                    "  L1 std    =  %.8f" %(t_l1_loss_std),
                    " ",
                    "Validation Losses:",
                    "  MSE mean  =  %.8f" %(v_mse_loss_mean),
                    "  MSE std   =  %.8f" %(v_mse_loss_std),
                    "  L1 mean   =  %.8f" %(v_l1_loss_mean),
                    "  L1 std    =  %.8f" %(v_l1_loss_std)]
            
            for row in rows:
                csvwriter.writerow(row)



