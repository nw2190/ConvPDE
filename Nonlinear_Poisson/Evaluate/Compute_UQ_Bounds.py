import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import os
import sys
import time

import csv

data_dir = "Data/"
mesh_dir = "Meshes/"
soln_dir = "Solutions/"

t_indices_file = "../Setup/DATA/t_indices_0.npy"
v_indices_file = "../Setup/DATA/v_indices_0.npy"

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
    parser.add_argument("--DATA_dir", default="../Setup/", type=str, help="Folder containing dataset subdirectories")
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

        
        ##
        # Shrink data_count for debugging
        #t_indices = t_indices[0:1500]
        #v_indices = v_indices[0:1500]
        ##

        DATA_COUNT = int(t_indices.size + v_indices.size)
        BATCH_SIZE = int(500)
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

        uq_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                              1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                              2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])

        uq_train = np.zeros([uq_levels.size, TRAIN_BATCHES])
        
        # Loop through training batches
        #uq_train = np.zeros([TRAIN_BATCHES])
        #uq2_train = np.zeros([TRAIN_BATCHES])
        #uq3_train = np.zeros([TRAIN_BATCHES]) 
        for n in range(0,TRAIN_BATCHES):

            
            sys.stdout.write("    Batch %d of %d\r" %(n+1, TRAIN_BATCHES))
            sys.stdout.flush()
            #print("    Batch %d of %d" %(n+1, TRAIN_BATCHES))

            
            data_batch, mesh_batch, soln_batch, domain_batch = get_batch(n, BATCH_SIZE, t_indices)

            # Compute network prediction
            y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                data: data_batch,
                mesh: mesh_batch,
                soln: soln_batch
            })

            errors = np.abs(y_out - soln_batch)
            mask = np.where( (mesh_batch > 0.0), 1, 0)

            for i, uq in enumerate(uq_levels):
                uq_success = np.sum(mask*np.where(errors <= uq*y_s, 1, 0), axis=(1,2,3))
                uq_train[i,n] = np.mean( uq_success / domain_batch , axis=0)
                #print(uq_train[i,n])

            """
            uq_success = np.sum(mask*np.where(errors <= y_s, 1, 0), axis=(1,2,3))
            #uq_success = np.sum(np.where( (errors <= y_s) and (mesh_batch > 0.0), 1, 0), axis=(1,2,3))
            uq_percentage = np.mean( uq_success / domain_batch , axis=0)

            uq2_success = np.sum(mask*np.where(errors <= 2.*y_s, 1, 0), axis=(1,2,3))            
            #uq2_success = np.sum(np.where(errors <= 2.*y_s, 1, 0), axis=(1,2,3))
            uq2_percentage = np.mean( uq2_success / domain_batch , axis=0)

            uq3_success = np.sum(mask*np.where(errors <= 3.*y_s, 1, 0), axis=(1,2,3))
            uq3_percentage = np.mean( uq3_success / domain_batch , axis=0)
            
            uq_train[n] = uq_percentage
            uq2_train[n] = uq2_percentage
            uq3_train[n] = uq3_percentage
            """
            #print(uq_percentage)
            #print(uq2_percentage)

        print("\n\n")
        print(" [ COMPUTING VALIDATION LOSS ]\n")

        uq_test = np.zeros([uq_levels.size, VAL_BATCHES])
        
        # Loop through validation batches
        #uq_test = np.zeros([VAL_BATCHES])
        #uq2_test = np.zeros([VAL_BATCHES])
        #uq3_test = np.zeros([VAL_BATCHES])
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

            errors = np.abs(y_out - soln_batch)
            mask = np.where( (mesh_batch > 0.0), 1, 0)

            for i, uq in enumerate(uq_levels):
                uq_success = np.sum(mask*np.where(errors <= uq*y_s, 1, 0), axis=(1,2,3))
                uq_test[i,n] = np.mean( uq_success / domain_batch , axis=0)


            """
            errors = np.abs(y_out - soln_batch)
            mask = np.where( (mesh_batch > 0.0), 1, 0)
            uq_success = np.sum(mask*np.where(errors <= y_s, 1, 0), axis=(1,2,3))
            #uq_success = np.sum(np.where(errors <= y_s, 1, 0), axis=(1,2,3))
            uq_percentage = np.mean( uq_success / domain_batch , axis=0)

            uq2_success = np.sum(mask*np.where(errors <= 2.*y_s, 1, 0), axis=(1,2,3))
            #uq2_success = np.sum(np.where(errors <= 2.*y_s, 1, 0), axis=(1,2,3))
            uq2_percentage = np.mean( uq2_success / domain_batch , axis=0)

            uq3_success = np.sum(mask*np.where(errors <= 3.*y_s, 1, 0), axis=(1,2,3))
            uq3_percentage = np.mean( uq3_success / domain_batch , axis=0)

            uq_test[n] = uq_percentage
            uq2_test[n] = uq2_percentage
            uq3_test[n] = uq3_percentage
            """
        print("\n\n")
        
        # Compute loss statistics
        t_stats = np.zeros([uq_levels.size, 2])
        v_stats = np.zeros([uq_levels.size, 2])

        for i in range(0,uq_levels.size):
            t_stats[i,0] = np.mean(uq_train[i,:])
            t_stats[i,1] = np.std(uq_train[i,:])
            v_stats[i,0] = np.mean(uq_test[i,:])
            v_stats[i,1] = np.std(uq_test[i,:])
        
        """
        t_uq_mean = np.mean(uq_train)
        t_uq_std = np.std(uq_train)
        t_uq2_mean = np.mean(uq2_train)
        t_uq3_mean = np.mean(uq3_train)
        t_uq2_std = np.std(uq2_train)
        t_uq3_std = np.std(uq3_train)

        v_uq_mean = np.mean(uq_test)
        v_uq_std = np.std(uq_test)
        v_uq2_mean = np.mean(uq2_test)
        v_uq2_std = np.std(uq2_test)
        v_uq3_mean = np.mean(uq3_test)
        v_uq3_std = np.std(uq3_test)
        """
        
        with open('UQ_Bounds_NEW.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            """
            rows = ["Uncertainty Quantification (Train):",
                    "  UQ mean  =  %.8f" %(t_uq_mean),
                    "  UQ std   =  %.8f" %(t_uq_std),
                    "  UQ2 mean  =  %.8f" %(t_uq2_mean),
                    "  UQ2 std   =  %.8f" %(t_uq2_std),
                    "  UQ3 mean  =  %.8f" %(t_uq3_mean),
                    "  UQ3 std   =  %.8f" %(t_uq3_std),
                    " ",
                    "Uncertainty Quantification (Validation):",
                    "  UQ mean  =  %.8f" %(v_uq_mean),
                    "  UQ std   =  %.8f" %(v_uq_std),
                    "  UQ2 mean  =  %.8f" %(v_uq2_mean),
                    "  UQ2 std   =  %.8f" %(v_uq2_std),
                    "  UQ3 mean  =  %.8f" %(v_uq3_mean),
                    "  UQ3 std   =  %.8f" %(v_uq3_std)]

            """

            rows = ["%.3f %.8f %.8f %.8f %.8f" %(uq_levels[i], t_stats[i,0], t_stats[i,1], v_stats[i,0], v_stats[i,1]) for i in range(0,uq_levels.size)]
            
            for row in rows:
                csvfile.write(row + "\n")
                #csvwriter.writerow(row)



