import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import os
import sys
import time
import multiprocessing
import threading
import csv

# Import flags specifying dataset parameters
from timer_flags import getFlags


DATA_COUNT = 10*20*50
#DATA_COUNT = 5000
increment = 1000
batch_size = 250
#batches = 4

MODEL_DIR = "/home/nick/Research/Poisson/Nonlinear/Model_4-32/"
SETUP_DIR = "./"

data_dir = "Data/"
mesh_dir = "Meshes/"
soln_dir = "Solutions/"

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


## Neural Network 
def network_times():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=MODEL_DIR, type=str, help="Model folder to export")
    parser.add_argument("--DATA_dir", default=SETUP_DIR, type=str, help="Folder containing dataset subdirectories")
    parser.add_argument("--default_res", default=128, type=int, help="Resolution of data")
    parser.add_argument("--ID", default=0, type=int, help="ID to plot")
    parser.add_argument("--slice_plot", default=False, action="store_true", help="Plot a slice of the prediction/solution")
    parser.add_argument("--show_error", default=False, action="store_true", help="Plot the error between the prediction and solution")
    parser.add_argument("--use_hires", default=False, action="store_true", help="Option to use high resolution data")
    parser.add_argument("--no_gpu", default=False, action="store_true", help="Specify if GPU is not being used")
    parser.add_argument("--save_solutions", default=False, action="store_true", help="Option to save solutions to file")
    parser.add_argument("--time_count", default=1, type=int, help="Time count for time tests")
    args = parser.parse_args()
    default_res = args.default_res
    DATA_dir = args.DATA_dir
    slice_plot = args.slice_plot
    show_error = args.show_error
    graph = load_graph(args.model_dir)
    ID = args.ID
    USE_HIRES = args.use_hires
    NO_GPU = args.no_gpu
    time_count = args.time_count
    save_solutions = args.save_solutions

        
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
        """
        # Read mesh and data files
        source = read_data(0, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES)
        data_batch = np.expand_dims(np.transpose(source, (1, 2, 0)),0)

        mesh_data = read_mesh(0, os.path.join(DATA_dir,mesh_dir), USE_HIRES=USE_HIRES)
        mesh_batch = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)

        # Compute network prediction
        y_out = sess.run(y_pred, feed_dict={
            data: data_batch,
            mesh: mesh_batch,
            soln: data_batch
            #soln: soln_batch
        })
        """

        batches = int(time_count*increment/batch_size)
        
        #for count, batches in enumerate([int(n*increment/batch_size) for n in range(1,11)]):
        for _ in range(0,1):

            
            #count = 0
            #batches = int(DATA_COUNT/batch_size)
            
            # Start count at 1
            #count += 1
            
            #print("\n [ Loading Data ] \n")
            #indices = np.array([n for n in range(0,DATA_COUNT)])
            indices = np.array([n for n in range(0,int(time_count*increment))])
            data_batches = []
            mesh_batches = []
            #soln_batches = []
            start = time.perf_counter()

            
            #mesh_array = np.load(mesh_dir + "Meshes.npy")
            #data_array = np.load(data_dir + "Data.npy")

            mesh_array = np.load(mesh_dir + "Meshes_0.npy")
            data_array = np.load(data_dir + "Data_0.npy")
            for n in range(1,time_count):
                tmp_mesh_array = np.load(mesh_dir + "Meshes_" + str(n) + ".npy")
                tmp_data_array = np.load(data_dir + "Data_" + str(n) + ".npy")
                mesh_array = np.concatenate([mesh_array, tmp_mesh_array], axis=0)
                data_array = np.concatenate([data_array, tmp_data_array], axis=0)
                

            mesh_batches = np.split(mesh_array, batches, axis=0)
            data_batches = np.split(data_array, batches, axis=0)

            
            """
            def load_batch(n,dlist,mlist,tinds):
                data_batch, mesh_batch = get_batch(n, batch_size, indices)
                dlist.append(data_batch)
                mlist.append(mesh_batch)
                tinds.append(n)

            remaining_batches = batches
            step = 0
            tinds = []

            # Specify number of threads for loading data
            THREADS = 8

            while remaining_batches > 0:
                sys.stdout.write("   Batch %d of %d\r" %(batches-remaining_batches+1, batches))
                sys.stdout.flush()
                THREADS = np.min([THREADS, remaining_batches])
                threadList = []
                for n in range(step,step+THREADS):
                    threadList.append(threading.Thread(target=load_batch, args=(n,data_batches,mesh_batches,tinds)))
                for t in threadList:
                    t.start()
                for t in threadList:
                    t.join()
                step += THREADS
                remaining_batches -= THREADS

            sys.stdout.write("   Batch %d of %d\r" %(batches, batches))
            sys.stdout.flush()

            permute = np.argsort(np.array(tinds)).tolist()
            data_batches = [data_batches[i] for i in permute]
            mesh_batches = [mesh_batches[i] for i in permute]
            #data_batches = np.reshape(np.array(data_batches)[permute], [-1,default_res,default_res,1])
            #mesh_batches = np.reshape(np.array(mesh_batches)[permute], [-1,default_res,default_res,1])
            """

            """
            for n in range(0,batches):
                sys.stdout.write("   Batch %d of %d\r" %(n+1, batches))
                sys.stdout.flush()
                #data_batch, mesh_batch, soln_batch = get_batch(n, batch_size, indices)
                data_batch, mesh_batch = get_batch(n, batch_size, indices)
                data_batches.append(data_batch)
                mesh_batches.append(mesh_batch)
                #soln_batches.append(soln_batch)
            """
            end = time.perf_counter()

            load_time = end - start
            #print("\n\nLoad Time: %.5f seconds" %(load_time))        

            print("\n")
            print("\n [ Evaluating Network ] \n")
            start = time.perf_counter()
            #for data_batch, mesh_batch, soln_batch in data:
            for n in range(0, batches):
                data_batch = data_batches[n]
                mesh_batch = mesh_batches[n]
                #soln_batch = soln_batches[n]

                # SCALE INPUT DATA
                #scaling_factors = np.amax(np.abs(data_batch), axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
                #data_batch = data_batch/scaling_factors

                sys.stdout.write("   Batch %d of %d\r" %(n+1, batches))
                sys.stdout.flush()

                # Compute network prediction
                y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                    data: data_batch,
                    mesh: mesh_batch,
                    soln: data_batch
                    #soln: soln_batch
                })

                # RESCALE OUTPUT DATA
                #y_out = y_out * scaling_factors
                
                if save_solutions:                    
                    batch_indices = [k for k in range(n*batch_size, (n+1)*batch_size)]
                    batch_IDs = indices[batch_indices]
                    for ID in batch_IDs:
                        filename = "./Solutions/network_solution_" + str(ID) + ".npy"
                        np.save(filename, y_out[ID - n*batch_size,:,:,0])

                
            end = time.perf_counter()


            ## TIMES WITHOUT LOADING
            #total_time = end - start
            #batch_time = total_time / batches
            #average_time = batch_time / batch_size
            #print("\nTotal Time: %.5f seconds" %(total_time))
            #print("\nBatch Time: %.5f seconds" %(batch_time))
            #print("\nAverage Time: %.5f seconds" %(average_time))


            ## TIMES INCLUDING LOADING
            ltotal_time = (end - start) + load_time
            lbatch_time = ltotal_time / batches
            laverage_time = lbatch_time / batch_size
            print("\n\n")
            print(" SOLVE TIMES:\n")
            print("\n - Total Time: %.5f seconds" %(ltotal_time))
            print(" - Batch Time: %.5f seconds" %(lbatch_time))
            print(" - Average Time: %.5f seconds\n" %(laverage_time))


            if NO_GPU:
                filename = "Network_Times_NO_GPU.csv"
            else:
                filename = "Network_Times.csv"
                
            ## Remove pre-existing file
            #if os.path.exists(filename):
            #    os.remove(filename)
            
            with open(filename, 'a') as csvfile:
                #csvfile.write("Total Time: %.5f\n" %(total_time))
                #csvfile.write("Batch Time: %.5f\n" %(batch_time))
                #csvfile.write("Average Time: %.5f\n" %(average_time))
                #csvfile.write("\nWITH LOADING:")
                #csvfile.write("Total Time: %.5f\n" %(ltotal_time))
                #csvfile.write("Batch Time: %.5f\n" %(lbatch_time))
                #csvfile.write("Average Time: %.5f\n" %(laverage_time))

                csvfile.write("%d %.7f %.7f %.7f\n" %(int((time_count)*increment), ltotal_time, lbatch_time, laverage_time))
                #csvfile.write("%d %.7f %.7f %.7f\n" %(DATA_COUNT, ltotal_time, lbatch_time, laverage_time))
            


# Evaluate network on specified input data and plot prediction
if __name__ == '__main__':
    network_times()


