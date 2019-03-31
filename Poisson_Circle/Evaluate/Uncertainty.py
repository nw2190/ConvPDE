import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import time

USE_HIRES = True
data_dir = "../Setup/Data/"
mesh_dir = "../Setup/Meshes/"
soln_dir = "../Setup/Solutions/"

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
    parser.add_argument("--default_res", default=128, type=int, help="Resolution of data")
    parser.add_argument("--ID", default=0, type=int, help="ID to plot")
    args = parser.parse_args()
    default_res = args.default_res
    graph = load_graph(args.model_dir)
    ID = args.ID
    
    # Display operators defined in graph
    #for op in graph.get_operations():
        #print(op.name)

    # Define input and output nodes
    data = graph.get_tensor_by_name('prefix/data_test:0')
    mesh = graph.get_tensor_by_name('prefix/mesh_test:0')
    soln = graph.get_tensor_by_name('prefix/soln_test:0')
    y_pred = graph.get_tensor_by_name('prefix/masked_pred_test:0')


    with tf.Session(graph=graph) as sess:

        # Run initial session to remove graph loading time

        # Read mesh and data files
        source = read_data(0, data_dir, USE_HIRES=USE_HIRES)
        data_batch = np.expand_dims(np.transpose(source, (1, 2, 0)),0)

        mesh_data = read_mesh(0, mesh_dir, USE_HIRES=USE_HIRES)
        mesh_batch = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)

        y_data = read_soln(0, soln_dir, USE_HIRES=USE_HIRES)
        soln_batch = np.expand_dims(np.transpose(y_data, (1, 2, 0)),0)

        # Compute network prediction
        y_out = sess.run(y_pred, feed_dict={
            data: data_batch,
            mesh: mesh_batch,
            soln: soln_batch
        })
        
        # Read mesh and data files
        source = read_data(ID, data_dir, USE_HIRES=USE_HIRES)
        data_batch = np.expand_dims(np.transpose(source, (1, 2, 0)),0)

        mesh_data = read_mesh(ID, mesh_dir, USE_HIRES=USE_HIRES)
        mesh_batch = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)

        y_data = read_soln(ID, soln_dir, USE_HIRES=USE_HIRES)
        soln_batch = np.expand_dims(np.transpose(y_data, (1, 2, 0)),0)

        # Compute network prediction
        start_time = time.time()

        #  Add noise and loop
        SAMPLES = 1000
        NOISE_LEVEL = 0.5
        noise_vals = np.random.normal(0.0,scale=NOISE_LEVEL, size=[SAMPLES,mesh_data.shape[1], mesh_data.shape[1], 1])

        y_out_list = []
        for n in range(0,SAMPLES):
            noise = np.array([noise_vals[n,:,:,:]])
            noise[mesh_batch == 0] = 0.0
            noise_data_batch = noise + data_batch
            y_out = sess.run(y_pred, feed_dict={
                data: noise_data_batch,
                mesh: mesh_batch,
                soln: soln_batch
            })
            y_out_list.append(y_out)
        end_time = time.time()

        print(y_out_list[0])
        y_out = np.mean(y_out_list,axis=0)
        y_stds = np.std(np.array(y_out_list), axis=0)
        print(y_stds.shape)
        
        # Display elapsed time and plot prediction
        time_elapsed = convert_time(end_time-start_time)
        print('\nComputation Time:  '  + time_elapsed + '\n') 
        #plot_prediction(ID, y_out, soln_dir, Model=0, CV=0, Rm_Outliers=False, Filter=True, Plot_Error=False, alt_res=alt_res)
        plot_prediction(ID, y_out, soln_dir, Model=0, CV=0, Rm_Outliers=False, Filter=True, Plot_Error=True, default_res=default_res, stds=y_stds)
