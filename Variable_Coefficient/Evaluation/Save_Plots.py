import argparse
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages

from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import os
import time

PLOTS = 10

USE_HIRES = True
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


# Evaluate network on specified input data and plot prediction
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../Model/", type=str, help="Model folder to export")
    parser.add_argument("--DATA_dir", default="../Setup/", type=str, help="Folder containing dataset subdirectories")
    parser.add_argument("--default_res", default=128, type=int, help="Resolution of data")
    parser.add_argument("--ID", default=0, type=int, help="ID to plot")
    args = parser.parse_args()
    default_res = args.default_res
    DATA_dir = args.DATA_dir
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
    y_scale = graph.get_tensor_by_name('prefix/masked_scale_test:0')


    with PdfPages('Predictions.pdf') as plot_file:
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
            y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                data: data_batch,
                mesh: mesh_batch,
                soln: soln_batch
            })

            #for n in range(ID, ID + PLOTS):
            for n in [123, 124, 532, 1238, 1242, 2034, 45354]:
                # Read mesh and data files
                source = read_data(n, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES)
                data_batch = np.expand_dims(np.transpose(source, (1, 2, 0)),0)

                mesh_data = read_mesh(n, os.path.join(DATA_dir,mesh_dir), USE_HIRES=USE_HIRES)
                mesh_batch = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)
                
                y_data = read_soln(n, os.path.join(DATA_dir,soln_dir), USE_HIRES=USE_HIRES)
                soln_batch = np.expand_dims(np.transpose(y_data, (1, 2, 0)),0)

                # Compute network prediction
                y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
                    data: data_batch,
                    mesh: mesh_batch,
                    soln: soln_batch
                })

                # Display elapsed time and plot prediction
                plot_prediction(n, y_out, y_s, os.path.join(DATA_dir,soln_dir), Model=0, CV=0, Rm_Outliers=False, Filter=True, Plot_Error=True, SHOW_PLOT=False, default_res=default_res)
                #plt.savefig(plot_file, format='pdf')
                plot_file.savefig(dpi=1000)

                
            
