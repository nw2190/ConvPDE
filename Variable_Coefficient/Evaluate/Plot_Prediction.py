import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, convert_time, read_data, read_mesh, read_soln

import os
import time


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
    parser.add_argument("--DATA_dir", default="./Setup/", type=str, help="Folder containing dataset subdirectories")
    parser.add_argument("--default_res", default=128, type=int, help="Resolution of data")
    parser.add_argument("--ID", default=0, type=int, help="ID to plot")
    parser.add_argument("--slice_plot", default=False, action="store_true", help="Plot a slice of the prediction/solution")
    parser.add_argument("--show_error", default=False, action="store_true", help="Plot the error between the prediction and solution")
    parser.add_argument("--use_hires", default=False, action="store_true", help="Option to use high resolution data")
    parser.add_argument("--save_plots", default=False, action="store_true", help="Option to save plots to file")
    parser.add_argument("--view_elev", default=30., type=float, help="Plot elevation")
    parser.add_argument("--view_angle", default=0., type=float, help="Plot angle")
    parser.add_argument("--error_view_elev", default=30., type=float, help="Plot elevation")
    parser.add_argument("--error_view_angle", default=0., type=float, help="Plot angle")
    parser.add_argument("--force_validation", default=False, action="store_true", help="Option to skip to next validation example")
    parser.add_argument("--plot_all", default=False, action="store_true", help="Option plot all error cases")
    args = parser.parse_args()
    default_res = args.default_res
    DATA_dir = args.DATA_dir
    slice_plot = args.slice_plot
    show_error = args.show_error
    save_plots = args.save_plots
    view_elev = args.view_elev
    view_angle = args.view_angle
    error_view_elev = args.error_view_elev
    error_view_angle = args.error_view_angle
    force_validation = args.force_validation
    plot_all = args.plot_all
    graph = load_graph(args.model_dir)
    ID = args.ID
    USE_HIRES = args.use_hires
    
    # Display operators defined in graph
    #for op in graph.get_operations():
        #print(op.name)

    # Define input and output nodes
    data = graph.get_tensor_by_name('prefix/data_test:0')
    coeff = graph.get_tensor_by_name('prefix/coeff_test:0')
    mesh = graph.get_tensor_by_name('prefix/mesh_test:0')
    soln = graph.get_tensor_by_name('prefix/soln_test:0')
    y_pred = graph.get_tensor_by_name('prefix/masked_pred_test:0')
    y_scale = graph.get_tensor_by_name('prefix/masked_scale_test:0')

    t_indices = np.load(os.path.join(DATA_dir, "DATA/t_indices_0.npy"))
    v_indices = np.load(os.path.join(DATA_dir, "DATA/v_indices_0.npy"))

    if force_validation:
        def check_val(k):
            validation = np.isin(k, v_indices, assume_unique=True)
            if validation:
                return k
            else:
                return check_val(k+1)
            
        new_ID = check_val(ID)

        if new_ID == ID:
            print("\n[ VALIDATION SET ]")
        else:
            ID = new_ID
            print("\n[ VALIDATION SET ]   (ID = %d)" %(ID))
    else:
        training = np.isin(ID, t_indices, assume_unique=True)
        validation = np.isin(ID, v_indices, assume_unique=True)
        if training:
            print("\n[ TRAINING SET ]")
        elif validation:
            print("\n[ VALIDATION SET ]")
        else:
            print("\n[*] Warning: ID not found in indices.")
    
    with tf.Session(graph=graph) as sess:

        # Run initial session to remove graph loading time

        # Read mesh and data files
        source = read_data(0, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES)
        data_batch = np.expand_dims(np.transpose(source, (1, 2, 0)),0)

        stiff = read_data(0, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES, STIFF=True)
        coeff_batch = np.expand_dims(np.transpose(stiff, (1, 2, 0)),0)
        
        mesh_data = read_mesh(0, os.path.join(DATA_dir,mesh_dir), USE_HIRES=USE_HIRES)
        mesh_batch = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)

        y_data = read_soln(0, os.path.join(DATA_dir,soln_dir), USE_HIRES=USE_HIRES)
        soln_batch = np.expand_dims(np.transpose(y_data, (1, 2, 0)),0)

        # Compute network prediction
        y_out = sess.run(y_pred, feed_dict={
            data: data_batch,
            coeff: coeff_batch,
            mesh: mesh_batch,
            soln: soln_batch
        })

        
        # Read mesh and data files
        source = read_data(ID, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES)
        data_batch = np.expand_dims(np.transpose(source, (1, 2, 0)),0)

        stiff = read_data(ID, os.path.join(DATA_dir,data_dir), USE_HIRES=USE_HIRES, STIFF=True)
        coeff_batch = np.expand_dims(np.transpose(stiff, (1, 2, 0)),0)
        
        mesh_data = read_mesh(ID, os.path.join(DATA_dir,mesh_dir), USE_HIRES=USE_HIRES)
        mesh_batch = np.expand_dims(np.transpose(mesh_data, (1, 2, 0)),0)

        y_data = read_soln(ID, os.path.join(DATA_dir,soln_dir), USE_HIRES=USE_HIRES)
        soln_batch = np.expand_dims(np.transpose(y_data, (1, 2, 0)),0)

        # Compute network prediction
        start_time = time.time()
        y_out, y_s = sess.run([y_pred, y_scale], feed_dict={
            data: data_batch,
            coeff: coeff_batch,
            mesh: mesh_batch,
            soln: soln_batch
        })
        end_time = time.time()

        # Display elapsed time and plot prediction
        time_elapsed = convert_time(end_time-start_time)
        print('\nComputation Time:  '  + time_elapsed + '\n') 
        #plot_prediction(ID, y_out, soln_dir, Model=0, CV=0, Rm_Outliers=False, Filter=True, Plot_Error=False, alt_res=default_res)
        plot_prediction(ID, y_out, y_s, os.path.join(DATA_dir,soln_dir), os.path.join(DATA_dir,mesh_dir), Model=0, CV=0, Rm_Outliers=False, Filter=True, Plot_Error=show_error, default_res=default_res, slice_plot=slice_plot, use_hires=USE_HIRES, save_plots=save_plots, view_elev=view_elev, view_angle=view_angle, error_view_elev=error_view_elev, error_view_angle=error_view_angle, PLOT_ALL=plot_all)
