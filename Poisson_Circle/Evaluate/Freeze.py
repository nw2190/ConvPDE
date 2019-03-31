import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# Freeze model from checkpoint file
def freeze_from_checkpoint(model_dir, checkpoint_subdir, model_name):
    checkpoint_dir = os.path.join(model_dir, checkpoint_subdir)
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    input_graph_path = os.path.join(checkpoint_dir, 'graph.pbtxt')
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "masked_pred_test,masked_scale_test"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = os.path.join(model_dir, 'frozen_' + model_name + '.pb')
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    
# Optimize frozen .pb file for inference
def optimize_frozen_file(model_dir, model_name):
    frozen_graph_filename = model_dir + "frozen_model.pb"
    
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    input_node_list = ["data_test","soln_test"]
    output_node_list = ["masked_pred_test","masked_scale_test"]
    output_optimized_graph_name = os.path.join(model_dir, 'optimized_frozen_' + model_name + '.pb')
    
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        graph_def,
        input_node_list,
        output_node_list,
        tf.float32.as_datatype_enum)

    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../Model/", type=str, help="Model folder containing checkpoint directory")
    parser.add_argument("--checkpoint_subdir", default="Checkpoints/", type=str, help="Model subdirectory containing checkpoints")
    parser.add_argument("--model_name", default="model", type=str, help="Model name to use for frozen graph file")
    args = parser.parse_args()

    # Create .pb frozen file
    freeze_from_checkpoint(args.model_dir, args.checkpoint_subdir, args.model_name)

    # Optimize .pb frozen file for inference
    optimize_frozen_file(args.model_dir, args.model_name)
