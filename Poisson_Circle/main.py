import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import layers
import numpy as np
import csv
import sys
import os

# Import parse function for tfrecords features  and EarlyStoppingHook from 'misc.py' file
from misc import _parse_data, EarlyStoppingHook, get_transformations

# Import base model for training
from Base_Model import Model

# Import Flags specifying model hyperparameters and training options
from flags import getFlags

# Initialize and train model 
def main():

    # Define model parameters and options in dictionary of flags
    FLAGS = getFlags()

    # Initialize model
    model = Model(FLAGS)
    
    # Specify number of training steps
    training_steps = FLAGS.__dict__['training_steps']

    # Define feed dictionary and loss name for EarlyStoppingHook
    loss_name = "loss_stopping:0"
    start_step = FLAGS.__dict__['early_stopping_start']
    stopping_step = FLAGS.__dict__['early_stopping_step']
    tolerance = FLAGS.__dict__['early_stopping_tol']

    # Define saver which only keeps previous 3 checkpoints (default=10)
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=3))

    # Enable GPU
    if FLAGS.__dict__['use_gpu']:
        config = tf.ConfigProto()
        config = tf.ConfigProto(device_count = {'GPU':1})
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(device_count = {'GPU':0})
        
    # Initialize TensorFlow monitored training session
    with tf.train.MonitoredTrainingSession(
            config = config,
            checkpoint_dir = os.path.join(FLAGS.__dict__['model_dir'], FLAGS.__dict__['checkpoint_dir']),
            hooks = [tf.train.StopAtStepHook(last_step=training_steps),
                     EarlyStoppingHook(loss_name, tolerance=tolerance, stopping_step=stopping_step, start_step=start_step)],
            save_summaries_steps = None, save_summaries_secs = None, save_checkpoint_secs = None,
            save_checkpoint_steps = FLAGS.__dict__['checkpoint_step'], scaffold=scaffold) as sess:

        # Set model session
        model.set_session(sess)
        
        # Train model
        model.train()

    print("\n[ TRAINING COMPLETE ]\n")

    # Create new session for model evaluation
    with tf.Session() as sess:

        # Restore network parameters from latest checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.__dict__['model_dir'], FLAGS.__dict__['checkpoint_dir'])))
            
        # Set model session using restored sess
        model.set_session(sess)

        # Initialize datasets
        model.initialize_datasets()
        
        # Reinitialize dataset handles
        model.reinitialize_handles()

        # Evaluate model
        print("[ Evaluating Model ]")
        #t_loss, v_loss, t_uq, v_uq = model.evaluate()
        t_loss, v_loss, t_uq, v_uq, t_l1, v_l1, t_l2, v_l2 = model.evaluate()

        print("\n\n[ Final Evaluations ]")
        print("Training loss: %.7f  [ UQ = %.7f ]" %(t_loss,t_uq))
        print("Validation loss: %.7f  [ UQ = %.7f ]\n" %(v_loss,v_uq))

        print(" ")
        print("Training relative loss:  %.7f [L1]    %.7f [L2]" %(t_l1,t_l2))
        print("Validation relative loss:  %.7f [L1]    %.7f [L2]\n" %(v_l1,v_l2))
        
        with open(os.path.join(FLAGS.__dict__['model_dir'], "final_losses.csv"), "w") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t_loss, v_loss])
            csvwriter.writerow([t_uq, v_uq])
            csvwriter.writerow([t_l1, v_l1])
            csvwriter.writerow([t_l2, v_l2])

# Run main() function when called directly
if __name__ == '__main__':
    main()
