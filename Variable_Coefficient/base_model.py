import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import layers
import numpy as np
import csv
import sys
import os
import importlib

# Import utility functions from 'utils.py' file
from utils import checkFolders, show_variables, add_suffix, backup_configs, _parse_data, EarlyStoppingHook, get_transformations

# Import convolution layer definitions from 'convolution layers.py' file
from convolution_layers import conv2d_layer, inception_v3, transpose_conv2d_layer, transpose_inception_v3, dense_layer, factored_conv2d, upsample

# Import AMSGrad optimizer
from Optimizers.AMSGrad import AMSGrad

# Import SGLD optimizer
from Optimizers.SGLD import SGLD

# Import TF Probability for SGLD optimizer
#import tensorflow_probability as tfp


# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, flags):

        # Read keys/values from flags and assign to self
        for key, val in flags.__dict__.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = val
                
        # Merge subdirectory paths
        self.log_dir = os.path.join(self.model_dir, self.log_dir)
        self.plot_dir = os.path.join(self.model_dir, self.plot_dir)


        # Create tfrecords if file does not exist
        #if not os.path.exists(os.path.join(self.data_dir,'training.tfrecords')):
        #    print("\n [ Creating tfrecords files ]\n")
        #    write_tfrecords(self.data_dir)

        # Initialize datasets for training, validation, and early stopping checks
        self.initialize_datasets()
        
        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Load network methods from specified file
        network = importlib.import_module("Networks.network_" + str(self.network))

        # Assign network methods
        self.encoder = network.encoder
        self.decoder = network.decoder
        self.evaluate_model = network.evaluate_model
        
        # Build graph for network model
        self.build_model()

        # Backup configuration files for logging
        backup_configs(self.model_dir)

        # Save current configuration to file
        with open(os.path.join(self.model_dir, "config.csv"), "w") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for key, val in flags.__dict__.items():
                csvwriter.writerow([key, val])


    # Create rotated dataset
    def make_dataset(self, training=True, transformation=None):
        #filenames = 'hires_training-*.tfrecords' if training else 'hires_validation-*.tfrecords'

        if training:
            if self.data_files == 0:
                if self.use_hires:
                    filenames = 'hires_training-*.tfrecords'
                else:
                    filenames = 'training-*.tfrecords'
                files = tf.data.Dataset.list_files(os.path.join(self.data_dir, filenames), shuffle=True)
            else:
                if self.use_hires:
                    filenames = ['hires_training-' + str(n) + '.tfrecords' for n in range(0,self.data_files)]
                else:
                    filenames = ['training-' + str(n) + '.tfrecords' for n in range(0,self.data_files)]
                filenames = [os.path.join(self.data_dir, f) for f in filenames]
                files = tf.data.Dataset.from_tensor_slices(filenames)
        else:
            if self.use_hires:
                filenames = 'hires_validation-*.tfrecords'
            else:
                filenames = 'validation-*.tfrecords'
            files = tf.data.Dataset.list_files(os.path.join(self.data_dir, filenames), shuffle=True)

        def tfrecord_dataset(filename):
            buffer_size = 4 * 1024 * 1024
            return tf.data.TFRecordDataset(filename, buffer_size=buffer_size) 
            
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            tfrecord_dataset, cycle_length=8, sloppy=True)) # cycle_length = number of input datasets to interleave from in parallel
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000)) # buffer_size here is just for 'randomness' of shuffling
        #dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(15000)) # buffer_size here is just for 'randomness' of shuffling
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(lambda x: _parse_data(x,res=self.default_res,transformation=transformation),
                                          self.batch_size, num_parallel_batches=self.prefetch_count))
        #if self.use_gpu:
        #    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=self.prefetch_count))
        #else:
        #    dataset = dataset.prefetch(self.prefetch_count)
        dataset = dataset.prefetch(self.prefetch_count) # prefetch is defined in terms of 'elements' of current dataset (i.e. batches)
        dataset = dataset.make_one_shot_iterator()
        return dataset
    
    # Initialize datasets
    def initialize_datasets(self):

        # Define prefetch count
        #self.prefetch_count = 2
        #self.prefetch_count = 4
        #self.prefetch_count = 8
        self.prefetch_count = 50
        #self.prefetch_count = 100
        
        # Specify which transformations to use for data augmentation
        self.transformations = get_transformations(self.rotate, self.flip)

        # Define iterators for training datasets
        self.training_datasets = [self.make_dataset(transformation=t) for t in self.transformations]

        # Define iterators for validation datasets
        self.validation_datasets = [self.make_dataset(training=False,transformation=t) for t in self.transformations]
        
        # Create early stopping batch from validation dataset
        if self.use_hires:
            filenames = 'hires_validation-*.tfrecords'
        else:
            filenames = 'validation-*.tfrecords'
        efiles = tf.data.Dataset.list_files(os.path.join(self.data_dir, filenames))
        self.edataset = tf.data.TFRecordDataset(efiles)
        self.edataset = self.edataset.map(lambda x: _parse_data(x,res=self.default_res))
        self.edataset = self.edataset.apply(tf.contrib.data.shuffle_and_repeat(self.stopping_size))
        self.edataset = self.edataset.batch(self.stopping_size)
        self.edataset = self.edataset.make_one_shot_iterator()

    # Specify session for model evaluations
    def set_session(self, sess):
        self.sess = sess

    # Reinitialize handles for datasets when restoring from checkpoint
    def reinitialize_handles(self):
        self.training_handles = self.sess.run([d.string_handle() for d in self.training_datasets])
        self.validation_handles = self.sess.run([d.string_handle() for d in self.validation_datasets])

    # Sample from multivariate Gaussian
    def sampleGaussian(self, mean, log_sigma, name=None):
        epsilon = tf.random_normal(tf.shape(log_sigma))
        return tf.identity(mean + epsilon * tf.exp(log_sigma), name=name)

    # Define sampler for generating self.z values
    def sample_z(self, batch_size):
        return np.random.normal(size=(batch_size, self.z_dim))

    # Compute mean square loss
    def compute_ms_loss(self, data, pred, name=None):

        # Unpack solution
        _, __, mesh, soln  = data

        pred, scale = pred

        if not (self.alt_res == self.default_res):
            mesh = tf.image.resize_images(mesh, [self.alt_res, self.alt_res], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            soln = tf.image.resize_images(soln, [self.alt_res, self.alt_res])

        # Retrieve mesh file
        #num_batches, _, __, ___ = soln.get_shape().as_list()
        #mesh = tf.tile(self.mesh, [tf.shape(soln)[0],1,1,1])

        # Convert domain and boundary mesh to boolean masks
        # [Interior indices = 1, Boundary indices = 2]
        interior_indices = tf.greater(mesh, 0)
        boundary_indices = tf.greater(mesh, 1)
        zero_tensor = tf.zeros_like(soln)
        min_variance = -10.0*tf.ones_like(soln)

        # Set values outside current domain to zero:
        # tf.where(MASK, VAL_IF_TRUE, VAL_IF_FALSE)
        masked_pred = tf.where(interior_indices, pred, zero_tensor)
        masked_soln = tf.where(interior_indices, soln, zero_tensor)
        masked_scale = tf.where(interior_indices, scale, min_variance)

        # Compute Boundary Values
        boundary_pred = tf.where(boundary_indices, pred, zero_tensor)
        boundary_soln = tf.where(boundary_indices, soln, zero_tensor)

        # Determine number of interior points
        interior_count = tf.reduce_sum(tf.cast(interior_indices, np.float32), axis=[1,2])

        # Determine the number of boundary points
        boundary_count = tf.reduce_sum(tf.cast(boundary_indices, np.float32), axis=[1,2])

        # Determine true l2 norms
        interior_l2 = tf.reduce_sum(tf.pow(masked_soln, 2), axis=[1,2])
        #boundary_l2 = tf.reduce_sum(tf.pow(boundary_soln, 2), axis=[1,2])


        if self.use_relative:
            # Compute relative interior/boundary losses
            interior_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(masked_pred-masked_soln, 2), axis=[1,2])/interior_l2)
            #boundary_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(boundary_pred-boundary_soln, 2), axis=[1,2])/boundary_l2)
            boundary_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(boundary_pred-boundary_soln, 2), axis=[1,2])/boundary_count)
        else:
            # Compute interior/boundary losses
            interior_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(masked_pred-masked_soln, 2), axis=[1,2])/interior_count)
            boundary_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(boundary_pred-boundary_soln, 2), axis=[1,2])/boundary_count)


        # Compute negative log probability
        #scale_diag = tf.nn.softplus(tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res]))
        #mvn = tfd.MultivariateNormalDiag(loc=tf.reshape(masked_pred, [-1, self.alt_res*self.alt_res]), scale_diag=scale_diag)
        #prob_loss = -tf.reduce_mean(mvn.log_prob(tf.reshape(masked_soln, [-1, self.alt_res*self.alt_res])))

        # Compute negative log probability [manually]
        if self.use_prob_loss:
            soln_vals = tf.reshape(masked_soln, [-1, self.alt_res*self.alt_res])
            means = tf.reshape(masked_pred, [-1, self.alt_res*self.alt_res])        

        if self.use_prob_loss and self.use_softplus_implementation:

            if self.use_laplace:
                # LAPLACE LOSS
                b = tf.nn.softplus(tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res]))
                prob_loss = -tf.reduce_mean(tf.reduce_sum(-tf.divide(tf.abs(soln_vals-means), b) - tf.log(2*b), axis=1))

            elif self.use_cauchy:
                # CAUCHY LOSS
                gamma = tf.nn.softplus(tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res]))
                #prob_loss = tf.reduce_mean(tf.reduce_sum( tf.log( np.pi * tf.multiply(gamma, 1.0 + tf.divide(tf.pow(soln_vals-means,2), tf.pow(gamma,2)))), axis=1))
                # ALTERNATE IMPLEMENTATION
                prob_loss = tf.reduce_mean(tf.reduce_sum( tf.log( np.pi * (gamma + tf.divide(tf.pow(soln_vals-means,2), gamma)))), axis=1)
            else:
                # NORMAL LOSS
                stds_2 = tf.pow(tf.nn.softplus(tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res])), 2)
                if self.use_int_count:
                    prob_loss = -tf.reduce_sum(tf.reduce_sum(-tf.divide(tf.pow(soln_vals-means,2), 2*stds_2) - 0.5*tf.log(2*np.pi*stds_2), axis=1)/interior_count)
                else:
                    prob_loss = -tf.reduce_mean(tf.reduce_sum(-tf.divide(tf.pow(soln_vals-means,2), 2*stds_2) - 0.5*tf.log(2*np.pi*stds_2), axis=1))


            masked_scale = tf.nn.softplus(masked_scale)

        elif self.use_prob_loss:
            
            ###
            ###   LOG SCALE IMPLEMENTATION
            ###

            if self.use_laplace:
                # LAPLACE LOSS
                log_b = tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res])
                prob_loss = -tf.reduce_mean(tf.reduce_sum(-tf.divide(tf.abs(soln_vals-means), tf.exp(log_b)) - tf.log(2) - log_b, axis=1))

            elif self.use_cauchy:
                # CAUCHY LOSS
                #log_gamma = tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res])
                gamma = tf.exp(tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res]))
                #prob_loss = tf.reduce_mean(tf.reduce_sum( tf.log( np.pi * tf.multiply(gamma, 1.0 + tf.divide(tf.pow(soln_vals-means,2), tf.pow(gamma,2)))), axis=1))
                # ALTERNATE IMPLEMENTATION
                prob_loss = tf.reduce_mean(tf.reduce_sum( tf.log( np.pi * (gamma + tf.divide(tf.pow(soln_vals-means,2), gamma)))), axis=1)
            else:
                # NORMAL LOSS
                #stds_2 = tf.pow(tf.nn.softplus(tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res])), 2)
                log_stds = tf.reshape(masked_scale, [-1, self.alt_res*self.alt_res])
                #stds_2 = tf.pow(tf.exp(log_stds), 2)
                if self.use_int_count:
                    prob_loss = -tf.reduce_sum(tf.reduce_sum(-tf.divide(tf.pow(soln_vals-means,2), 2*tf.pow(tf.exp(log_stds), 2)) - 0.5*tf.log(2*np.pi) - 0.5*2.*log_stds, axis=1)/interior_count)
                else:
                    prob_loss = -tf.reduce_mean(tf.reduce_sum(-tf.divide(tf.pow(soln_vals-means,2), 2*tf.pow(tf.exp(log_stds), 2)) - 0.5*tf.log(2*np.pi) - 0.5*2.*log_stds, axis=1))


            masked_scale = tf.exp(masked_scale)

        else:
            # MSE LOSS
            prob_loss = 0.0
        
        return masked_soln, masked_pred, masked_scale, interior_loss, boundary_loss, prob_loss

    # Compute relative losses
    def compute_relative_loss(self, masked_soln, masked_pred, name=None):

        # Compute true solution L1 and L2 norms
        interior_l1 = tf.reduce_sum(tf.abs(masked_soln), axis=[1,2])
        interior_l2 = tf.reduce_sum(tf.pow(masked_soln, 2), axis=[1,2])

        # Compute average of relative L1 and L2 errors
        rel_l1 = tf.reduce_mean(tf.reduce_sum(tf.abs(masked_pred-masked_soln), axis=[1,2])/interior_l1)
        rel_l2 = tf.reduce_mean(tf.reduce_sum(tf.pow(masked_pred-masked_soln, 2), axis=[1,2])/interior_l2)
        
        return rel_l1, rel_l2

    
    # Compute average uncertainty in predictions
    def compute_uncertainty(self, data, scale):
        _, __, mesh, ___ = data
        interior_indices = tf.greater(mesh, 0)
        interior_count = tf.reduce_sum(tf.cast(interior_indices, np.float32), axis=[1,2])
        uq = tf.reduce_mean(tf.reduce_sum(scale, axis=[1,2])/interior_count)
        return uq
    
    # Compute Kullback–Leibler (KL) divergence
    def compute_kl_loss(self, mean, log_sigma, name=None):
        kl_loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(mean) + tf.exp(2*log_sigma) - \
                                                    2.*log_sigma - 1., axis=[-1]), name=name)
        return kl_loss
    
    # Define graph for model
    def build_model(self):

        # Define placeholder for noise vector
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # Define placeholder for dataset handle (to select training or validation)
        self.dataset_handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        self.iterator = tf.data.Iterator.from_string_handle(self.dataset_handle,
                                                            self.training_datasets[0].output_types,
                                                            self.training_datasets[0].output_shapes)
        self.data = self.iterator.get_next()

        # Define learning rate with exponential decay
        self.learning_rt = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                      self.lr_decay_step, self.lr_decay_rate, staircase=True)

        # Define placeholder for training status
        self.training = tf.placeholder(tf.bool, name='training')
        self.kl_wt = tf.placeholder(tf.float32, name='kl_wt')
        tf_false = tf.constant(False, dtype=tf.bool)

        # Define variable for storing kl_wt 
        self.kl_wt_value = tf.get_variable("kl_wt_value", dtype=tf.float32, shape=[],
                                           initializer=tf.initializers.zeros, trainable=False)
        self.assign_kl_wt_value = self.kl_wt_value.assign(self.kl_wt)

        # Compute predictions and loss for training/validation datasets
        self.masked_soln, self.masked_pred, self.masked_scale, self.interior_loss, self.boundary_loss, self.kl_loss, self.prob_loss = self.evaluate_model(self, self.data, training=self.training)

        # Compute average uncertainty in predictions
        self.uncertainty = self.compute_uncertainty(self.data, self.masked_scale)

        # Compute relative losses
        self.rel_l1, self.rel_l2 = self.compute_relative_loss(self.masked_soln, self.masked_pred, name="relative_loss")
        
        # Define l2 loss as weighted sum of interior and boundary loss
        self.l2_loss = tf.add(self.int_weight*self.interior_loss, self.bdry_weight*self.boundary_loss, name='l2_loss')

        # Compute regularization loss
        self.reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.multiply(self.reg_weight, tf.reduce_sum(self.reg_list), name='reg_loss')

        # Specify final loss function
        if self.use_prob_loss:
            if self.use_kl:
                if self.regularize:
                    self.loss = tf.add(tf.add(self.prob_loss, self.reg_loss), self.kl_loss, name="loss")
                else:
                    self.loss = tf.add(self.prob_loss, self.kl_loss, name="loss")
            else:
                if self.regularize:
                    self.loss = tf.add(self.prob_loss, self.reg_loss, name="loss")
                else:
                    self.loss = tf.identity(self.prob_loss, name="loss")
        else:
            if self.regularize:
                if self.use_kl:
                    self.loss = tf.add(tf.add(self.l2_loss, self.kl_loss), self.reg_loss, name="loss")
                else:
                    self.loss = tf.add(self.l2_loss, self.reg_loss, name="loss")
            else:
                if self.use_kl:
                    self.loss = tf.add(self.l2_loss, self.kl_loss, name="loss")
                else:
                    self.loss = tf.identity(self.l2_loss, name="loss")


        # Compute predictions and loss for early stopping checks
        _, self.epred, __, self.eloss, ___, ____, _____ = self.evaluate_model(self, self.edataset.get_next(), reuse=True,
                                                                              training=tf_false, suffix="_stopping")
        loss_stopping = tf.identity(self.eloss, name="loss_stopping")

        # Define optimizer for training
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if self.use_AMSGrad:
                self.optim = AMSGrad(learning_rate=self.learning_rt, beta1=self.adam_beta1, epsilon=1e-08) \
                    .minimize(self.loss, global_step=self.global_step)
            elif self.use_SGLD:
                self.optim = SGLD(learning_rate=self.learning_rt, beta1=self.adam_beta1, epsilon=1e-08) \
                    .minimize(self.loss, global_step=self.global_step)
                """
                start_rate = self.learning_rate
                gamma = 0.75
                decay_steps = 100000
                end_rate = 0.000001
                # decayed_rate = (init_rate - end_rate) * (1 - global_step / decay_steps) ^ (power) + end_rate
                epsilon = tf.train.polynomial_decay(start_rate, self.global_step, decay_steps, end_learning_rate=end_rate, power=gamma)
                
                
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rt)
                gvs = optimizer.compute_gradients(self.loss)
                #noise = tf.random_normal(shape=tf.shape(gvs), mean=0.0, stddev=epsilon)
                noisy_gvs = [(tf.add(epsilon/2.0*grad,
                                     tf.random_normal(shape=tf.shape(grad),stddev=epsilon)), var) for grad, var in gvs]
                self.optim = optimizer.apply_gradients(noisy_gvs, global_step=self.global_step)
                """
                
                
                """
                self.optim = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate = self.learning_rt,
                                                                              preconditioner_decay_rate = 0.95,
                                                                              data_size = self.data_count,
                                                                              burnin = 50, diagonal_bias = 1e-6) \
                                          .minimize(self.loss, global_step=self.global_step)
                """
                
            else:
                self.optim = tf.train.AdamOptimizer(self.learning_rt, beta1=self.adam_beta1, epsilon=1e-08) \
                                     .minimize(self.loss, global_step=self.global_step)

        # Define summary operations
        loss_sum = tf.summary.scalar("loss", self.loss)
        kl_loss_sum = tf.summary.scalar("kl_loss", self.kl_loss)
        reg_loss_sum = tf.summary.scalar("reg_loss", self.reg_loss)
        interior_loss_sum = tf.summary.scalar("interior_loss", self.interior_loss)
        boundary_loss_sum = tf.summary.scalar("boundary_loss", self.boundary_loss)
        self.merged_summaries = tf.summary.merge([loss_sum, kl_loss_sum, reg_loss_sum, interior_loss_sum, boundary_loss_sum])


        # Define placeholders for directly feeding data for manual testing
        tdata = tf.placeholder(tf.float32, [None, None, None, 1], name='data_test')
        tcoeff = tf.placeholder(tf.float32, [None, None, None, 1], name='coeff_test')
        tmesh = tf.placeholder(tf.float32, [None, None, None, 1], name='mesh_test')
        tsoln = tf.placeholder(tf.float32, [None, None, None, 1], name='soln_test')
        test_soln, test_pred, test_scale, _, __, ____, _____ = self.evaluate_model(self, [tdata, tcoeff, tmesh, tsoln], reuse=True, training=tf_false, suffix="_test")

        
    # Train model
    def train(self):

        # Define summary writer for saving log files (for training and validation)
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'training/'), graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter(os.path.join(self.log_dir, 'validation/'), graph=tf.get_default_graph())

        # Show list of all variables and total parameter count
        show_variables()

        # Recover kl_wt when restoring from checkpoint
        # Get handles for training and validation datasets
        self.training_handles, self.validation_handles, kl_wt = self.sess.run([[d.string_handle() for d in self.training_datasets],
                                                                               [d.string_handle() for d in self.validation_datasets],
                                                                               self.kl_wt_value])
        if not (kl_wt == 0.0):
            self.kl_weight = kl_wt
            print("\n[ Restoring Variables ]\n")
        else:
            print("\n[ Initializing Variables ]\n")

        self.current_handle = 0
        self.training_handle = self.training_handles[self.current_handle]
        self.validation_handle = self.validation_handles[self.current_handle]
        
        # Iterate through training steps
        while not self.sess.should_stop():

            # Update global step            
            step = tf.train.global_step(self.sess, self.global_step)

            # Switch to next rotated dataset after each epoch
            #if step % self.prefetch_count == 0:
            if step % self.dataset_step == 0:
                self.current_handle = int(np.mod(self.current_handle + 1, len(self.training_handles)))
                self.training_handle = self.training_handles[self.current_handle]
                self.validation_handle = self.validation_handles[self.current_handle]
                
            # Break if early stopping hook requests stop after sess.run()
            if self.sess.should_stop():
                break

            # Specify feed dictionary
            fd = {self.dataset_handle: self.training_handle, self.training: True,
                  self.z: np.zeros([self.batch_size, self.z_dim]), self.kl_wt: self.kl_weight}

            # Save summaries, display progress, and update model
            if (step % self.summary_step == 0) and (step % self.display_step == 0):
                summary, kl_loss, i_loss, b_loss, loss, _ = self.sess.run([self.merged_summaries, self.kl_loss,
                                                                           self.interior_loss, self.boundary_loss,
                                                                           self.loss, self.optim], feed_dict=fd)
                if self.use_kl:
                    print("Step %d:  %.7f [kl_loss]   %.7f [i_loss]  %.7f [b_loss]   %.7f [loss] " %(step,kl_loss,i_loss,b_loss,loss))
                else:
                    print("Step %d:  %.7f [i_loss]  %.7f [b_loss]   %.7f [loss] " %(step,i_loss,b_loss,loss))
                self.writer.add_summary(summary, step); self.writer.flush()

                if np.isnan(loss):
                    raise ValueError("NaN loss value encountered at global step %d." %(step))

            # Save summaries and update model
            elif step % self.summary_step == 0:
                summary, _ = self.sess.run([self.merged_summaries, self.optim], feed_dict=fd)
                self.writer.add_summary(summary, step); self.writer.flush()
            # Display progress and update model
            elif step % self.display_step == 0:
                kl_loss, i_loss, b_loss, loss, _ = self.sess.run([self.kl_loss, self.interior_loss, self.boundary_loss,
                                                                  self.loss, self.optim], feed_dict=fd)
                if self.use_kl:
                    print("Step %d:  %.7f [kl_loss]   %.7f [i_loss]  %.7f [b_loss]   %.7f [loss] " %(step,kl_loss,i_loss,b_loss,loss))
                else:
                    print("Step %d:  %.7f [i_loss]  %.7f [b_loss]   %.7f [loss] " %(step,i_loss,b_loss,loss))

            # Update model
            else:
                self.sess.run([self.optim], feed_dict=fd)
                
            # Break if early stopping hook requests stop after sess.run()
            if self.sess.should_stop():
                break

            # Plot predictions
            #if step % self.plot_step == 0:
            #    #self.plot_comparisons(step)
            #    self.plot_data(step, handle=self.training_handle)

            # Break if early stopping hook requests stop after sess.run()
            if self.sess.should_stop():
                break

            if step % self.summary_step == 0:
                if step >= self.kl_start_step:
                    # Save validation summaries and update kl_weight to avoid underfitting
                    fd = {self.dataset_handle: self.validation_handle, self.z: np.zeros([self.batch_size, self.z_dim]),
                          self.training: False, self.kl_wt: self.kl_weight}
                    vsummary, vkl_l, vi_l  = self.sess.run([self.merged_summaries, self.kl_loss, self.interior_loss], feed_dict=fd)
                    self.vwriter.add_summary(vsummary, step); self.vwriter.flush()
                    if self.use_kl_decay:
                        self.kl_weight = np.min([self.kl_weight, 0.5*self.kl_weight*(0.4*vi_l/vkl_l)])
                        self.sess.run(self.assign_kl_wt_value, feed_dict={self.kl_wt: self.kl_weight})
                else:
                    # Save validation summaries and update kl_weight to avoid underfitting
                    fd = {self.dataset_handle: self.validation_handle, self.z: np.zeros([self.batch_size, self.z_dim]),
                          self.training: False, self.kl_wt: self.kl_weight}
                    vsummary  = self.sess.run(self.merged_summaries, feed_dict=fd)
                    self.vwriter.add_summary(vsummary, step); self.vwriter.flush()


            if self.validation_checks:
                if step % self.evaluation_step == 0:
                    self.evaluate_validation(step)

                
    # Define method for computing model predictions
    def predict(self):
        fd = {self.dataset_handle: self.validation_handle, self.z: np.zeros([self.batch_size, self.z_dim]),
              self.training: False, self.kl_wt: self.kl_weight}
        soln, pred =  self.sess.run([self.masked_soln, self.masked_pred], feed_dict=fd)

        return soln, pred

    # Plot true and predicted images
    def plot_comparisons(self, step):
        plot_subdir = self.plot_dir + str(step) + "/"
        checkFolders([self.plot_dir, plot_subdir])
        soln, pred = self.predict()
        for n in range(0, self.batch_size):
            soln_name = 'soln_' + str(n) + '.npy'; pred_name = 'pred_' + str(n) + '.npy'
            np.save(os.path.join(plot_subdir, soln_name), soln[n,:,:,0])
            np.save(os.path.join(plot_subdir, pred_name), pred[n,:,:,0])
    
    # Plot input data from batch
    def plot_data(self, step, handle=None):
        if handle is None:
            handle = self.validation_handle
        plot_subdir = self.plot_dir + str(step) + "/"
        checkFolders([self.plot_dir, plot_subdir])
        fd = {self.dataset_handle: handle, self.z: np.zeros([self.batch_size, self.z_dim]),
              self.training: False, self.kl_wt: self.kl_weight}
        in_data, msoln, pred =  self.sess.run([self.data, self.masked_soln, self.masked_pred], feed_dict=fd)
        data, coeff, mesh, soln = in_data
        for n in range(0, self.batch_size):
            soln_name = 'soln_' + str(n) + '.npy'; data_name = 'data_' + str(n) + '.npy'; mesh_name = 'mesh_' + str(n) + '.npy'
            msoln_name = 'msoln_' + str(n) + '.npy'; pred_name = 'pred_' + str(n) + '.npy'
            coeff_name = 'coeff_' + str(n) + '.npy'
            np.save(os.path.join(plot_subdir, soln_name), soln[n,:,:,0])
            np.save(os.path.join(plot_subdir, data_name), data[n,:,:,0])
            np.save(os.path.join(plot_subdir, coeff_name), coeff[n,:,:,0])
            np.save(os.path.join(plot_subdir, mesh_name), mesh[n,:,:,0])
            np.save(os.path.join(plot_subdir, msoln_name), msoln[n,:,:,0])
            np.save(os.path.join(plot_subdir, pred_name), pred[n,:,:,0])
    
    # Compute cumulative loss over multiple batches
    def compute_cumulative_loss(self, loss, loss_ops, dataset_handle, batches):
        for n in range(0, batches):
            fd = {self.dataset_handle: dataset_handle, self.training: False, self.kl_wt: self.kl_weight}
            current_loss = self.sess.run(loss_ops, feed_dict=fd)
            loss = np.add(loss, current_loss)
            sys.stdout.write('Batch {0} of {1}\r'.format(n+1,batches))
            sys.stdout.flush()
        return loss

    # Evaluate model
    def evaluate_validation(self, step):
        v_batches = int(np.floor(0.2 * self.data_count/self.batch_size))
        validation_loss, validation_uq = self.compute_cumulative_loss([0.,0.],
                                                                      [self.l2_loss, self.uncertainty],
                                                                      self.validation_handles[0], v_batches)
        validation_loss = validation_loss/v_batches
        validation_uq = validation_uq/v_batches

        #print(validation_loss.shape)
        #print(validation_uq.shape)
        
        with open(os.path.join(self.model_dir, "evaluation_losses.csv"), "a") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([step, validation_loss])

        with open(os.path.join(self.model_dir, "evaluation_uncertainties.csv"), "a") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([step, validation_uq])

        """
        v_batches = int(np.floor(0.2 * self.data_count/self.batch_size))
        validation_loss = self.compute_cumulative_loss([0.], [self.l2_loss], self.validation_handles[0], v_batches)
        validation_loss = validation_loss/v_batches
        with open(os.path.join(self.model_dir, "evaluation_losses.csv"), "a") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([step, validation_loss[0]])
        """
        
    # Evaluate model
    def evaluate(self):
        t_batches = int(np.floor(0.8 * self.data_count/self.batch_size))
        v_batches = int(np.floor(0.2 * self.data_count/self.batch_size))
        print("\nTraining dataset:")
        #training_loss = self.compute_cumulative_loss([0.], [self.l2_loss], self.training_handles[0], t_batches)
        #t_loss, t_uq = self.compute_cumulative_loss([0.,0.], [self.l2_loss, self.uncertainty], self.training_handles[0], t_batches)
        t_loss, t_uq, t_l1, t_l2 = self.compute_cumulative_loss([0.,0.,0.,0.],
                                                                [self.l2_loss, self.uncertainty, self.rel_l1, self.rel_l2],
                                                                self.training_handles[0], t_batches)
        print("\n\nValidation dataset:")
        #validation_loss = self.compute_cumulative_loss([0.], [self.l2_loss], self.validation_handles[0], v_batches)
        #v_loss, v_uq = self.compute_cumulative_loss([0.,0.], [self.l2_loss, self.uncertainty], self.validation_handles[0], v_batches)
        v_loss, v_uq, v_l1, v_l2 = self.compute_cumulative_loss([0.,0.,0.,0.],
                                                                [self.l2_loss, self.uncertainty, self.rel_l1, self.rel_l2],
                                                                self.validation_handles[0], v_batches)
        training_loss = t_loss/t_batches
        validation_loss = v_loss/v_batches
        training_uq = t_uq/t_batches
        validation_uq = v_uq/v_batches
        training_l1 = t_l1/t_batches
        validation_l1 = v_l1/v_batches
        training_l2 = t_l2/t_batches
        validation_l2 = v_l2/v_batches
        return training_loss, validation_loss, training_uq, validation_uq, training_l1, validation_l1, training_l2, validation_l2
