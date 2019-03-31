import os
import sys
import numpy as np
import multiprocessing

# Import flags specifying dataset parameters
from flags import getFlags


def preprocess_data(start_index, data_count, data_dir, mesh_dir, soln_dir, RESCALE=False, use_hires=False):

    for i in range(start_index, start_index + data_count):
        if use_hires:
            hires_mesh = np.load(mesh_dir + 'hires_mesh_' + str(i) + '.npy')
            hires_out_of_domain = (hires_mesh == 0)

            hires_data = np.load(data_dir + 'hires_data_' + str(i) + '.npy')
            hires_data[hires_out_of_domain] = 0.0
            hires_soln = np.load(soln_dir + 'hires_solution_' + str(i) + '.npy')

            if RESCALE:
                ## Rescale data and solutions
                hires_scaling = np.max(np.abs(hires_data))
                hires_data = hires_data/hires_scaling
                hires_soln = hires_soln/hires_scaling

            np.save(data_dir + 'hires_data_' + str(i) + '.npy', hires_data)
            np.save(soln_dir + 'hires_solution_' + str(i) + '.npy', hires_soln)
        else:
            mesh = np.load(mesh_dir + 'mesh_' + str(i) + '.npy')
            out_of_domain = (mesh == 0)

            data = np.load(data_dir + 'data_' + str(i) + '.npy')
            data[out_of_domain] = 0.0
            soln = np.load(soln_dir + 'solution_' + str(i) + '.npy')

            if RESCALE:
                ## Rescale data and solutions
                scaling = np.max(np.abs(data))
                data = data/scaling
                soln = soln/scaling

            np.save(data_dir + 'data_' + str(i) + '.npy', data)
            np.save(soln_dir + 'solution_' + str(i) + '.npy', soln)


if __name__ == '__main__':
    FLAGS = getFlags()

    # Divide tasks into smaller pieces
    subdivision = 5

    #hires_mesh = np.load(FLAGS.mesh_dir + 'hires_mesh_' + str(0) + '.npy')

    def preprocess(d):
        preprocess_data(d, int(FLAGS.data_count/subdivision), FLAGS.data_dir, FLAGS.mesh_dir, FLAGS.soln_dir, use_hires=FLAGS.use_hires)

    # Create multiprocessing pool
    NumProcesses = FLAGS.cpu_count
    pool = multiprocessing.Pool(processes=NumProcesses)

    start_indices = [int(n*FLAGS.data_count/subdivision) for n in range(0,subdivision*FLAGS.cov_count)]
    start_indices = [FLAGS.data_start_count + n for n in start_indices]
    
    print('\n [ Preprocessing Data ]\n')
    num_tasks = subdivision*FLAGS.cov_count
    for i, _ in enumerate(pool.imap_unordered(preprocess, [d for d in start_indices]), 1):
        sys.stdout.write('\r  Progress:  {0:.1%}'.format(i/num_tasks))
        sys.stdout.flush()
    print('\n')
