from __future__ import division

import os
import numpy as np
import multiprocessing
import sys
import time

# Garbage collection
import gc

# Import flags specifying dataset parameters
from flags import getFlags

# Import solver function
from solver import gen_soln

# Specify callback function for clearing memory
ready_list = []
tasks_completed = 0
num_tasks = 0
def callback(return_vals):
    global tasks_completed
    global num_tasks
    tasks_completed += 1
    sys.stdout.write('\r  Progress:  {0:.1%}'.format(tasks_completed/num_tasks))
    sys.stdout.flush()
    if tasks_completed == num_tasks:
        print("\n")
    global ready_list
    ready_list.append(index)
    gc.collect()

# Define worker function for multiprocessing pool
def solve(index):
    FLAGS = getFlags()
    subdivision = 5
    start_indices = [int(n*FLAGS.data_count/subdivision) for n in range(0,subdivision*FLAGS.cov_count)]
    start_indices = [FLAGS.data_start_count + n for n in start_indices]
    gen_soln(start_indices[index], int(FLAGS.data_count/subdivision), FLAGS.resolution, FLAGS.mesh_resolution)
    return index

if __name__ == '__main__':
    FLAGS = getFlags()

    # Divide tasks into smaller pieces
    subdivision = 5

    # Create multiprocessing pool
    NumProcesses = FLAGS.cpu_count
    pool = multiprocessing.Pool(processes=NumProcesses, maxtasksperchild=2)

    
    start_indices = [int(n*FLAGS.data_count/subdivision) for n in range(0,subdivision*FLAGS.cov_count)]
    start_indices = [FLAGS.data_start_count + n for n in start_indices]

    def get_progress(step, total_steps, start_time):
        current_time = time.clock()
        time_elapsed = current_time - start_time
        rate = time_elapsed/step
        approx_finish = rate * (total_steps - step)
        hours = np.floor(approx_finish/3600.0)
        if hours > 0:
            minutes = np.floor((approx_finish/3600.0 - hours) * 60)
            seconds = np.floor(((approx_finish/3600.0 - hours) * 60 - minutes) * 60)
            progress = '    [ Estimated Time  ~  ' + str(int(hours)) + 'h  ' + str(int(minutes))+'m  '+str(int(seconds))+'s ]'
        else:
            minutes = np.floor(approx_finish/60.0)
            seconds = np.floor((approx_finish/60.0 - minutes) * 60)
            progress = '    [ Estimated Time  ~  ' + str(int(minutes)) + 'm  ' + str(int(seconds)) + 's ]'
        return progress


    """
    print('\n [ Solving Systems ]\n')
    num_tasks = subdivision*FLAGS.cov_count
    for i, _ in enumerate(pool.imap_unordered(sample, [d for d in start_indices]), 1):
        sys.stdout.write('\r  Progress:  {0:.1%}'.format(i/num_tasks))
        sys.stdout.flush()
    print('\n')
    """

    print('\n [ Solving Systems ]\n')
    num_tasks = subdivision*FLAGS.cov_count
    result = {}
    for index in range(0, num_tasks):
        result[index] = (pool.apply_async(solve, (index,), callback=callback))
        for ready in ready_list:
            result[ready].wait()
            del result[ready]
        ready_list = []
        

    # Close multiprocessing pool
    pool.close()
    pool.join()


    
