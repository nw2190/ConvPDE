from __future__ import division

import os
import numpy as np
import multiprocessing
import sys
import time

# Import flags specifying dataset parameters
from flags import getFlags

# Import solver function
from solver import gen_soln


if __name__ == '__main__':
    FLAGS = getFlags()

    # Divide tasks into smaller pieces
    subdivision = 10
    #subdivision = 20
    #subdivision = 5
    #subdivision = 1
    
    def sample(d):
        gen_soln(d, int(FLAGS.data_count/subdivision), FLAGS.resolution, FLAGS.mesh_resolution)

    # Create multiprocessing pool
    NumProcesses = FLAGS.cpu_count
    #pool = multiprocessing.Pool(processes=NumProcesses)
    #pool = multiprocessing.Pool(processes=NumProcesses, maxtasksperchild=2)

    
    start_indices = [int(n*FLAGS.data_count/subdivision) for n in range(0,subdivision*FLAGS.cov_count)]
    start_indices = [FLAGS.data_start_count + n for n in start_indices]
    
    #pool.map(sample, [d for d in start_indices])



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


    # Split tasks over multiple pools
    POOLS = FLAGS.solver_pools
    tasks_per_pool = int(np.floor(len(start_indices)/POOLS))
    
    print('\n [ Solving Systems ]\n')    
    for P in range(0,POOLS):
        
        indices = start_indices[P*tasks_per_pool:(P+1)*tasks_per_pool]
                                
        pool = multiprocessing.Pool(processes=NumProcesses)

        num_tasks = subdivision*FLAGS.cov_count
        #for i, _ in enumerate(pool.imap_unordered(sample, [d for d in start_indices]), 1):
        for i, _ in enumerate(pool.imap_unordered(sample, [d for d in indices]), 1):
            sys.stdout.write('\r  Progress:  {0:.1%}'.format((P*tasks_per_pool + i)/num_tasks))
            sys.stdout.flush()

        pool.close()
        pool.join()
        
    print('\n')


    
    """
    print('\n [ Solving Systems ]\n')
    #start_time = time.clock()
    num_tasks = subdivision*FLAGS.cov_count
    for i, _ in enumerate(pool.imap_unordered(sample, [d for d in start_indices]), 1):
        if i == 1:
            start_time = time.clock()
            sys.stdout.write('\r  Progress:  {0:.1%}   {1:}'.format(i/num_tasks, '    [  Estimated Time  ~  N/A  ]'))
            sys.stdout.flush()
        else:
            sys.stdout.write('\r  Progress:  {0:.1%}   {1:}'.format((i-1)/(num_tasks-1), get_progress(i-1, num_tasks-1, start_time)))
            sys.stdout.flush()
        print('\n')
    """
