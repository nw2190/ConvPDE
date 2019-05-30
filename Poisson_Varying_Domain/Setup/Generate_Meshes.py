import numpy as np
from mshr import *
from dolfin import *
import multiprocessing
import sys

# Import flags specifying dataset parameters
from setup_flags import getFlags

# Import mesh generation function
from mesh import gen_mesh_batch


if __name__ == '__main__':
    FLAGS = getFlags()

    # Divide tasks into smaller pieces
    subdivision = 2

    def mesh_batch(d):
        gen_mesh_batch(FLAGS.resolution, FLAGS.vertex_min, FLAGS.vertex_max + 1, FLAGS.mesh_resolution, FLAGS.mesh_dir, int(FLAGS.data_count/subdivision), d, use_hires=FLAGS.use_hires)
    
    # Create multiprocessing pool
    NumProcesses = 2*FLAGS.cpu_count
    pool = multiprocessing.Pool(processes=NumProcesses)
    
    start_indices = [int(n*FLAGS.data_count/subdivision) for n in range(0,subdivision*FLAGS.cov_count)]
    start_indices = [FLAGS.data_start_count + n for n in start_indices]
    
    #pool.map(convert, [d for d in start_indices])

    print('\n [ Generating Meshes ]\n')
    num_tasks = subdivision*FLAGS.cov_count
    for i, _ in enumerate(pool.imap_unordered(mesh_batch, [d for d in start_indices]), 1):
        sys.stdout.write('\r  Progress:  {0:.1%}'.format(i/num_tasks))
        sys.stdout.flush()
    print('\n')
