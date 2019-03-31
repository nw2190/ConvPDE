import numpy as np
from mshr import *
from dolfin import *

# Import flags specifying dataset parameters
from flags import getFlags

# Import mesh generation function
from mesh import gen_mesh


if __name__ == '__main__':
    FLAGS = getFlags()
    gen_mesh(FLAGS.resolution, FLAGS.mesh_resolution, FLAGS.mesh_dir, 0, FLAGS.coarse_resolution)
