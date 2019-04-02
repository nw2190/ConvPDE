import sys
import os

# Import flags specifying dataset parameters
from setup_flags import getFlags


if __name__ == '__main__':
    FLAGS = getFlags()

    data_count = int(FLAGS.cov_count * FLAGS.data_count)
    data_prefix = "./Data/data_"
    mesh_prefix = "./Meshes/mesh_"

    data_missed = 0
    mesh_missed = 0
    progress_step = 1000
    print('\n [ Cleaning XML Files ]\n')
    for n in range(0,data_count):
        try:
            os.remove(data_prefix + str(n) + ".xml")
        except:
            data_missed += 1
        try:
            os.remove(mesh_prefix + str(n) + ".xml")
        except:
            mesh_missed += 1
        if n % progress_step == 0:
            sys.stdout.write('\r  Progress:  {0:.1%}'.format((n+1)/data_count))
            sys.stdout.flush()
    print('\n')
    
    #print("\n Data / Mesh XML Files Missing:")
    #print(data_missed)
    #print(mesh_missed)

