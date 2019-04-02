import sys
import os

# Import flags specifying dataset parameters
from flags import getFlags


if __name__ == '__main__':
    FLAGS = getFlags()

    data_count = int(FLAGS.cov_count * FLAGS.data_count)
    data_prefix = "./Data/data_"
    mesh_prefix = "./Meshes/mesh_"

    data_missed = 0
    soln_missed = 0
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
            soln_missed += 1
        if n % progress_step == 0:
            sys.stdout.write('\r  Progress:  {0:.1%}'.format((n+1)/data_count))
            sys.stdout.flush()

    #print("\n Data / Solution Missing XML File Counts:")
    #print(data_missed)
    #print(soln_missed)

