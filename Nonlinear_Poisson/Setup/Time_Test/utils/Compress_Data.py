import os
import sys
import numpy as np
import multiprocessing

# Import flags specifying dataset parameters
from flags import getFlags


# Data count per file
increment = 1000

if __name__ == '__main__':
    FLAGS = getFlags()
    data_count = FLAGS.data_count*FLAGS.cov_count
    data_dir = FLAGS.data_dir
    mesh_dir = FLAGS.mesh_dir
    resolution = FLAGS.resolution
    

    file_count = int(data_count/increment)
    data_per_file = int(increment)
    step = 0
    for fc in range(0,file_count):
        
        mesh_array = np.zeros([data_per_file,resolution,resolution,1])
        data_array = np.zeros([data_per_file,resolution,resolution,1])
        
        for i in range(0,data_per_file):
            mesh = np.load(mesh_dir + 'mesh_' + str(step) + '.npy')
            data = np.load(data_dir + 'data_' + str(step) + '.npy')
            mesh_array[i,:,:,0] = mesh
            data_array[i,:,:,0] = data
            step += 1
            
        np.save(mesh_dir + "Meshes_" + str(fc) + ".npy", mesh_array)
        np.save(data_dir + "Data_" + str(fc) + ".npy", data_array)

    """
    mesh_array = np.zeros([data_count,resolution,resolution,1])
    data_array = np.zeros([data_count,resolution,resolution,1])

    for i in range(0,data_count):
        mesh = np.load(mesh_dir + 'mesh_' + str(i) + '.npy')
        data = np.load(data_dir + 'data_' + str(i) + '.npy')
        mesh_array[i,:,:,0] = mesh
        data_array[i,:,:,0] = data

    np.save(mesh_dir + "Meshes.npy", mesh_array)
    np.save(data_dir + "Data.npy", data_array)
    """
