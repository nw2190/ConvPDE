#import pycuda.gpuarray as gpuarray
#import skcuda.linalg as linalg
#linalg.init()
#import pycuda.autoinit

import os
import sys
import csv
from dolfin import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp2d
import cv2


"""
from ctypes import *

# Load the shared object files
kernel = CDLL('./kernel.so')

#compile using:
# gcc -shared -Wl,-soname,k_uu -o k_uu.so -fPIC k_uu.c
# OR
# gcc -c -fPIC kernel.c -o kernel.o
# gcc kernel.o -shared  -o kernel.so

# Specify argument/result types
kernel.kernel.argtypes = [c_double, c_double, c_double, c_double]
kernel.kernel.restype = c_double

def eval_kernel(x_val, y_val, a_val, l_val):
    return kernel.kernel(x_val,y_val,a_val,l_val)

# Vectorize functions
kernel = np.vectorize(eval_kernel)

def eval_kernel_grid(x_vals,y_vals,a_val,l_val):
    tiled_x_vals = np.tile(x_vals,[y_vals.shape[0],1]).transpose()
    tiled_y_vals = np.tile(y_vals,[x_vals.shape[0],1])
    return kernel(tiled_x_vals,tiled_y_vals,a_val,l_val)
"""
            
# Generate covariance matrix and save Cholesky factor
def generate_covariance(a_val, b_val, resolution=64, filename='L.npy', alpha=1.0, length=0.25):

    #print("\nGenerating covariance matrix...")
    
    x_vals = np.linspace(a_val,b_val,resolution)
    y_vals = np.linspace(a_val,b_val,resolution)

    N = resolution*resolution
    coords = []
    for i in range(0,resolution):
        for j in range(0,resolution):
            coord = [x_vals[i],y_vals[j]]
            coords.append(coord)

    """    
    K = np.zeros([N,N])
    py_kernel = lambda x1,x2: 1.0/alpha*np.exp(-(np.square(x1[0]-x2[0])+np.square(x1[1]-x2[1]))/(2.0*np.square(length)))

    for i in range(0,N):
        sys.stdout.write('  - Assembling Row: {0} of {1}\r'.format(i+1,N))
        sys.stdout.flush()
        for j in range(0,N):
            coord_i = coords[i]
            coord_j = coords[j]
            K[i,j] = py_kernel(coord_i,coord_j)
    """

    # Compute pairwise distances and corresponding covariance matrix
    # (calculates the n*(n-1)/2 distances between distinct coordinates
    #  corresponding to the upper-triangular entries of the squareform)
    distances = squareform(pdist(coords, 'euclidean'))
    K = np.exp(-0.5*np.square(distances)/np.square(length))

    def compute_cholesky(K, filename):
        
        
        N = K.shape[0]
        jitter = 10e-9
        K = K + jitter*np.eye(N)
        L = np.linalg.cholesky(K)
        np.save(filename, L)

        #K_gpu = gpuarray.to_gpu(K) 
        #cholesky(K_gpu)
        #np.save(filename, K_gpu)
        
    #print('  - Computing Cholesky Factorization')
    compute_cholesky(K, filename)


def sample_gaussian(data_count, current_data, resolution=64, filename="L.npy", GEN_STIFFNESS=False, use_hires=False):

    #print("\nSampling Gaussian processes...")

    # Reset random seed from parent process
    np.random.seed(seed=current_data)
    
    L = np.load(filename)


    if GEN_STIFFNESS:

        def get_samples(L,count):
            N = L.shape[0]
            n = int(np.sqrt(N))
            rvs = np.random.normal(0.0,1.0,size=[N,count])
            #print('  - Matrix-Vector Multiplication')
            mult_rvs = np.matmul(L,rvs)
            for i in range(0,count):
                #sys.stdout.write('  - Sample: {0} of {1}\r'.format(i+1,count))
                #sys.stdout.flush()
                sample = np.reshape(mult_rvs[:,i],[n,n])

                # Enforce coercivity requirement
                sample = sample - np.mean(sample)
                delta = 0.2            
                scaling = np.max(np.abs(sample))
                if scaling > 1.0-delta:
                    sample = (1.0-delta)/scaling*sample
                sample = sample + 1.0


                filename = './Data/coeff_' + str(current_data + i) + '.npy'
                np.save(filename,sample)

    else:

        def get_samples(L,count):
            N = L.shape[0]
            n = int(np.sqrt(N))
            rvs = np.random.normal(0.0,1.0,size=[N,count])
            #print('  - Matrix-Vector Multiplication')
            mult_rvs = np.matmul(L,rvs)
            for i in range(0,count):
                #sys.stdout.write('  - Sample: {0} of {1}\r'.format(i+1,count))
                #sys.stdout.flush()
                sample = np.reshape(mult_rvs[:,i],[n,n])
                filename = './Data/data_' + str(current_data + i) + '.npy'
                np.save(filename,sample)


            
    #def save_samples(samples,count):
    #    for i in range(0,count):
    #        filename = './Data/data_' + str(current_data + i) + '.npy'
    #        np.save(filename,samples[i])

    get_samples(L, data_count)
    #samples = get_samples(L, data_count)
    #save_samples(samples, data_count)



def convert_samples(data_count, current_data, resolution=64):
    #set_log_level(ERROR)
    set_log_level(40)
    #print("\n\nConverting samples...")
        
    # Avoid having to reindex
    parameters["reorder_dofs_serial"] = False

    # Mesh and functionspace
    resolution = resolution
    mesh = UnitSquareMesh(resolution-1,resolution-1)

    for n in range(current_data,current_data + data_count):

        #sys.stdout.write('  - Converting: {0} of {1}\r'.format(n - current_data + 1, data_count))
        #sys.stdout.flush()
        #print('\nConverting %d of %d\n' %(n - current_data + 1, data_count))

        # Create some function
        resolution = resolution
        mesh = UnitSquareMesh(resolution-1,resolution-1)
        V = FunctionSpace(mesh, 'CG', 1)
        f_exp = Expression('1.0', degree=1)
        f = interpolate(f_exp, V)


        f_nodal_values = f.vector()
        #f_array = f_nodal_values.array()
        f_array = f_nodal_values.get_local()
        #print(f_array.shape)
        data = np.reshape(np.load('./Data/data_' + str(n) +  '.npy'),[resolution*resolution,])
        #f_array = reindex(data)
        f_array = data
        f.vector()[:] = f_array
        f.vector().set_local(f_array)  # alternative
        #print f.vector().array()

        new_resolution = 2*resolution
        new_mesh = UnitSquareMesh(new_resolution-1,new_resolution-1)
        new_V = FunctionSpace(new_mesh, 'CG', 1)
        new_f = project(f, new_V)

        data_function_filename = './Data/data_' + str(n) + '.xml'
        File(data_function_filename) << f

        #new_data_function_filename = './Data/hires_data_' + str(n) + '.xml'
        #File(new_data_function_filename) << new_f

        if use_hires:
            ## Save hi-res data array
            new_resolution = 2*resolution
            step = 1.0/new_resolution
            start = 0.0 + step/2.0

            vals = np.zeros([new_resolution,new_resolution])
            for i in range(0,new_resolution):
                for j in range(0,new_resolution):
                    x_coord = start + i*step
                    y_coord = start + (new_resolution - 1 - j)*step
                    pt = Point(x_coord, y_coord)
                    cell_id = new_mesh.bounding_box_tree().compute_first_entity_collision(pt)
                    #if mesh.bounding_box_tree().collides(pt):
                    if cell_id < new_mesh.num_cells():
                        try:
                            # Interior points can be evaluated directly
                            vals[j,i] = new_f(pt)
                        except:
                            # Points near the boundary have issues due to rounding...
                            cell = Cell(new_mesh, cell_id)
                            coords = cell.get_vertex_coordinates()
                            new_x_coord = coords[0]
                            new_y_coord = coords[1]
                            new_pt = Point(new_x_coord, new_y_coord)
                            vals[j,i] = new_f(new_pt)

            hires_filename = './Data/hires_data_' + str(n) + '.npy'
            np.save(hires_filename, vals)




def fast_convert_samples(data_count, current_data, resolution=64, GEN_STIFFNESS=False, use_hires=False):
    #set_log_level(ERROR)
    set_log_level(40)
    #print("\n\nConverting samples...")
        
    # Avoid having to reindex
    parameters["reorder_dofs_serial"] = False

    # Mesh and functionspace
    resolution = resolution
    mesh = UnitSquareMesh(resolution-1,resolution-1)
    V = FunctionSpace(mesh, 'CG', 1)
    f_exp = Expression('1.0', degree=1)
    f = interpolate(f_exp, V)

    for n in range(current_data,current_data + data_count):

        #sys.stdout.write('  - Converting: {0} of {1}\r'.format(n - current_data + 1, data_count))
        #sys.stdout.flush()

        f_nodal_values = f.vector()
        #f_array = f_nodal_values.array()
        f_array = f_nodal_values.get_local()
        #print(f_array.shape)
        if GEN_STIFFNESS:
            data = np.reshape(np.load('./Data/coeff_' + str(n) +  '.npy'),[resolution*resolution,])
        else:
            data = np.reshape(np.load('./Data/data_' + str(n) +  '.npy'),[resolution*resolution,])
        #f_array = reindex(data)
        f_array = data
        f.vector()[:] = f_array
        f.vector().set_local(f_array)  # alternative
        #print f.vector().array()
        if GEN_STIFFNESS:
            data_function_filename = './Data/coeff_' + str(n) + '.xml'
        else:
            data_function_filename = './Data/data_' + str(n) + '.xml'
        File(data_function_filename) << f

        if use_hires:
            ## Save hi-res data array
            new_resolution = 2*resolution
            data_array = np.reshape(data, [resolution, resolution])
            resized_array = cv2.resize(data_array, (new_resolution, new_resolution), interpolation = cv2.INTER_CUBIC) 
            
            if GEN_STIFFNESS:
                hires_filename = './Data/hires_coeff_' + str(n) + '.npy'
            else:
                hires_filename = './Data/hires_data_' + str(n) + '.npy'

            np.save(hires_filename, resized_array)



