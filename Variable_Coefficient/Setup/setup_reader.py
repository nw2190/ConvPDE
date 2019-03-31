import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import numpy as np
import os


    
def plot_data(ID, model_dir='./Data/', Rm_Outliers=False, Filter=True, Plot_Error=False, HIRES=False, COEFF=False):
    mpl.style.use('classic')
    if HIRES:
        if COEFF:
            data_file = model_dir + 'hires_coeff_' + str(ID) + '.npy'
        else:
            data_file = model_dir + 'hires_data_' + str(ID) + '.npy'
    else:
        if COEFF:
            data_file = model_dir + 'coeff_' + str(ID) + '.npy'
        else:
            data_file = model_dir + 'data_' + str(ID) + '.npy'
            
    data = np.load(data_file)
    
    data_vals, plot_X, plot_Y = preprocesser(data, Rm_Outliers=False, Filter=Filter)
    data_min = np.min(data_vals)
    data_max = np.max(data_vals)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(plot_X,plot_Y, data_vals, cmap='hot')
    data_title = 'Data:    min = %.6f    max = %.6f'  %(data_min, data_max)
    ax1.set_title(data_title)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
        
    plt.show()
    return data


def plot_mesh(ID, model_dir='./Meshes/', Rm_Outliers=False, Filter=True, Plot_Error=False, HIRES=False):
    mpl.style.use('classic')
    if HIRES:
        mesh_file = model_dir + 'hires_mesh_' + str(ID) + '.npy'
    else:
        mesh_file = model_dir + 'mesh_' + str(ID) + '.npy'
    mesh = np.load(mesh_file)
    plt.imshow(mesh)
    plt.show()
    return mesh

def plot_soln(ID, model_dir='./Solutions/', Rm_Outliers=False, Filter=True, Plot_Error=False, HIRES=False):
    mpl.style.use('classic')
    if HIRES:
        soln_file = model_dir + 'hires_solution_' + str(ID) + '.npy'
    else:
        soln_file = model_dir + 'solution_' + str(ID) + '.npy'
        
    soln = np.load(soln_file)
    
    soln_vals, plot_X, plot_Y = preprocesser(soln, Rm_Outliers=False, Filter=False)
    soln_min = np.min(soln_vals)
    soln_max = np.max(soln_vals)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(plot_X,plot_Y, soln_vals, cmap='hot')
    soln_title = 'Soln:    min = %.6f    max = %.6f'  %(soln_min, soln_max)
    ax1.set_title(soln_title)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
        
    plt.show()
    return soln


# Apply median filter to two-dimensional array
def median_filter(vals):
    resolution = vals.shape[0]
    padded = np.lib.pad(vals, (1,), 'constant', constant_values=(0.0,0.0))
    
    for i in range(1,resolution+1):
        for j in range(1,resolution+1):
            vals[i-1,j-1] = np.median(padded[i-1:i+2,j-1:j+2])

    return vals

# Apply mean filter to two-dimensional array
def mean_filter(vals):
    resolution = vals.shape[0]
    padded = np.lib.pad(vals, (1,), 'constant', constant_values=(0.0,0.0))
    
    for i in range(1,resolution+1):
        for j in range(1,resolution+1):
            vals[i-1,j-1] = np.mean(padded[i-1:i+2,j-1:j+2])

    return vals

    
# Plots predictions with matplotlib
def preprocesser(vals, refine=2, Rm_Outliers=False, Filter=True, Median=False, Mean=True):

    # Determine spatial resolution
    resolution = vals.shape[0]
    
    if Rm_Outliers:
        # Identify and remove outliers
        outlier_buffer = 5
        
        vals_list = vals.reshape((resolution*resolution,))
        vals_mins = heapq.nsmallest(outlier_buffer, vals_list)
        vals_maxes = heapq.nlargest(outlier_buffer, vals_list)

        # Cap max and min
        vals_min = np.max(vals_mins)
        vals_max = np.min(vals_maxes)
        
        # Trim outliers
        over  = (vals > vals_max)
        under = (vals < vals_min)

        # Remove outliers
        vals[over] = vals_max
        vals[under] = vals_min
        
    else:
        vals_min = np.max(vals)
        vals_max = np.min(vals)

    if Filter:
        # Apply median/mean filter
        if Median:
            vals = median_filter(vals)
        if Mean:
            vals = mean_filter(vals)

    # Create grid
    start = 0.0
    end = 1.0
    x = np.linspace(start,end,resolution)
    y = np.linspace(start,end,resolution)

    [X, Y] = np.meshgrid(x,y)

    interp_vals = interp2d(x,y, vals, kind='cubic')

    # Create refined grid
    plot_start = 0.0
    plot_end = 1.0
    
    plot_x = np.linspace(plot_start,plot_end,refine*resolution)
    plot_y = np.linspace(plot_start,plot_end,refine*resolution)

    [plot_X, plot_Y] = np.meshgrid(plot_x, plot_y)

    vals_int_values = interp_vals(plot_x, plot_y)

    return vals_int_values, plot_X, plot_Y
