from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

#from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

import glob
import sys
import csv
import time
import string
import heapq

import cv2

SCALING = 100.0

# Recover values and alpha mask corresponding to data image
def read_mesh(mesh_ID, mesh_directory, USE_HIRES=False, **keyword_parameters):
    if USE_HIRES:
        mesh_label = 'hires_mesh_' + str(mesh_ID)
    else:
        mesh_label = 'mesh_' + str(mesh_ID) 
    mesh_file = mesh_directory + mesh_label + '.npy'
    vals = np.load(mesh_file)

    if ('transformation' in keyword_parameters):
        [R, F] = keyword_parameters['transformation']
        vals = np.rot90(vals, k=R)
        if F == 1:
            vals = np.flipud(vals)
            
    vals_array = np.array([vals])
    return vals_array

# Recover values and alpha mask corresponding to data image
def read_data(data_ID, data_directory, USE_HIRES=False, STIFF=False, **keyword_parameters):
    if STIFF:
        if USE_HIRES:
            data_label = 'hires_coeff_' + str(data_ID)
        else:
            data_label = 'coeff_' + str(data_ID)
    else:
        if USE_HIRES:
            data_label = 'hires_data_' + str(data_ID)
        else:
            data_label = 'data_' + str(data_ID)
            
    data_file = data_directory + data_label + '.npy'
    vals = np.load(data_file)

    if ('transformation' in keyword_parameters):
        [R, F] = keyword_parameters['transformation']
        vals = np.rot90(vals, k=R)
        if F == 1:
            vals = np.flipud(vals)

    vals_array = np.array([vals])
    return vals_array

# Recover values and alpha mask corresponding to solution image 
def read_soln(source_ID, solution_directory, USE_HIRES=False, **keyword_parameters):
    ID_label = str(source_ID)
    if USE_HIRES:
        soln_label = 'hires_solution_' + ID_label
    else:
        soln_label = 'solution_' + ID_label 
    soln_file = solution_directory + soln_label + '.npy'
    vals = np.load(soln_file)

    if ('transformation' in keyword_parameters):
        [R, F] = keyword_parameters['transformation']
        vals = np.rot90(vals, k=R)
        if F == 1:
            vals = np.flipud(vals)
            
    vals = SCALING*vals
    vals_array = np.array([vals])
    return vals_array




def get_time():
    return time.time()

def convert_time(t):
    hours = np.floor(t/3600.0)
    minutes = np.floor((t/3600.0 - hours) * 60)
    seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
    if hours > 0:
        minutes = np.floor((t/3600.0 - hours) * 60)
        seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
        t_str = str(int(hours)) + 'h  ' + \
                str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    elif (hours == 0) and (minutes >= 1):
        minutes = np.floor(t/60.0)
        seconds = np.ceil((t/60.0 - minutes) * 60)
        t_str = str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    else:
        seconds = (t/60.0 - minutes) * 60
        t_str = str(seconds) + 's'

        
    return t_str







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
def plot_prediction(ID, vals, scale, soln_dir, mesh_dir, Model=0, CV=1, Rm_Outliers=False, Filter=True, Plot_Error=True, SHOW_PLOT=True, default_res=128, refine=1, slice_plot=False, use_hires=False, save_plots=False, view_elev=30, view_angle=0, error_view_elev=30, error_view_angle=0, PLOT_ALL=False):

    #PLOT_ALL = False
    
    USE_LATEX = False
    
    if save_plots:
        USE_LATEX = False
        plot_file = PdfPages('Figures/Predictions_' + str(ID) + '.pdf')
        dpi = 1000
        SHOW_PLOT = False
        linewidth = 0.01

        if USE_LATEX:
            fontsize1 = 18
            fontsize2 = 24
            fontsize3 = 14
        else:
            fontsize1 = 16
            fontsize2 = 18
            fontsize3 = 12
            
        if USE_LATEX:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            #plt.rc('text', usetex=True) #rc('text', usetex=True)
            #plt.rc('font', family='serif')
        #plt.locator_params(axis='y', nbins=4)
        #plt.locator_params(axis='x', nbins=4)

        tick_count = 5

        def get_ticks(xmin, xmax, ymin, ymax):
            xeps = 0.15*(xmax-xmin)
            yeps = 0.15*(ymax-ymin)
            xticks = np.linspace(xmin+xeps, xmax-xeps, tick_count)
            xticks = np.around(xticks, decimals=1)
            yticks = np.linspace(ymin+yeps, ymax-yeps, tick_count)
            yticks = np.around(yticks, decimals=1)

            #xticks = []
            #yticks = []
            xtick_labels = ['']*len(xticks)
            ytick_labels = ['']*len(yticks)
            return xticks, yticks, xtick_labels, ytick_labels

        #xticks_labels = [r"$\mathrm{-0.8}$", r"$\mathrm{-0.4}$", r"$\mathrm{0.0}$", r"$\mathrm{0.4}$", r"$\mathrm{0.8}$"]
        #yticks = [-.8, -.4, 0., .4, .8]
        #yticks_labels = [r"$\mathrm{-0.8}$", r"$\mathrm{-0.4}$", r"$\mathrm{0.0}$", r"$\mathrm{0.4}$", r"$\mathrm{0.8}$"]
    else:
        linewidth = 0.1
        fontsize1 = 24
        fontsize2 = 20
        fontsize3 = 12

    # Specify padding for axis tick labels
    label_padding = 10.5
        
    mpl.style.use('classic')
    #mpl.style.use('bmh')
    #['seaborn-bright', 'seaborn-deep', 'bmh', '_classic_test', 'seaborn', 'ggplot', 'seaborn-notebook', 'seaborn-whitegrid', 'seaborn-poster', 'seaborn-white', 'dark_background', 'Solarize_Light2', 'fivethirtyeight', 'grayscale', 'seaborn-muted', 'seaborn-ticks', 'seaborn-dark-palette', 'classic', 'seaborn-paper', 'seaborn-dark', 'seaborn-colorblind', 'seaborn-darkgrid', 'tableau-colorblind10', 'seaborn-talk', 'seaborn-pastel', 'fast']
    cmap = 'hot'

    cmap_2 = 'jet'
    #cmap_2 = 'viridis'
    
    #soln_file = '../Setup/Solutions/solution_0_' + str(ID) + '.npy'
    if use_hires:
        soln_label = 'hires_solution_' + str(ID)
    else:
        soln_label = 'solution_' + str(ID)
    soln_file = soln_dir + soln_label + '.npy'

    if use_hires:
        mesh_label = 'hires_mesh_' + str(ID)
    else:
        mesh_label = 'mesh_' + str(ID)
    mesh_file = mesh_dir + mesh_label + '.npy'
    mesh = np.load(mesh_file)
    
    # Retrieve solution
    soln = SCALING*np.load(soln_file)

    # Load network prediction
    pred = vals[0,:,:,0]
    stds = scale[0,:,:,0]

    original_soln = soln
    original_pred = pred
    original_stds = stds

    if slice_plot:
        soln = soln[0:int(default_res/2),0:int(default_res/2)]
        pred = pred[0:int(default_res/2),0:int(default_res/2)]
        stds = stds[0:int(default_res/2),0:int(default_res/2)]

    # Resize true solution if necessary
    if not (pred.shape[0] == 128):
        soln = cv2.resize(soln, dsize=(pred.shape[0], pred.shape[1]), interpolation=cv2.INTER_CUBIC)
        refine = 1

    pred_vals, plot_X, plot_Y = preprocesser(pred, Rm_Outliers=False, Filter=True, refine=refine)
    std_vals, plot_X, plot_Y = preprocesser(stds, Rm_Outliers=False, Filter=True, refine=refine)
    soln_vals, plot_X, plot_Y = preprocesser(soln, Rm_Outliers=False, Filter=False, refine=refine)


    # Determine solution/prediction extrema
    soln_min = np.min(soln_vals)
    soln_max = np.max(soln_vals)
    pred_min = np.min(pred_vals)
    pred_max = np.max(pred_vals)

    z_min = np.min([pred_min, soln_min])
    z_max = np.max([pred_max, soln_max])
    epsilon = 0.05*(z_max - z_min)

    def l2_error(y_,y):
        resolution = y.shape[0]
        scaling = np.power(1.0/(resolution - 1.0),2)
        return scaling*np.sqrt(np.sum(np.power(y_ - y, 2)))

    def rel_l2_error(y_,y):
        return np.sqrt(np.sum(np.power(y_ - y, 2)))/np.sqrt(np.sum(np.power(y, 2)))

    def l1_error(y_,y):
        resolution = y.shape[0]
        scaling = np.power(1.0/(resolution - 1.0),2)
        return scaling*np.sum(np.abs(y_ - y))

    def rel_l1_error(y_,y):
        return np.sum(np.abs(y_ - y))/np.sum(np.abs(y))

    l2_e = l2_error(pred_vals, soln_vals)
    rl2_e = rel_l2_error(pred_vals, soln_vals)
    l1_e = l1_error(pred_vals, soln_vals)
    rl1_e = rel_l1_error(pred_vals, soln_vals)

    
    """ TRISURF ATTEMPT """
    plot_X_flat = []
    plot_Y_flat = []
    pred_vals_flat = []
    soln_vals_flat = []
    diff_vals_flat = []
    abs_diff_vals_flat = []
    std_vals_flat = []
    neg_std_vals_flat = []
    two_std_vals_flat = []
    neg_two_std_vals_flat = []
    pred_plus_stds_flat = []
    pred_minus_stds_flat = []
    pred_plus_two_stds_flat = []
    pred_minus_two_stds_flat = []
    R = mesh.shape[0]
    mesh_x_vals = np.linspace(-1.0,1.0,R)
    mesh_y_vals = np.linspace(-1.0,1.0,R)

    diff_vals_smooth = mean_filter(pred_vals - soln_vals)
    
    for i in range(0,R):
        for j in range(0,R):
            if mesh[i,j] > 0.0:
                plot_X_flat.append(mesh_x_vals[i])
                plot_Y_flat.append(mesh_y_vals[j])
                pred_vals_flat.append(pred_vals[i,j])
                soln_vals_flat.append(soln_vals[i,j])
                std_vals_flat.append(std_vals[i,j])
                neg_std_vals_flat.append(-std_vals[i,j])
                two_std_vals_flat.append(2.0*std_vals[i,j])
                neg_two_std_vals_flat.append(-2.0*std_vals[i,j])
                pred_plus_stds_flat.append(pred_vals[i,j] + std_vals[i,j])
                pred_minus_stds_flat.append(pred_vals[i,j] - std_vals[i,j])
                pred_plus_two_stds_flat.append(pred_vals[i,j] + 2.0*std_vals[i,j])
                pred_minus_two_stds_flat.append(pred_vals[i,j] - 2.0*std_vals[i,j])
                #diff_vals_flat.append(pred_vals[i,j] - soln_vals[i,j])
                #abs_diff_vals_flat.append(np.abs(pred_vals[i,j] - soln_vals[i,j]))
                diff_vals_flat.append(diff_vals_smooth[i,j])
                abs_diff_vals_flat.append(np.abs(diff_vals_smooth[i,j]))
                
    tri_fig = plt.figure()
    tri_ax1 = tri_fig.add_subplot(121, projection='3d')
    tri_ax1.plot_trisurf(plot_X_flat,plot_Y_flat, pred_vals_flat, cmap=cmap, linewidth=linewidth, antialiased=True)
    if SHOW_PLOT:
        #pred_title = r'$\mathrm{Prediction}$'
        pred_title = 'Prediction:    min = %.6f    max = %.6f'  %(pred_min, pred_max)
    elif USE_LATEX:
        pred_title = r'$\mathrm{Prediction}$'
    else:
        pred_title = 'Prediction'
    if save_plots:
        y_shift = 1.025
    else:
        y_shift = 1.05
    tri_ax1.set_title(pred_title, fontsize=fontsize2, y=y_shift)
    tri_ax1.set_zlim([z_min - epsilon, z_max + epsilon])
    
    if not save_plots:
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')

    if save_plots:
        #tri_ax1.set_xlim(left=,right=)
        #tri_ax1.set_ylim(bottom=,top=)
        xmin, xmax = tri_ax1.get_xlim()
        ymin, ymax = tri_ax1.get_ylim()
        xticks, yticks, xtick_labels, ytick_labels = get_ticks(xmin, xmax, ymin, ymax)
        plt.xticks(xticks, xtick_labels)
        plt.yticks(yticks, ytick_labels)
    
    tri_ax2 = tri_fig.add_subplot(122, projection='3d')
    tri_ax2.plot_trisurf(plot_X_flat,plot_Y_flat, soln_vals_flat, cmap=cmap, linewidth=linewidth, antialiased=True)
    if SHOW_PLOT:
        #soln_title = r'$\mathrm{Solution}$'
        soln_title = 'Solution:    min = %.6f    max = %.6f'  %(soln_min, soln_max)
    elif USE_LATEX:
        soln_title = r'$\mathrm{Solution}$'
    else:
        soln_title = 'Solution'
    tri_ax2.set_title(soln_title, fontsize=fontsize2, y=y_shift)
    tri_ax2.set_zlim([z_min - epsilon, z_max + epsilon])

    if not save_plots:
        tri_ax1.tick_params(pad=label_padding)
        tri_ax2.tick_params(pad=label_padding)

        plt.xlabel('x - axis')
        plt.ylabel('y - axis')

    if save_plots:
        tri_ax1.tick_params(pad=label_padding)
        tri_ax2.tick_params(pad=label_padding)
        
        xmin, xmax = tri_ax2.get_xlim()
        ymin, ymax = tri_ax2.get_ylim()
        xticks, yticks, xtick_labels, ytick_labels = get_ticks(xmin, xmax, ymin, ymax)
        plt.xticks(xticks, xtick_labels)
        plt.yticks(yticks, ytick_labels)


    if SHOW_PLOT:
        tri_fig.suptitle('L^1: %.5f        L^2: %.5f\nL^1 Rel: %.5f     L^2 Rel: %.5f' %(l1_e,l2_e,rl1_e,rl2_e), fontsize=fontsize1)
    else:
        if not save_plots:
            tri_fig.suptitle('L^1: %.5f   |   L^2: %.5f\nL^1 Rel: %.5f   |   L^2 Rel: %.5f' %(l1_e,l2_e,rl1_e,rl2_e), fontsize=fontsize3, verticalalignment='bottom', y=0.0)

    # Bind axes for comparison
    def tri_on_move(event):
        if event.inaxes == tri_ax1:
            if tri_ax1.button_pressed in tri_ax1._rotate_btn:
                tri_ax2.view_init(elev=tri_ax1.elev, azim=tri_ax1.azim)
            elif tri_ax1.button_pressed in tri_ax1._zoom_btn:
                tri_ax2.set_xlim3d(tri_ax1.get_xlim3d())
                tri_ax2.set_ylim3d(tri_ax1.get_ylim3d())
                tri_ax2.set_zlim3d(tri_ax1.get_zlim3d())
        elif event.inaxes == tri_ax2:
            if tri_ax2.button_pressed in tri_ax2._rotate_btn:
                tri_ax1.view_init(elev=tri_ax2.elev, azim=tri_ax2.azim)
            elif tri_ax2.button_pressed in tri_ax2._zoom_btn:
                tri_ax1.set_xlim3d(tri_ax2.get_xlim3d())
                tri_ax1.set_ylim3d(tri_ax2.get_ylim3d())
                tri_ax1.set_zlim3d(tri_ax2.get_zlim3d())
        else:
            return
        tri_fig.canvas.draw_idle()
                
    tri_c1 = tri_fig.canvas.mpl_connect('motion_notify_event', tri_on_move)

    # Set initial view angles
    tri_ax1.view_init(view_elev, view_angle)
    tri_ax2.view_init(view_elev, view_angle)
    
    if save_plots:
        #tri_ax1.tick_params(axis='y', which='major', rotation=0)
        #tri_ax1.tick_params(axis='x', which='major', rotation=0)        
        plt.tight_layout()
        plot_file.savefig(dpi=dpi)
    
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()


    if Plot_Error:

        diff_min = np.min(diff_vals_flat)
        diff_max = np.max(diff_vals_flat)

        soln_max_abs = np.max(np.abs(soln_vals_flat))
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_trisurf(plot_X_flat,plot_Y_flat,diff_vals_flat, cmap=cmap, linewidth=linewidth, antialiased=True)
        ax.plot_trisurf(plot_X_flat,plot_Y_flat,two_std_vals_flat, cmap=cmap, linewidth=linewidth, antialiased=True, alpha=0.1)
        ax.plot_trisurf(plot_X_flat,plot_Y_flat,neg_two_std_vals_flat, cmap=cmap, linewidth=linewidth, antialiased=True, alpha=0.1)
        if save_plots:
            #ax.set_title('Error Min: %.4f      Error Max:  %.4f     [Soln Max:  %.4f]' %(diff_min, diff_max, soln_max_abs), fontsize=fontsize1, y=1.05)
            if USE_LATEX:
                ax.set_title(r"$\mathrm{Error\ with\ Predicted\ }$ $\pm$ $\mathrm{\ Two\ Standard\ Deviations}$", fontsize=fontsize1, y=1.05)
            else:
                ax.set_title("Error with Predicted Two Standard Deviations", fontsize=fontsize1, y=1.05)
                #fig2.suptitle("Error with Predicted Two Standard Deviations", fontsize=fontsize1, y=1.05)
        else:
            ax.set_title('Error Min: %.6f          Error Max:  %.6f      [Soln Max Abs:  %.6f]' %(diff_min, diff_max, soln_max_abs), fontsize=fontsize1)
        #ax.set_title('L^1: %.5f , L^1 Rel: %.5f , L^2: %.5f , L^2 Rel: %.5f' %(l1_e,rl1_e,l2_e,rl2_e))

        # Set initial view angles
        ax.view_init(error_view_elev, error_view_angle)
        
        ax.tick_params(pad=label_padding)
        
        if not save_plots:
            plt.xlabel('x - axis')
            plt.ylabel('y - axis')

        if save_plots:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xticks, yticks, xtick_labels, ytick_labels = get_ticks(xmin, xmax, ymin, ymax)
            plt.xticks(xticks, xtick_labels)
            plt.yticks(yticks, ytick_labels)

        if save_plots:
            plot_file.savefig(dpi=dpi)


        if PLOT_ALL:

            #"""
            fig3 = plt.figure()
            ax = fig3.add_subplot(111, projection='3d')

            #abs_diff = np.ma.masked_where(abs_diff > 0.001, abs_diff)
            #std_vals = np.ma.masked_where(std_vals > 0.001, std_vals)
            #colormap = 'YlOrRd'
            colormap = 'hot'
            ax.plot_trisurf(plot_X_flat,plot_Y_flat,abs_diff_vals_flat, cmap=colormap, linewidth=linewidth, antialiased=True)
            ax.plot_trisurf(plot_X_flat,plot_Y_flat,two_std_vals_flat, cmap=colormap, linewidth=linewidth, antialiased=True, alpha=0.1)
            rel_abs_error = np.max(abs_diff_vals_flat)/np.max(np.abs(soln_vals_flat))
            if save_plots:
                if USE_LATEX:
                    ax.set_title(r'$\mathrm{Error\ (Absolute\ Value)\ and\ Predicted\ Two\ Standard\ Deviations}$', fontsize=fontsize1, y=1.075)
                else:
                    ax.set_title('Error (Absolute Value) and Predicted Two Standard Deviations', fontsize=fontsize1, y=1.075)
            else:
                ax.set_title('Error (Absolute Value) and Two Standard Deviation Prediction  [Max Relative Error:  %.5f]'  %(rel_abs_error), fontsize=fontsize1)
            #ax.set_title('L^1: %.5f , L^1 Rel: %.5f , L^2: %.5f , L^2 Rel: %.5f' %(l1_e,rl1_e,l2_e,rl2_e))

            ax.tick_params(pad=label_padding)

            if not save_plots:
                plt.xlabel('x - axis')
                plt.ylabel('y - axis')

            if save_plots:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                xticks, yticks, xtick_labels, ytick_labels = get_ticks(xmin, xmax, ymin, ymax)
                plt.xticks(xticks, xtick_labels)
                plt.yticks(yticks, ytick_labels)
            #"""

            if save_plots:
                plot_file.savefig(dpi=dpi)

            #print(mpl.style.available)
            #mpl.style.use('bmh')


            # Determine solution/prediction extrema
            soln_min = np.min(soln_vals_flat)
            soln_max = np.max(soln_vals_flat)
            pred_min = np.min(pred_vals_flat)
            pred_max = np.max(pred_vals_flat)

            z_min = np.min([pred_min, soln_min])
            z_max = np.max([pred_max, soln_max])
            epsilon = 0.05*(z_max - z_min)
            #epsilon = 0.0


            #cmap_2.set_bad('white',1.)

            fig4, axes = plt.subplots(nrows=1, ncols=2)
            if save_plots:
                if USE_LATEX:
                    title_1 = r'$\mathrm{True\ Solution}$'
                    title_2 = r'$\mathrm{Network\ Prediction}$'
                else:
                    title_1 = 'True Solution'
                    title_2 = 'Network Prediction'
            else:
                title_1 = 'True Solution'
                title_2 = 'Network Prediction'
            for ax, vals, title in zip(*[axes.flat,[soln_vals, pred_vals],[title_1,title_2]]):
                vals = np.ma.array(vals, mask=(mesh == 0.0))
                im = ax.imshow(vals, vmin=z_min-epsilon, vmax=z_max+epsilon, cmap=cmap_2, interpolation='nearest')
                # Turn off tick labels
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                y_shift = 1.05
                ax.set_title(title, fontsize=fontsize1, y=y_shift)

            #fig4.suptitle('True Solution / Network Prediction', fontsize=fontsize1)

            ax.tick_params(pad=label_padding)

            if save_plots:
                fig4.subplots_adjust(bottom=0.2)
                cbar_ax = fig4.add_axes([0.125, 0.2, 0.775, 0.05])  # [ Left, Bottom, W, H ]
                fig4.colorbar(im, cax=cbar_ax, orientation='horizontal')
            else:
                fig4.subplots_adjust(bottom=0.2)
                cbar_ax = fig4.add_axes([0.125, 0.1, 0.775, 0.05])  # [ Left, Bottom, W, H ]
                fig4.colorbar(im, cax=cbar_ax, orientation='horizontal')

            if save_plots:
                plot_file.savefig(dpi=dpi)


            std_vals_2 = 2.0*std_vals
            diff_vals = np.abs(pred_vals - soln_vals)
            diff_min = np.min(diff_vals)
            diff_max = np.max(diff_vals)
            std_min = np.min(std_vals_2)
            std_max = np.max(std_vals_2)

            z_min = np.min([std_min, diff_min])
            z_max = np.max([std_max, diff_max])
            #epsilon = 0.05*(z_max - z_min)
            epsilon = 0.0

            fig5, axes = plt.subplots(nrows=1, ncols=3)
            if save_plots:
                if USE_LATEX:
                    title_1 = r'$\mathrm{Error\ (Absolute\ Value)}$'
                    title_2 = r'$\mathrm{One\ Standard\ Deviation}$'
                    title_3 = r'$\mathrm{Two\ Standard\ Deviations}$'
                else:
                    title_1 = 'Error (Absolute Value)'
                    title_2 = 'One Standard Deviation'
                    title_3 = 'Two Standard Deviations'
            else:
                title_1 = 'Error (Absolute Value)'
                title_2 = 'One Standard Deviation'
                title_3 = 'Two Standard Deviations'
            for ax, vals, title in zip(*[axes.flat,[diff_vals, std_vals, std_vals_2], [title_1, title_2, title_3]]):
                vals = np.ma.array(vals, mask=(mesh == 0.0))
                im = ax.imshow(vals, vmin=z_min, vmax=z_max+epsilon, cmap=cmap_2)
                # Turn off tick labels
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                y_shift = 1.05
                if save_plots:
                    ax.set_title(title, fontsize=fontsize3, y=y_shift)
                else:
                    ax.set_title(title, fontsize=fontsize1, y=y_shift)

            ax.tick_params(pad=label_padding)

            #fig5.suptitle('Error / 1*Standard Deviation / 2*Standard Deviation', fontsize=fontsize1)
            if save_plots:
                fig5.subplots_adjust(bottom=0.2)
                #cbar_ax = fig5.add_axes([0.125, 0.1, 0.775, 0.05])  # [ Left, Bottom, W, H ]
                cbar_ax = fig5.add_axes([0.125, 0.3, 0.775, 0.05])  # [ Left, Bottom, W, H ]
                fig5.colorbar(im, cax=cbar_ax, orientation='horizontal')
            else:
                fig5.subplots_adjust(bottom=0.2)
                #cbar_ax = fig5.add_axes([0.125, 0.1, 0.775, 0.05])  # [ Left, Bottom, W, H ]
                cbar_ax = fig5.add_axes([0.125, 0.2, 0.775, 0.05])  # [ Left, Bottom, W, H ]
                fig5.colorbar(im, cax=cbar_ax, orientation='horizontal')

            if save_plots:
                plot_file.savefig(dpi=dpi)

            # SYSTEM 1
            #mng = plt.get_current_fig_manager()
            #mng.frame.Maximize(True)

            # SYSTEM 2
            #mng = plt.get_current_fig_manager()
            #mng.resize(*mng.window.maxsize())

            # SYSTEM 3
            #figManager = plt.get_current_fig_manager()
            #figManager.window.showMaximized()
        

    if SHOW_PLOT:
        plt.show()

    if save_plots:
        plot_file.close()







# Plots predictions with matplotlib
def old_plot_prediction(ID, vals, scale, soln_dir, mesh_dir, Model=0, CV=1, Rm_Outliers=False, Filter=True, Plot_Error=True, SHOW_PLOT=True, default_res=128, refine=1, slice_plot=False, use_hires=False):

    mpl.style.use('classic')
    #mpl.style.use('bmh')
    #['seaborn-bright', 'seaborn-deep', 'bmh', '_classic_test', 'seaborn', 'ggplot', 'seaborn-notebook', 'seaborn-whitegrid', 'seaborn-poster', 'seaborn-white', 'dark_background', 'Solarize_Light2', 'fivethirtyeight', 'grayscale', 'seaborn-muted', 'seaborn-ticks', 'seaborn-dark-palette', 'classic', 'seaborn-paper', 'seaborn-dark', 'seaborn-colorblind', 'seaborn-darkgrid', 'tableau-colorblind10', 'seaborn-talk', 'seaborn-pastel', 'fast']
    cmap = 'hot'

    cmap_2 = 'jet'
    #cmap_2 = 'viridis'
    
    #soln_file = '../Setup/Solutions/solution_0_' + str(ID) + '.npy'
    if use_hires:
        soln_label = 'hires_solution_' + str(ID)
    else:
        soln_label = 'solution_' + str(ID)
    soln_file = soln_dir + soln_label + '.npy'

    if use_hires:
        mesh_label = 'hires_mesh_' + str(ID)
    else:
        mesh_label = 'mesh_' + str(ID)
    mesh_file = mesh_dir + mesh_label + '.npy'
    mesh = np.load(mesh_file)
    
    # Retrieve solution
    soln = SCALING*np.load(soln_file)

    # Load network prediction
    pred = vals[0,:,:,0]
    stds = scale[0,:,:,0]
    #print(np.min(stds))
    #print(np.max(stds))

    original_soln = soln
    original_pred = pred
    original_stds = stds

    if slice_plot:
        soln = soln[0:int(default_res/2),0:int(default_res/2)]
        pred = pred[0:int(default_res/2),0:int(default_res/2)]
        stds = stds[0:int(default_res/2),0:int(default_res/2)]

    # Resize true solution if necessary
    if not (pred.shape[0] == 128):
        soln = cv2.resize(soln, dsize=(pred.shape[0], pred.shape[1]), interpolation=cv2.INTER_CUBIC)
        refine = 1

    pred_vals, plot_X, plot_Y = preprocesser(pred, Rm_Outliers=False, Filter=True, refine=refine)
    std_vals, plot_X, plot_Y = preprocesser(stds, Rm_Outliers=False, Filter=True, refine=refine)
    soln_vals, plot_X, plot_Y = preprocesser(soln, Rm_Outliers=False, Filter=False, refine=refine)


    # Determine solution/prediction extrema
    soln_min = np.min(soln_vals)
    soln_max = np.max(soln_vals)
    pred_min = np.min(pred_vals)
    pred_max = np.max(pred_vals)

    z_min = np.min([pred_min, soln_min])
    z_max = np.max([pred_max, soln_max])
    epsilon = 0.05*(z_max - z_min)

    def l2_error(y_,y):
        resolution = y.shape[0]
        scaling = np.power(1.0/(resolution - 1.0),2)
        return scaling*np.sqrt(np.sum(np.power(y_ - y, 2)))

    def rel_l2_error(y_,y):
        return np.sqrt(np.sum(np.power(y_ - y, 2)))/np.sqrt(np.sum(np.power(y, 2)))

    def l1_error(y_,y):
        resolution = y.shape[0]
        scaling = np.power(1.0/(resolution - 1.0),2)
        return scaling*np.sum(np.abs(y_ - y))

    def rel_l1_error(y_,y):
        return np.sum(np.abs(y_ - y))/np.sum(np.abs(y))

    l2_e = l2_error(pred_vals, soln_vals)
    rl2_e = rel_l2_error(pred_vals, soln_vals)
    l1_e = l1_error(pred_vals, soln_vals)
    rl1_e = rel_l1_error(pred_vals, soln_vals)

    
    """ TRISURF ATTEMPT """
    plot_X_flat = []
    plot_Y_flat = []
    pred_vals_flat = []
    soln_vals_flat = []
    R = mesh.shape[0]
    mesh_x_vals = np.linspace(-1.0,1.0,R)
    mesh_y_vals = np.linspace(-1.0,1.0,R)
    for i in range(0,R):
        for j in range(0,R):
            if mesh[i,j] > 0.0:
                plot_X_flat.append(mesh_x_vals[i])
                plot_Y_flat.append(mesh_y_vals[j])
                pred_vals_flat.append(pred_vals[i,j])
                soln_vals_flat.append(soln_vals[i,j])
                
    tri_fig = plt.figure()
    tri_ax1 = tri_fig.add_subplot(121, projection='3d')
    tri_ax1.plot_trisurf(plot_X_flat,plot_Y_flat, pred_vals_flat, cmap=cmap, linewidth=linewidth, antialiased=True)
    tri_ax2 = tri_fig.add_subplot(122, projection='3d')
    tri_ax2.plot_trisurf(plot_X_flat,plot_Y_flat, soln_vals_flat, cmap=cmap, linewidth=linewidth, antialiased=True)


    # Bind axes for comparison
    def tri_on_move(event):
        if event.inaxes == tri_ax1:
            if tri_ax1.button_pressed in tri_ax1._rotate_btn:
                tri_ax2.view_init(elev=tri_ax1.elev, azim=tri_ax1.azim)
            elif tri_ax1.button_pressed in tri_ax1._zoom_btn:
                tri_ax2.set_xlim3d(tri_ax1.get_xlim3d())
                tri_ax2.set_ylim3d(tri_ax1.get_ylim3d())
                tri_ax2.set_zlim3d(tri_ax1.get_zlim3d())
        elif event.inaxes == tri_ax2:
            if tri_ax2.button_pressed in tri_ax2._rotate_btn:
                tri_ax1.view_init(elev=tri_ax2.elev, azim=tri_ax2.azim)
            elif tri_ax2.button_pressed in tri_ax2._zoom_btn:
                tri_ax1.set_xlim3d(tri_ax2.get_xlim3d())
                tri_ax1.set_ylim3d(tri_ax2.get_ylim3d())
                tri_ax1.set_zlim3d(tri_ax2.get_zlim3d())
        else:
            return
        tri_fig.canvas.draw_idle()
                
    tri_c1 = tri_fig.canvas.mpl_connect('motion_notify_event', tri_on_move)


    

    """ ORIGINAL """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(plot_X,plot_Y, pred_vals, cmap=cmap)
    PLOT_STDS = False
    if PLOT_STDS:
        ax1.plot_surface(plot_X,plot_Y, pred_vals+std_vals, alpha=0.1, cmap=cmap)
        ax1.plot_surface(plot_X,plot_Y, pred_vals-std_vals, alpha=0.1, cmap=cmap)
    if SHOW_PLOT:
        pred_title = 'Prediction:    min = %.6f    max = %.6f'  %(pred_min, pred_max)
    else:
        pred_title = 'Prediction'
    #y_shift = 1.1
    y_shift = 1.05
    ax1.set_title(pred_title, fontsize=fontsize2, y=y_shift)
    ax1.set_zlim([z_min - epsilon, z_max + epsilon])
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(plot_X,plot_Y, soln_vals, cmap=cmap)
    if SHOW_PLOT:
        soln_title = 'Solution:    min = %.6f    max = %.6f'  %(soln_min, soln_max)
    else:
        soln_title = 'Solution'
    ax2.set_title(soln_title, fontsize=fontsize2, y=y_shift)
    ax2.set_zlim([z_min - epsilon, z_max + epsilon])
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')

    if SHOW_PLOT:
        fig.suptitle('L^1: %.5f        L^2: %.5f\nL^1 Rel: %.5f     L^2 Rel: %.5f' %(l1_e,l2_e,rl1_e,rl2_e), fontsize=fontsize1)
    else:
        fig.suptitle('L^1: %.5f   |   L^2: %.5f\nL^1 Rel: %.5f   |   L^2 Rel: %.5f' %(l1_e,l2_e,rl1_e,rl2_e), fontsize=fontsize3, verticalalignment='bottom', y=0.0)


    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()


    if Plot_Error:
        diff = soln_vals - pred_vals
        diff_min = np.min(diff)
        diff_max = np.max(diff)

        soln_max_abs = np.max(np.abs(soln_vals))
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(plot_X,plot_Y,diff, cmap=cmap)
        ax.plot_surface(plot_X,plot_Y,std_vals, alpha=0.1, cmap=cmap)
        ax.plot_surface(plot_X,plot_Y,-std_vals, alpha=0.1, cmap=cmap)
        ax.set_title('Error Min: %.6f          Error Max:  %.6f      [Soln Max Abs:  %.6f]' %(diff_min, diff_max, soln_max_abs), fontsize=fontsize1)
        #ax.set_title('L^1: %.5f , L^1 Rel: %.5f , L^2: %.5f , L^2 Rel: %.5f' %(l1_e,rl1_e,l2_e,rl2_e))
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')


        #"""
        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection='3d')
        abs_diff = np.abs(diff)
        #abs_diff = np.ma.masked_where(abs_diff > 0.001, abs_diff)
        #std_vals = np.ma.masked_where(std_vals > 0.001, std_vals)
        #colormap = 'YlOrRd'
        colormap = 'hot'
        ax.plot_surface(plot_X,plot_Y,abs_diff, cmap=colormap)
        ax.plot_surface(plot_X,plot_Y,2*std_vals, cmap=colormap, alpha=0.1)
        rel_abs_error = np.max(abs_diff)/np.max(np.abs(soln_vals))
        #ax.set_title('Error (Absolute Value) and Two Standard Deviation Prediction', fontsize=fontsize1)
        ax.set_title('Error (Absolute Value) and Two Standard Deviation Prediction  [Max Relative Error:  %.5f]'  %(rel_abs_error), fontsize=fontsize1)
        #ax.set_title('L^1: %.5f , L^1 Rel: %.5f , L^2: %.5f , L^2 Rel: %.5f' %(l1_e,rl1_e,l2_e,rl2_e))
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        #"""
        
        """
        fig4 = plt.figure()
        ax = fig4.add_subplot(121)
        abs_diff = np.abs(original_soln - original_pred)
        std_max = np.min(original_stds)
        diff_max = np.max(abs_diff)
        max_val = np.max([std_max, diff_max])
        plt.imshow(abs_diff/max_val, cmap='hot')
        ax = fig4.add_subplot(122)
        plt.imshow(original_stds/max_val, cmap='hot')
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        """


        #print(mpl.style.available)
        #mpl.style.use('bmh')
        
        # Determine solution/prediction extrema
        soln_min = np.min(soln_vals)
        soln_max = np.max(soln_vals)
        pred_min = np.min(pred_vals)
        pred_max = np.max(pred_vals)

        z_min = np.min([pred_min, soln_min])
        z_max = np.max([pred_max, soln_max])
        epsilon = 0.05*(z_max - z_min)
        #epsilon = 0.0

        fig4, axes = plt.subplots(nrows=1, ncols=2)
        title_1 = 'True Solution'
        title_2 = 'Network Prediction'
        for ax, vals, title in zip(*[axes.flat,[soln_vals, pred_vals],[title_1,title_2]]):
            im = ax.imshow(vals, vmin=z_min-epsilon, vmax=z_max+epsilon, cmap=cmap_2)
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            y_shift = 1.05
            ax.set_title(title, fontsize=fontsize1, y=y_shift)

        #fig4.suptitle('True Solution / Network Prediction', fontsize=fontsize1)
        
        fig4.subplots_adjust(bottom=0.2)
        cbar_ax = fig4.add_axes([0.125, 0.1, 0.775, 0.05])  # [ Left, Bottom, W, H ]
        fig4.colorbar(im, cax=cbar_ax, orientation='horizontal')

        std_vals_2 = 2.0*std_vals
        diff_vals = np.abs(pred_vals - soln_vals)
        diff_min = np.min(diff_vals)
        diff_max = np.max(diff_vals)
        std_min = np.min(std_vals_2)
        std_max = np.max(std_vals_2)

        z_min = np.min([std_min, diff_min])
        z_max = np.max([std_max, diff_max])
        #epsilon = 0.05*(z_max - z_min)
        epsilon = 0.0
        
        fig5, axes = plt.subplots(nrows=1, ncols=3)
        title_1 = 'Error (Absolute Value)'
        title_2 = 'One Standard Deviation'
        title_3 = 'Two Standard Deviations'
        for ax, vals, title in zip(*[axes.flat,[diff_vals, std_vals, std_vals_2], [title_1, title_2, title_3]]):
            im = ax.imshow(vals, vmin=z_min, vmax=z_max+epsilon, cmap=cmap_2)
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            y_shift = 1.05
            ax.set_title(title, fontsize=fontsize1, y=y_shift)

        #fig5.suptitle('Error / 1*Standard Deviation / 2*Standard Deviation', fontsize=fontsize1)
        fig5.subplots_adjust(bottom=0.2)
        #cbar_ax = fig5.add_axes([0.125, 0.1, 0.775, 0.05])  # [ Left, Bottom, W, H ]
        cbar_ax = fig5.add_axes([0.125, 0.2, 0.775, 0.05])  # [ Left, Bottom, W, H ]
        fig5.colorbar(im, cax=cbar_ax, orientation='horizontal')

        # SYSTEM 1
        #mng = plt.get_current_fig_manager()
        #mng.frame.Maximize(True)
        
        # SYSTEM 2
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        
        # SYSTEM 3
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        
        
    # Bind axes for comparison
    def on_move(event):
        if event.inaxes == ax1:
            if ax1.button_pressed in ax1._rotate_btn:
                ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            elif ax1.button_pressed in ax1._zoom_btn:
                ax2.set_xlim3d(ax1.get_xlim3d())
                ax2.set_ylim3d(ax1.get_ylim3d())
                ax2.set_zlim3d(ax1.get_zlim3d())
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            elif ax2.button_pressed in ax2._zoom_btn:
                ax1.set_xlim3d(ax2.get_xlim3d())
                ax1.set_ylim3d(ax2.get_ylim3d())
                ax1.set_zlim3d(ax2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()
                
    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    if SHOW_PLOT:
        plt.show()




# Plots data functions with matplotlib
def plot_data(ID):
    data_file = './Setup/Data/data_' + str(ID) + '.npy'
    mesh_file = './Setup/Meshes/mesh_' + str(ID) + '.npy'

    # Load data and mesh arrays
    data = np.load(data_file)
    mesh = np.load(mesh_file)
    resolution = data.shape[0]

    #mpl.style.use('classic')
    #mpl.style.use('bmh')

    ## MPL STYLES
    #['seaborn-bright', 'seaborn-deep', 'bmh', '_classic_test', 'seaborn', 'ggplot', 'seaborn-notebook', 'seaborn-whitegrid', 'seaborn-poster', 'seaborn-white', 'dark_background', 'Solarize_Light2', 'fivethirtyeight', 'grayscale', 'seaborn-muted', 'seaborn-ticks', 'seaborn-dark-palette', 'classic', 'seaborn-paper', 'seaborn-dark', 'seaborn-colorblind', 'seaborn-darkgrid', 'tableau-colorblind10', 'seaborn-talk', 'seaborn-pastel', 'fast']

    ## COLOR MAPS
    #Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r

    mpl.style.use('seaborn-deep')
    cmap = 'jet'
    
    epsilon = 0.05
    y_min = np.min(data)
    y_max = np.max(data)

    fig, ax = plt.subplots()
    vals = np.ma.array(data, mask=(mesh == 0.0))
    im = ax.imshow(vals, vmin=y_min-epsilon, vmax=y_max+epsilon, cmap=cmap, interpolation='nearest')
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.show()
    
    """
    data_vals, plot_X, plot_Y = preprocesser(data, Rm_Outliers=False, Filter=False)
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
    """



# Plots data functions with matplotlib
def plot_soln(soln_ID):
    data_file = '../DATA/Solutions/solution_' + str(soln_ID) + '.npy'
        
    # Load data function
    data = SCALING*np.load(data_file)
    resolution = data.shape[0]

    data_vals, plot_X, plot_Y = preprocesser(data, Rm_Outliers=False, Filter=False)
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



    
# Run main() function when called directly
#if __name__ == '__main__':
#    plot_data(ID=1233)
