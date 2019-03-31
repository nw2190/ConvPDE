import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import numpy as np
import csv
import os

import tensorflow as tf

# Plots predictions with matplotlib
def plot_prediction(step, ID, model_dir='./Model/', Rm_Outliers=False, Filter=True, Plot_Error=False):

    mpl.style.use('classic')

    pred_file = model_dir + 'predictions/' + str(step) + '/pred_' + str(ID) + '.npy'
    soln_file = model_dir + 'predictions/' + str(step) + '/soln_' + str(ID) + '.npy'
        
    # Retrieve prediction and solution
    pred = np.load(pred_file)
    soln = np.load(soln_file)

    pred_vals, plot_X, plot_Y = preprocesser(pred, Rm_Outliers=False, Filter=True)
    soln_vals, plot_X, plot_Y = preprocesser(soln, Rm_Outliers=False, Filter=True)

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
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(plot_X,plot_Y, pred_vals, cmap='hot')
    pred_title = 'Prediction:    min = %.6f    max = %.6f'  %(pred_min, pred_max)
    ax1.set_title(pred_title)
    ax1.set_zlim([z_min - epsilon, z_max + epsilon])
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(plot_X,plot_Y, soln_vals, cmap='hot')
    soln_title = 'Solution:    min = %.6f    max = %.6f'  %(soln_min, soln_max)
    ax2.set_title(soln_title)
    ax2.set_zlim([z_min - epsilon, z_max + epsilon])
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')

    fig.suptitle('L^1: %.5f        L^2: %.5f\nL^1 Rel: %.5f     L^2 Rel: %.5f' %(l1_e,l2_e,rl1_e,rl2_e), fontsize=24)


    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()


    if Plot_Error:
        diff = soln_vals - pred_vals
        diff_min = np.min(diff)
        diff_max = np.max(diff)
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(plot_X,plot_Y,diff, cmap='hot')
        ax.set_title('Error Min: %.6f          Error Max:  %.6f' %(diff_min,diff_max))
        #ax.set_title('L^1: %.5f , L^1 Rel: %.5f , L^2: %.5f , L^2 Rel: %.5f' %(l1_e,rl1_e,l2_e,rl2_e))
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')

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

    plt.show()



# Plots data functions with matplotlib
def plot_data_vals(vals, Rm_Outliers=False, Filter=True, Plot_Error=False):
    mpl.style.use('classic')

    # Load data function
    data = vals
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


    
def plot_data(step, ID, model_dir='./Model/', Rm_Outliers=False, Filter=True, Plot_Error=False):
    mpl.style.use('classic')
    data_file = model_dir + 'predictions/' + str(step) + '/data_' + str(ID) + '.npy'
        
    data = np.load(data_file)
    
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
    return data


def plot_mesh(step, ID, model_dir='./Model/', Rm_Outliers=False, Filter=True, Plot_Error=False):
    mpl.style.use('classic')
    mesh_file = model_dir + 'predictions/' + str(step) + '/mesh_' + str(ID) + '.npy'
    mesh = np.load(mesh_file)
    plt.imshow(mesh)
    plt.show()
    return mesh

def plot_soln(step, ID, model_dir='./Model/', Rm_Outliers=False, Filter=True, Plot_Error=False):
    mpl.style.use('classic')
    soln_file = model_dir + 'predictions/' + str(step) + '/soln_' + str(ID) + '.npy'
        
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


def test_rotations(DATA_DIR="Frozen/Setup/", ID=0):
    data_dir = os.path.join(DATA_DIR, "Data/")
    mesh_dir = os.path.join(DATA_DIR, "Meshes/")
    soln_dir = os.path.join(DATA_DIR, "Solutions/")
    data = np.load(os.path.join(data_dir, "data_" + str(ID) + ".npy"))
    mesh = np.load(os.path.join(mesh_dir, "mesh_" + str(ID) + ".npy"))
    soln = np.load(os.path.join(soln_dir, "solution_" + str(ID) + ".npy"))
    transformations = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]]

    data = np.expand_dims(data, 2)
    mesh = np.expand_dims(mesh, 2)
    soln = np.expand_dims(soln, 2)

    tdata = tf.placeholder(tf.float32, [None, None, 1])
    tmesh = tf.placeholder(tf.float32, [None, None, 1])
    tsoln = tf.placeholder(tf.float32, [None, None, 1])
    #ttransform = tf.placeholder(tf.int32, [2])
    trotation = tf.placeholder(tf.int32, ())
    tflip = tf.placeholder(tf.int32, ())
    
    def transform(tdata, tmesh, tsoln, rotation, flip):

        #[rotation, flip] = transformation
        #rotation = transformation[0]
        #flip = transformation[1]
        
        # Stacked data
        stacked = tf.stack([tdata, tmesh, tsoln], 0)

        # Rotate data
        stacked = tf.image.rot90(stacked, k=rotation)

        # Flip data
        #if flip == 1:
        #    stacked = tf.image.flip_left_right(stacked)
        true_fn = lambda: tf.image.flip_left_right(stacked)
        false_fn = lambda: stacked
        stacked = tf.cond(tf.math.equal(flip, 1), true_fn, false_fn)

        # Unstack data
        ttdata, ttmesh, ttsoln = tf.unstack(stacked)
        return ttdata, ttmesh, ttsoln
    
    #ttdata, ttmesh, ttsoln = transform(tdata, tmesh, tsoln, ttransform)
    ttdata, ttmesh, ttsoln = transform(tdata, tmesh, tsoln, trotation, tflip)

    with tf.Session() as sess:
        for t in transformations:
            print(t)
            #fd = {tdata: data, tmesh: mesh, tsoln: soln, ttransform: t}
            fd = {tdata: data, tmesh: mesh, tsoln: soln, trotation: t[0], tflip: t[1]}
            data_vals, mesh_vals, soln_vals = sess.run([ttdata, ttmesh, ttsoln], feed_dict=fd)
            fig, axes = plt.subplots(1,3)
            axes[0].imshow(mesh_vals[:,:,0])
            axes[1].imshow(data_vals[:,:,0])
            axes[2].imshow(soln_vals[:,:,0])
            plt.show()
            


def plot_model_data(step, model=4, files=32, ID=0):
    model_dir = "Model_" + str(model) + "-" + str(files) + "/predictions/"
    data_dir = os.path.join(model_dir, str(step) + "/")    
    mesh_file = os.path.join(data_dir, "mesh_" + str(ID) + ".npy")
    data_file = os.path.join(data_dir, "data_" + str(ID) + ".npy")
    pred_file = os.path.join(data_dir, "pred_" + str(ID) + ".npy")
    soln_file = os.path.join(data_dir, "soln_" + str(ID) + ".npy")
    msoln_file = os.path.join(data_dir, "msoln_" + str(ID) + ".npy")
    mesh = np.load(mesh_file)
    data = np.load(data_file)
    pred = np.load(pred_file)
    soln = np.load(soln_file)
    msoln = np.load(msoln_file)
    print(np.all(soln == msoln))
    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(mesh)
    axes[0,1].imshow(data)
    axes[1,0].imshow(pred)
    axes[1,1].imshow(soln)
    plt.show()


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



def plot_eval(ID=1):
    legend_entries = []
    tfrecords_files = 32
    data_per_file = int(np.floor(100000/tfrecords_files))
    for k in range(1,33):
        model_dir = "./Model_" + str(ID) + "-" + str(k) + "/"
        filename = "evaluation_losses.csv"


        def smooth(vals, window_len=12, window='hanning'):
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            # force window_len to be odd
            print(window_len)
            print(np.mod(window_len,2))
            print(np.mod(window_len,2) == 0)
            print( "1" if not True else "2" )
            window_len = (window_len if (np.mod(window_len,2) == 1) else window_len + 1)

            

            print(window_len)
            if window == 'flat':
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')
                
            s=np.r_[vals[window_len-1:0:-1],vals,vals[-1:-window_len:-1]]
            smooth_vals = np.convolve(w/w.sum(),s,mode='valid')

            new_vals = np.zeros([len(vals)])
            
            #print(vals.shape)
            #print(smooth_vals.shape)
            #print(new_vals.shape)
            #print(w.shape)

            offset = int(np.floor((len(smooth_vals) - len(vals))/2))

            new_vals = smooth_vals[offset:-offset]

            """
            for k in range(0,window_len-1):
                new_vals[k] = vals[k]
            for k in range(len(vals)-window_len+1,len(vals)):
                new_vals[k] = vals[k]
            
            #print(new_vals)
                
            new_vals[window_len-1:len(vals)-window_len] = smooth_vals
            """
            
            return new_vals
        
        if os.path.exists(model_dir):

            steps = []
            losses = []
            with open(os.path.join(model_dir, filename), "r") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csvreader:
                    step, loss = row
                    steps.append(step)
                    losses.append(loss)

            steps = np.array(steps).astype(np.float32)
            losses = np.array(losses).astype(np.float32)

            ## Smooth out losses
            SMOOTH = True
            if SMOOTH:
                losses = smooth(losses)
            
            plt.semilogy(steps, losses)
            plt.scatter(steps, losses, alpha=0.75, marker="s", edgecolor='black', s=100)
            ax = plt.gca()
            ax.set_yscale('log')
            legend_entries.append('%d Training Points' %(data_per_file*k))
    ax.legend(legend_entries, fontsize=24)
    plt.show()
    

def plot_eval(ID=1):
    legend_entries = []
    tfrecords_files = 32
    data_per_file = int(np.floor(100000/tfrecords_files))
    for k in range(1,33):
        model_dir = "./Model_" + str(ID) + "-" + str(k) + "/"
        filename = "evaluation_losses.csv"


        def smooth(vals, window_len=12, window='hanning'):
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            # force window_len to be odd
            print(window_len)
            print(np.mod(window_len,2))
            print(np.mod(window_len,2) == 0)
            print( "1" if not True else "2" )
            window_len = (window_len if (np.mod(window_len,2) == 1) else window_len + 1)

            

            print(window_len)
            if window == 'flat':
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')
                
            s=np.r_[vals[window_len-1:0:-1],vals,vals[-1:-window_len:-1]]
            smooth_vals = np.convolve(w/w.sum(),s,mode='valid')

            new_vals = np.zeros([len(vals)])
            
            #print(vals.shape)
            #print(smooth_vals.shape)
            #print(new_vals.shape)
            #print(w.shape)

            offset = int(np.floor((len(smooth_vals) - len(vals))/2))

            new_vals = smooth_vals[offset:-offset]

            """
            for k in range(0,window_len-1):
                new_vals[k] = vals[k]
            for k in range(len(vals)-window_len+1,len(vals)):
                new_vals[k] = vals[k]
            
            #print(new_vals)
                
            new_vals[window_len-1:len(vals)-window_len] = smooth_vals
            """
            
            return new_vals
        
        if os.path.exists(model_dir):

            steps = []
            losses = []
            with open(os.path.join(model_dir, filename), "r") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csvreader:
                    step, loss = row
                    steps.append(step)
                    losses.append(loss)

            steps = np.array(steps).astype(np.float32)
            losses = np.array(losses).astype(np.float32)

            ## Smooth out losses
            SMOOTH = True
            if SMOOTH:
                losses = smooth(losses)
            
            plt.semilogy(steps, losses)
            plt.scatter(steps, losses, alpha=0.75, marker="s", edgecolor='black', s=100)
            ax = plt.gca()
            ax.set_yscale('log')
            legend_entries.append('%d Training Points' %(data_per_file*k))
    ax.legend(legend_entries, fontsize=24)
    plt.show()
    

def plot_eval(ID=1):
    legend_entries = []
    tfrecords_files = 32
    data_per_file = int(np.floor(100000/tfrecords_files))
    for k in range(1,33):
        model_dir = "./Model_" + str(ID) + "-" + str(k) + "/"
        filename = "evaluation_losses.csv"


        def smooth(vals, window_len=12, window='hanning'):
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            # force window_len to be odd
            print(window_len)
            print(np.mod(window_len,2))
            print(np.mod(window_len,2) == 0)
            print( "1" if not True else "2" )
            window_len = (window_len if (np.mod(window_len,2) == 1) else window_len + 1)

            

            print(window_len)
            if window == 'flat':
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')
                
            s=np.r_[vals[window_len-1:0:-1],vals,vals[-1:-window_len:-1]]
            smooth_vals = np.convolve(w/w.sum(),s,mode='valid')

            new_vals = np.zeros([len(vals)])
            
            #print(vals.shape)
            #print(smooth_vals.shape)
            #print(new_vals.shape)
            #print(w.shape)

            offset = int(np.floor((len(smooth_vals) - len(vals))/2))

            new_vals = smooth_vals[offset:-offset]

            """
            for k in range(0,window_len-1):
                new_vals[k] = vals[k]
            for k in range(len(vals)-window_len+1,len(vals)):
                new_vals[k] = vals[k]
            
            #print(new_vals)
                
            new_vals[window_len-1:len(vals)-window_len] = smooth_vals
            """
            
            return new_vals
        
        if os.path.exists(model_dir):

            steps = []
            losses = []
            with open(os.path.join(model_dir, filename), "r") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csvreader:
                    step, loss = row
                    steps.append(step)
                    losses.append(loss)

            steps = np.array(steps).astype(np.float32)
            losses = np.array(losses).astype(np.float32)

            ## Smooth out losses
            SMOOTH = True
            if SMOOTH:
                losses = smooth(losses)
            
            plt.semilogy(steps, losses)
            plt.scatter(steps, losses, alpha=0.75, marker="s", edgecolor='black', s=100)
            ax = plt.gca()
            ax.set_yscale('log')
            legend_entries.append('%d Training Points' %(data_per_file*k))
    ax.legend(legend_entries, fontsize=24)
    plt.show()
    

def plot_eval(ID=1):
    legend_entries = []
    tfrecords_files = 32
    data_per_file = int(np.floor(100000/tfrecords_files))
    for k in range(1,33):
        model_dir = "./Model_" + str(ID) + "-" + str(k) + "/"
        filename = "evaluation_losses.csv"


        def smooth(vals, window_len=12, window='hanning'):
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            # force window_len to be odd
            print(window_len)
            print(np.mod(window_len,2))
            print(np.mod(window_len,2) == 0)
            print( "1" if not True else "2" )
            window_len = (window_len if (np.mod(window_len,2) == 1) else window_len + 1)

            

            print(window_len)
            if window == 'flat':
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')
                
            s=np.r_[vals[window_len-1:0:-1],vals,vals[-1:-window_len:-1]]
            smooth_vals = np.convolve(w/w.sum(),s,mode='valid')

            new_vals = np.zeros([len(vals)])
            
            #print(vals.shape)
            #print(smooth_vals.shape)
            #print(new_vals.shape)
            #print(w.shape)

            offset = int(np.floor((len(smooth_vals) - len(vals))/2))

            new_vals = smooth_vals[offset:-offset]

            """
            for k in range(0,window_len-1):
                new_vals[k] = vals[k]
            for k in range(len(vals)-window_len+1,len(vals)):
                new_vals[k] = vals[k]
            
            #print(new_vals)
                
            new_vals[window_len-1:len(vals)-window_len] = smooth_vals
            """
            
            return new_vals
        
        if os.path.exists(model_dir):

            steps = []
            losses = []
            with open(os.path.join(model_dir, filename), "r") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csvreader:
                    step, loss = row
                    steps.append(step)
                    losses.append(loss)

            steps = np.array(steps).astype(np.float32)
            losses = np.array(losses).astype(np.float32)

            ## Smooth out losses
            SMOOTH = True
            if SMOOTH:
                losses = smooth(losses)
            
            plt.semilogy(steps, losses)
            plt.scatter(steps, losses, alpha=0.75, marker="s", edgecolor='black', s=100)
            ax = plt.gca()
            ax.set_yscale('log')
            legend_entries.append('%d Training Points' %(data_per_file*k))
    ax.legend(legend_entries, fontsize=24)
    plt.show()
    

def plot_eval(ID=1):
    legend_entries = []
    tfrecords_files = 32
    data_per_file = int(np.floor(100000/tfrecords_files))
    for k in range(1,33):
        model_dir = "./Model_" + str(ID) + "-" + str(k) + "/"
        filename = "evaluation_losses.csv"


        def smooth(vals, window_len=12, window='hanning'):
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            # force window_len to be odd
            print(window_len)
            print(np.mod(window_len,2))
            print(np.mod(window_len,2) == 0)
            print( "1" if not True else "2" )
            window_len = (window_len if (np.mod(window_len,2) == 1) else window_len + 1)

            

            print(window_len)
            if window == 'flat':
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')
                
            s=np.r_[vals[window_len-1:0:-1],vals,vals[-1:-window_len:-1]]
            smooth_vals = np.convolve(w/w.sum(),s,mode='valid')

            new_vals = np.zeros([len(vals)])
            
            #print(vals.shape)
            #print(smooth_vals.shape)
            #print(new_vals.shape)
            #print(w.shape)

            offset = int(np.floor((len(smooth_vals) - len(vals))/2))

            new_vals = smooth_vals[offset:-offset]

            """
            for k in range(0,window_len-1):
                new_vals[k] = vals[k]
            for k in range(len(vals)-window_len+1,len(vals)):
                new_vals[k] = vals[k]
            
            #print(new_vals)
                
            new_vals[window_len-1:len(vals)-window_len] = smooth_vals
            """
            
            return new_vals
        
        if os.path.exists(model_dir):

            steps = []
            losses = []
            with open(os.path.join(model_dir, filename), "r") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csvreader:
                    step, loss = row
                    steps.append(step)
                    losses.append(loss)

            steps = np.array(steps).astype(np.float32)
            losses = np.array(losses).astype(np.float32)

            ## Smooth out losses
            SMOOTH = True
            if SMOOTH:
                losses = smooth(losses)
            
            plt.semilogy(steps, losses)
            plt.scatter(steps, losses, alpha=0.75, marker="s", edgecolor='black', s=100)
            ax = plt.gca()
            ax.set_yscale('log')
            legend_entries.append('%d Training Points' %(data_per_file*k))
    ax.legend(legend_entries, fontsize=24)
    plt.show()

