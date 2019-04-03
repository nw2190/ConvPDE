import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def main():
    ID = 4
    legend_entries = []
    tfrecords_files = 32
    data_per_file = int(np.floor(100000/tfrecords_files))

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    
    def smooth(vals, N=20):
        new_vals = vals.copy()
        for n in range(0,vals.size):
            padding = (vals.size - n) - N
            if padding < 0:
                window = [n+k for k in range(padding,vals.size-n)]
            else:
                window = [n+k for k in range(0,N)]
            window_vals = vals[window]
            new_vals[n] = np.mean(window_vals)
        return new_vals

    def remove_outliers(vals, N=20):
        new_vals = vals.copy()
        tol = 300.0
        start = 10
        end = None
        for n in range(start, vals.size-1):
            if vals[n+1] >= tol*vals[n]:
                start = n+1
                print(start)
                break
            
        for n in range(start, vals.size-1):
            if vals[n+1] <= vals[n]/tol:
                end = n+1
                print(end)
                break

        if end is not None:

            end = end + 15
            steps = end - start
            start_val = vals[start-1]
            end_val = vals[end]
            print(start_val)
            print(end_val)
                
            for k in range(0,steps+1):
                new_vals[start + k] = k/steps*end_val + (1.0 - k/steps)*start_val
                
        return new_vals

    color_count = 0
    
    for k in range(1,33):
        model_dir = "../Model_" + str(ID) + "-" + str(k) + "/"
        filename = "evaluation_losses.csv"

        
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

            original_losses = losses.copy()
            original_losses = remove_outliers(original_losses)
            
            ## Smooth out losses
            SMOOTH = True
            if SMOOTH:
                losses = smooth(losses)
                losses = remove_outliers(losses)

            #colors = ['tab:blue', 'tab:yellow', 'tab:green', 'tab:red', 'tab:purple']
            colors = ['C0', 'C1', 'C2', 'C3', 'C4']

            USE_LOG = True

            if USE_LOG:
                plt.semilogy(steps, original_losses, linewidth=3.0, color=colors[color_count], alpha=0.2, label=None)
                plt.semilogy(steps, losses, linewidth=3.5, color=colors[color_count], label='%d Training Points' %(data_per_file*k))
            else:
                plt.plot(steps, original_losses, linewidth=3.0, color=colors[color_count], alpha=0.2, label=None)
                plt.plot(steps, losses, linewidth=3.5, color=colors[color_count], label='%d Training Points' %(data_per_file*k))
            
            #plt.scatter(steps, losses, alpha=0.75, marker="s", edgecolor='black', s=100)
            ax = plt.gca()
            ax.set_yscale('log')
            #legend_entries.append('%d Training Points' %(data_per_file*k))
            color_count += 1


    # Plot parameters
    linewidth = 3
    titlesize = 24
    ylabelsize = 24
    xlabelsize = 24
    xticksize = 16
    yticksize = 16
    ylabelpad = 20
    xlabelpad = 20

    ax.tick_params(axis='x', labelsize=xticksize)     
    ax.tick_params(axis='y', labelsize=yticksize)
    
    ax.set_xlabel('Training Steps', fontsize=xlabelsize, labelpad=20)
    ax.set_ylabel(r'$L^2\,$ Absolute Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
            
    #ax.legend(legend_entries, fontsize=24)
    ax.legend(fontsize=24)
    plt.show()




    
# Run main() function when called directly
if __name__ == '__main__':
    main()
