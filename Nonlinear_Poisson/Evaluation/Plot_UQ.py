import numpy as np
import matplotlib.pyplot as plt
import csv
import os

import scipy.special as special

def main():

    legend_entries = []

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


    filename = "UQ_Bounds_NEW.csv"

    stds = []
    t_uqs = []
    v_uqs = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            std, t_uq, t_std, v_uq, v_std = row
            stds.append(std)
            t_uqs.append(t_uq)
            v_uqs.append(v_uq)
            
    stds = np.array(stds).astype(np.float32)
    t_uqs = 100*np.array(t_uqs).astype(np.float32)
    v_uqs = 100*np.array(v_uqs).astype(np.float32)

    """
    original_losses = losses.copy()
    original_losses = remove_outliers(original_losses)
    ## Smooth out losses
    SMOOTH = True
    if SMOOTH:
        losses = smooth(losses)
        losses = remove_outliers(losses)
    """

    plt.plot(stds, t_uqs, linewidth=3.0, color="C0", label="Training Dataset")
    plt.plot(stds, v_uqs, linewidth=3.0, color="C1", label='Validation Dataset')

    alpha = 0.1
    y1 = np.zeros(t_uqs.shape)
    plt.fill_between(stds, y1, t_uqs, where=t_uqs >= y1, facecolor="C0", alpha=alpha, interpolate=True, label=None)#, hatch="X", edgecolor="white")
    
    alpha = 0.1
    y1 = np.zeros(v_uqs.shape)
    plt.fill_between(stds, y1, v_uqs, where=v_uqs >= y1, facecolor="C1", alpha=alpha, interpolate=True, label=None)

    
    #plt.scatter(steps, losses, alpha=0.75, marker="s", edgecolor='black', s=100)
    #ax = plt.gca()
    #ax.set_yscale('log')
    #legend_entries.append('%d Training Points' %(data_per_file*k))

    ax = plt.gca()

    # Plot parameters
    linewidth = 3
    titlesize = 24
    ylabelsize = 24
    xlabelsize = 24
    xticksize = 18
    yticksize = 16
    ylabelpad = 20
    xlabelpad = 20

    ax.tick_params(axis='x', labelsize=xticksize)     
    ax.tick_params(axis='y', labelsize=yticksize)
    
    ax.set_xlabel('Standard Deviations', fontsize=xlabelsize, labelpad=20)
    ax.set_ylabel('Percentage of Dataset', color='k', fontsize=ylabelsize, labelpad=ylabelpad)


    ticks = [n*20 for n in [0,1,2,3,4,5]]
    labels = tuple(["{0:}%".format(n) for n in ticks])
    plt.yticks(ticks, labels, fontsize=yticksize)

    ticks = [n*0.5 for n in [1,2,3,4,5,6]]
    labels = tuple([r"{0:}$\sigma$".format(n).replace(".0"," ",1) for n in ticks])
    plt.xticks(ticks, labels, fontsize=xticksize)
    

    training_vals = [0.73471046, 0.96861186, 0.99774161]
    validation_vals = [0.68202384, 0.94649767, 0.99434461]

    #alpha=0.85
    alpha=1.0
    
    textstr = '\n'.join((
        #r"3$\sigma$",
        r"Training:    99.77",
        r"Validation: 99.43"))

    props = dict(boxstyle='round', facecolor='white', alpha=alpha)
    ax.text(0.8, 1.04, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)


    textstr = '\n'.join((
        #r"3$\sigma$",
        r"Training:    96.86",
        r"Validation: 94.65"))

    props = dict(boxstyle='round', facecolor='white', alpha=alpha)
    #ax.text(0.635, 0.855, textstr, transform=ax.transAxes, fontsize=18,
    ax.text(0.49, 1.015, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)

    textstr = '\n'.join((
        #r"3$\sigma$",
        r"Training:    73.47",
        r"Validation: 68.20"))

    props = dict(boxstyle='round', facecolor='white', alpha=alpha)
    #ax.text(0.345, 0.625, textstr, transform=ax.transAxes, fontsize=18,
    ax.text(0.17, 0.8, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)


    sigma = 1.0
    #func = lambda x: 2.*100.*(0.5*(1 + special.erf(x/(sigma*np.sqrt(2)))) - 0.5)
    func = lambda x: 100.*special.erf(x/(sigma*np.sqrt(2)))
    #plt.plot(stds, func(stds), 'r--', linewidth=2, alpha=0.5, zorder=0, label=r"$\operatorname{erf}(x/\sqrt{2})$")
    plt.plot(stds, func(stds), 'r--', linewidth=2, alpha=0.5, zorder=0, label=r"$P(|\mathcal{N}(0,\sigma)| \, < \, x)$")
    
    # Add dots at 1/2/3 standard deviation points
    xvals = [1,2,3]
    coords = [73.471046, 96.861186, 99.774161]
    plt.scatter(xvals,coords, color='C0', s=60, zorder=5)

    coords = [68.202384, 94.649767, 99.434461]
    plt.scatter(xvals,coords, color='C1', s=60, zorder=9)


    # Add dashed lines
    alpha = 0.25
    plt.plot([1, 1], [0.0, 73.47], 'k--', lw=2, alpha=alpha)
    plt.plot([2, 2], [0.0, 96.86], 'k--', lw=2, alpha=alpha)
    plt.plot([3, 3], [0.0, 99.77], 'k--', lw=2, alpha=alpha)

    
    #ax.legend(legend_entries, fontsize=24)
    #ax.legend(fontsize=24)
    #ax.legend(fontsize=24, loc=(0.05,0.85), framealpha=1.0)
    ax.legend(fontsize=24, loc=(0.05,0.88), framealpha=1.0)
    plt.show()




    
# Run main() function when called directly
if __name__ == '__main__':
    main()
