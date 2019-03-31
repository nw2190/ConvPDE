import numpy as np
import matplotlib.pyplot as plt
import csv



def main():
    legend_entries = []
    filename = "class_losses.csv"
    alt_filename = "noprob_class_losses.csv"
    classes = 20
    length_scales = [0.2, 0.2125, 0.225, 0.24, 0.25, 0.2625, 0.275, 0.2875, 0.3, 0.325,
                     0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.533, 0.566, 0.6]


    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    

    def get_data(f):
        cs = []
        l1_means = []
        l1_stds = []
        mse_means = []
        mse_stds = []

        with open(f, "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                c, l1_mean, l1_std, mse_mean, mse_std = row
                cs.append(float(c))
                l1_means.append(float(l1_mean))
                l1_stds.append(float(l1_std))
                mse_means.append(float(mse_mean))
                mse_stds.append(float(mse_std))
                #print([c, l1_mean, l1_std, mse_mean, mse_std])
        cs = np.array(cs)
        l1_means = np.array(l1_means)
        l1_stds = np.array(l1_stds)
        mse_means = np.array(mse_means)
        mse_stds = np.array(mse_stds)
        return cs, l1_means, l1_stds, mse_means, mse_stds

    cs, l1_means, l1_stds, mse_means, mse_stds = get_data(filename)
    acs, al1_means, al1_stds, amse_means, amse_stds = get_data(alt_filename)


    # Plot parameters
    linewidth = 3
    titlesize = 24
    ylabelsize = 24
    xlabelsize = 24
    xticksize = 16
    yticksize = 16
    ylabelpad = 20
    xlabelpad = 20
    

    # Bar parametrs
    width = 0.4
    
    # Plot L^2 errors
    

    #alt_color = 'tab:purple'
    fig, ax1 = plt.subplots(figsize=(14.0,7.0))
    
    ax1.set_xlabel('Length Scale', fontsize=xlabelsize, labelpad=xlabelpad)
    #ax1.set_ylabel('L^2 Error', color=color, fontsize=ylabelsize, labelpad=ylabelpad)
    #ax1.set_ylabel('Average L^2 Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    #ax1.set_ylabel('Relative L^2 Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    ax1.set_ylabel(r'Average $\,L^2\,$ Relative Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    #ax1.plot(cs, mse_means, color=color, label="L^2 Error", linewidth=linewidth)

    error_kw = {"capsize": 4.5,
                "elinewidth": 1.75,
                "capthick": 2.25}
    
    color = 'tab:orange'
    #rects1 = ax1.bar(acs - 0.5*width, amse_means, width, color=color, yerr=amse_stds, label="MSE Training", alpha=0.7,
    rects1 = ax1.bar(acs - 0.5*width, amse_means, width, color=color, yerr=amse_stds, label="MSE Network", alpha=0.7,
                     ecolor="black", error_kw=error_kw)
    #, edgecolor='black', hatch="-")

    color = 'tab:blue'
    #rects2 = ax1.bar(cs + 0.5*width, mse_means, width, color=color, yerr=mse_stds, label="Probability Training", alpha=0.7,
    rects2 = ax1.bar(cs + 0.5*width, mse_means, width, color=color, yerr=mse_stds, label="Probability Network", alpha=0.7,
                     edgecolor='white', hatch="/", ecolor="black", error_kw=error_kw)


    
    #ax1.plot(acs, amse_means, linestyle='dashed', color=alt_color, label="L^2 Error (noprob)")
    #ax1.plot(acs, amse_means, linestyle='dashed', color=alt_color, label="(without probability)", linewidth=linewidth)
    #ax1.tick_params(axis='y', labelcolor=color, labelsize=yticksize)
    #legend_entries.append("L^2 Mean Error")

    ax1.tick_params(axis='y', labelsize=yticksize)


    # Set xticks to length scale values
    ticks = [n+1 for n in range(0,classes)]
    #labels = (0.2, 0.2125, 0.225, 0.24, 0.25, 0.2625, 0.275, 0.2875, 0.3, 0.325,
    #          0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.533, 0.566, 0.6)
    labels = tuple(["{0:.2}".format(l).replace("0","",1) for l in length_scales])
    plt.xticks(ticks, labels, fontsize=xticksize)

    """
    # Plot L^1 errors
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    alt_color = color
    #alt_color = 'tab:red'
    #ax2.set_ylabel('L^1 Error', color=color, fontsize=ylabelsize, labelpad=ylabelpad)
    ax2.set_ylabel('L^1 Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    #ax2.plot(cs, l1_means, color=color, linestyle='dashed', label="L^1 Error")
    rects3 = ax2.bar(cs + 2*width, l1_means, width, color=color, yerr=l1_stds)
    rects4 = ax2.bar(acs + 3*width, al1_means, width, color=color, yerr=al1_stds)
    
    #ax2.plot(acs, al1_means, color=alt_color, linestyle='dashed', label="L^1 Error (noprob)")
    #ax2.plot(acs, al1_means, color=alt_color, linestyle='dashed', label="(without probability)", linewidth=linewidth)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=yticksize)
    #legend_entries.append("L^1 Mean Error")
    """
    

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #fig.legend(fontsize=24, loc=(0.525,0.7))
    ax1.legend(fontsize=24, loc=(0.675,0.775))
    plt.show()





    
def old_main():
    NOPROB = True
    
    legend_entries = []
    filename = "class_losses.csv"
    alt_filename = "noprob_class_losses.csv"
    classes = 20
    length_scales = [0.2, 0.2125, 0.225, 0.24, 0.25, 0.2625, 0.275, 0.2875, 0.3, 0.325,
                     0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.533, 0.566, 0.6]


    def get_data(f):
        cs = []
        l1_means = []
        l1_stds = []
        mse_means = []
        mse_stds = []

        with open(f, "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                c, l1_mean, l1_std, mse_mean, mse_std = row
                cs.append(float(c))
                l1_means.append(float(l1_mean))
                l1_stds.append(float(l1_std))
                mse_means.append(float(mse_mean))
                mse_stds.append(float(mse_std))
                #print([c, l1_mean, l1_std, mse_mean, mse_std])
        cs = np.array(cs)
        l1_means = np.array(l1_means)
        l1_stds = np.array(l1_stds)
        mse_means = np.array(mse_means)
        mse_stds = np.array(mse_stds)
        return cs, l1_means, l1_stds, mse_means, mse_stds

    cs, l1_means, l1_stds, mse_means, mse_stds = get_data(filename)

    if NOPROB:
        acs, al1_means, al1_stds, amse_means, amse_stds = get_data(alt_filename)

        
    """
    cs = []
    l1_means = []
    l1_stds = []
    mse_means = []
    mse_stds = []
    
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            c, l1_mean, l1_std, mse_mean, mse_std = row
            cs.append(float(c))
            l1_means.append(float(l1_mean))
            l1_stds.append(float(l1_std))
            mse_means.append(float(mse_mean))
            mse_stds.append(float(mse_std))
            #print([c, l1_mean, l1_std, mse_mean, mse_std])

    cs = np.array(cs)
    l1_means = np.array(l1_means)
    l1_stds = np.array(l1_stds)
    mse_means = np.array(mse_means)
    mse_stds = np.array(mse_stds)
    """


    # Plot parameters
    linewidth = 3
    titlesize = 24
    ylabelsize = 20
    xlabelsize = 24
    xticksize = 16
    yticksize = 16
    ylabelpad = 20
    xlabelpad = 20
    
    # Plot L^2 errors
    color = 'tab:blue'
    alt_color = color
    #alt_color = 'tab:purple'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Length Scale', fontsize=xlabelsize, labelpad=xlabelpad)
    #ax1.set_ylabel('L^2 Error', color=color, fontsize=ylabelsize, labelpad=ylabelpad)
    ax1.set_ylabel('L^2 Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    ax1.plot(cs, mse_means, color=color, label="L^2 Error", linewidth=linewidth)
    if NOPROB:
        #ax1.plot(acs, amse_means, linestyle='dashed', color=alt_color, label="L^2 Error (noprob)")
        ax1.plot(acs, amse_means, linestyle='dashed', color=alt_color, label="(without probability)", linewidth=linewidth)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=yticksize)
    #legend_entries.append("L^2 Mean Error")

    # Plot L^2 standard deviations
    alpha = 0.1
    y1 = np.array(mse_means - mse_stds, dtype=np.float32)
    y2 = np.array(mse_means + mse_stds, dtype=np.float32)
    plt.fill_between(cs, y1, y2, where=y2 >= y1, facecolor=color, alpha=alpha, interpolate=True, label=None)

    if NOPROB:
        # Plot L^2 standard deviations (noprob)
        y1 = np.array(amse_means - amse_stds, dtype=np.float32)
        y2 = np.array(amse_means + amse_stds, dtype=np.float32)
        plt.fill_between(acs, y1, y2, where=y2 >= y1, facecolor=alt_color, alpha=alpha/2., interpolate=True, label=None, hatch='X', edgecolor='k')
        
    # Set xticks to length scale values
    ticks = [n+1 for n in range(0,classes)]
    #labels = (0.2, 0.2125, 0.225, 0.24, 0.25, 0.2625, 0.275, 0.2875, 0.3, 0.325,
    #          0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.533, 0.566, 0.6)
    labels = tuple(["{0:.2}".format(l).replace("0","",1) for l in length_scales])
    plt.xticks(ticks, labels, fontsize=xticksize)

    
    # Plot L^1 errors
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    alt_color = color
    #alt_color = 'tab:red'
    #ax2.set_ylabel('L^1 Error', color=color, fontsize=ylabelsize, labelpad=ylabelpad)
    ax2.set_ylabel('L^1 Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    #ax2.plot(cs, l1_means, color=color, linestyle='dashed', label="L^1 Error")
    ax2.plot(cs, l1_means, color=color, label="L^1 Error", linewidth=linewidth)
    if NOPROB:
        #ax2.plot(acs, al1_means, color=alt_color, linestyle='dashed', label="L^1 Error (noprob)")
        ax2.plot(acs, al1_means, color=alt_color, linestyle='dashed', label="(without probability)", linewidth=linewidth)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=yticksize)
    #legend_entries.append("L^1 Mean Error")

    
    # Plot L^1 standard deviations
    alpha = 0.15
    y1 = np.array(l1_means - l1_stds, dtype=np.float32)
    y2 = np.array(l1_means + l1_stds, dtype=np.float32)
    plt.fill_between(cs, y1, y2, where=y2 >= y1, facecolor=color, alpha=alpha, interpolate=True, label=None)

    if NOPROB:
        # Plot L^1 standard deviations (noprob)
        y1 = np.array(al1_means - al1_stds, dtype=np.float32)
        y2 = np.array(al1_means + al1_stds, dtype=np.float32)
        plt.fill_between(acs, y1, y2, where=y2 >= y1, facecolor=alt_color, alpha=alpha/2, interpolate=True, label=None, hatch='X', edgecolor='k')


    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if NOPROB:
        fig.legend(fontsize=24, loc=(0.525,0.7))
    else:
        fig.legend(fontsize=24, loc=(0.55,0.8))
        
    plt.show()

# Run main() function when called directly
if __name__ == '__main__':
    main()
