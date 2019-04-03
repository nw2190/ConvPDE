import numpy as np
import matplotlib.pyplot as plt
import csv
import os

import matplotlib as mpl
from scipy.stats.stats import pearsonr
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def main():
    NOPROB = True
    model_dir = "../Model_4-32/"
    legend_entries = []
    filename = os.path.join(model_dir, "evaluation_losses.csv")
    uq_filename = os.path.join(model_dir, "evaluation_uncertainties.csv")

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    
    def get_data(f):
        SKIP = 100
        steps = []
        values = []
        with open(os.path.join(model_dir, f), "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                step, val = row
                steps.append(step)
                values.append(val)
        steps = np.array(steps[SKIP:]).astype(np.float32)
        values = np.array(values[SKIP:]).astype(np.float32)
        return steps, values


    steps, loss = get_data(filename)
    steps, uq = get_data(uq_filename)

    def smooth(vals, N):
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
    
    smooth_loss = smooth(loss, 20)
    smooth_uq = smooth(uq, 20)


    # Plot parameters
    #"""
    linewidth = 3
    titlesize = 24
    legendsize = 24
    ylabelsize = 22
    xlabelsize = 22
    xticksize = 16
    yticksize = 16
    ylabelpad = 20
    xlabelpad = 20
    #"""
    """
    linewidth = 3
    titlesize = 26
    legendsize = 26
    ylabelsize = 24
    xlabelsize = 26
    xticksize = 18
    yticksize = 18
    ylabelpad = 20
    xlabelpad = 20
    """


    
    # Plot loss values
    color = 'tab:blue'
    alt_color = color
    fig, ax1 = plt.subplots()

    # Set xticks to length scale values
    ticks = [n*1000 for n in [100, 150, 200, 250, 300, 350, 400, 450, 500]]
    labels = tuple(["{0:}K".format(n) for n in [100, 150, 200, 250, 300, 350, 400, 450, 500]])
    plt.xticks(ticks, labels, fontsize=xticksize)

    ax1.set_xlabel('Training Steps', fontsize=xlabelsize, labelpad=xlabelpad)
    ax1.set_ylabel(r'$L^2\,$ Absolute Error', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    #ax1.plot(steps, loss, color=color, label="L^2 Error")
    ax1.plot(steps, loss, color=color, label=None, linestyle='dashed', alpha=0.25)
    ax1.plot(steps, smooth_loss, color=color, label=r"$L^2$ Error", linewidth=linewidth)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=yticksize)
    #legend_entries.append("L^2 Mean Error")

    # Set xticks to length scale values
    #ticks = [n+1 for n in range(0,classes)]
    #labels = (0.2, 0.2125, 0.225, 0.24, 0.25, 0.2625, 0.275, 0.2875, 0.3, 0.325,
    #          0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.533, 0.566, 0.6)
    #labels = tuple(["{0:.2}".format(l).replace("0","",1) for l in length_scales])
    #plt.xticks(ticks, labels, fontsize=xticksize)
    plt.xticks(fontsize=xticksize)

    
    # Plot L^1 errors
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    alt_color = color
    ax2.set_ylabel('Model Uncertainty', color='k', fontsize=ylabelsize, labelpad=ylabelpad)
    #ax2.plot(cs, l1_means, color=color, linestyle='dashed', label="L^1 Error")
    ax2.plot(steps, uq, color=color, label=None, linestyle='dashed', alpha=0.25)
    ax2.plot(steps, smooth_uq, color=color, label="Uncertainty", linewidth=linewidth)
    
    ax2.tick_params(axis='y', labelcolor=color, labelsize=yticksize)
    #legend_entries.append("L^1 Mean Error")

    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped


    fig.legend(fontsize=legendsize, loc=(0.575,0.8))



    plt.show()    



    ## REGRESSION / CORRELATION

    # Use set_aspect before plt.show()
    #w, h = mpl.figure.figaspect(1.)
    #fig = plt.Figure(figsize=(w, h))
    fig = plt.figure()
    ax = plt.gca()
    #fig, ax = plt.subplots()
    
    ax.set_xlabel(r'$L^2\,$ Absolute Error', fontsize=xlabelsize, labelpad=xlabelpad)
    ax.set_ylabel("Model Uncertainty", color='k', fontsize=ylabelsize, labelpad=ylabelpad)

    ax.scatter(loss,uq)

    USE_LOG = False
    if USE_LOG:
        ax.set_xscale('log')
        ax.set_yscale('log')

    OUT = 3
    inds = np.argsort(uq)
    uq = uq[inds]; uq = uq[:len(uq)-OUT]
    loss = loss[inds]; loss = loss[:len(loss)-OUT]
        
    epsilon = 0.05
    plt.xlim(((1-epsilon)*np.min(loss), (1+epsilon)*np.max(loss)))
    plt.ylim(((1-0.5*epsilon)*np.min(uq), (1+epsilon)*np.max(uq)))
    #plt.show()
    
    epsilon1 = epsilon/2
    epsilon2 = -0.1*epsilon
    regr = linear_model.LinearRegression()
    regr.fit(np.expand_dims(loss, 1), np.expand_dims(uq, 1))
    x_pred = np.linspace((1-epsilon1)*np.min(loss), (1+epsilon2)*np.max(loss), 100)
    y_pred = regr.predict(np.expand_dims(x_pred,1))
    ax.plot(x_pred,y_pred[:,0], linewidth=4, color='C1', alpha=1.0, linestyle='dashed')

    plt.xticks(fontsize=xticksize)
    plt.yticks(fontsize=xticksize)

    #fig.set_figheight(1)
    #fig.set_figwidth(1)
    plt.axes().set_aspect(0.025)
    
    #print(regr.coef_)
    pearson_coef, p_val = pearsonr(loss,uq)
    print("Pearson correlation coefficient: %.5f" %(pearson_coef))

    textstr = r"Pearson Correlation Coeff:  %.3f" %(pearson_coef)


    ##props = dict(boxstyle='round', facecolor='gray', alpha=0.1, edgecolor='black')
    #props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
    ##ax.text(0.635, 0.855, textstr, transform=ax.transAxes, fontsize=18,
    #ax.text(0.4, 0.15, textstr, transform=ax.transAxes, fontsize=20,
    #        verticalalignment='top', bbox=props)

    #plt.legend()
    #ax.annotate("$C_{3}H_{8}$", xy=(0.9,0.9),xycoords='axes fraction', fontsize=14, boxstyle="Round,pad=0.3")
    bbox_props = dict(boxstyle="Round,pad=0.3", fc="white", ec="black", alpha=0.125, lw=1.5)
    t = plt.text(0.31, 0.125, textstr, transform=ax.transAxes, fontsize=24, bbox=bbox_props)
    #t.set_bbox(dict(facecolor='white', alpha=1.0))
    plt.show()
    
# Run main() function when called directly
if __name__ == '__main__':
    main()
