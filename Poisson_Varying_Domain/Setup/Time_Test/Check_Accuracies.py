import numpy as np
import os
import sys
import time

# Import flags specifying dataset parameters
from timer_flags import getFlags

if __name__ == '__main__':

    FLAGS = getFlags()

    #data_count = FLAGS.time_count*FLAGS.data_count
    data_count = 10*20*FLAGS.data_count
    
    l1_rels = []
    l2_rels = []
    l1_rels_network = []
    l2_rels_network = []
    for n in range(0,data_count):

        coarse_solution_filename = './Solutions/coarse_solution_' + str(n) + '.npy'
        solution_filename = './Solutions/solution_' + str(n) + '.npy'
        network_filename = './Solutions/network_solution_' + str(n) + '.npy'

        # Note that the SOLN_SCALING factor used for neural network training stability
        # ( see definition of "_parse_data" function in "misc.py" )
        SOLN_SCALING = 100.0
        coarse_solution = SOLN_SCALING * np.load(coarse_solution_filename)
        solution = SOLN_SCALING * np.load(solution_filename)
        network_solution = np.load(network_filename)

        #print("NET:")
        #print(network_solution[60:65,60:65])
        #print("SOLN:")
        #print(solution[60:65,60:65])

        diff = coarse_solution - solution
        network_diff = network_solution[:,:] - solution

        # Compute solution norms on interior of domain
        interior_l1 = np.sum(np.abs(solution), axis=(0,1))
        interior_l2 = np.sum(np.power(solution,2), axis=(0,1))
        
        # Compute relative norms for coarse FEniCS solutions
        l1_rels.append(np.sum(np.abs(diff), axis=(0,1)) / interior_l1)
        l2_rels.append(np.sum(np.power(diff, 2), axis=(0,1)) / interior_l2)

        # Compute relative norms for neural network solutions
        l1_rels_network.append(np.sum(np.abs(network_diff), axis=(0,1)) / interior_l1)
        l2_rels_network.append(np.sum(np.power(network_diff, 2), axis=(0,1)) / interior_l2)

        #if np.mod(n,100) == 0:
        #    print("{}: \t {} \t {}".format(n, l1_rels_network[n], l2_rels_network[n]))
        #if n < 500:
        #    print("{}: \t {} \t {}".format(n, l1_rels_network[n], l2_rels_network[n]))


    # Display accuracy information
    print("\nCoarse FEniCS Accuracy:")    
    print(np.mean(l1_rels))
    print(np.mean(l2_rels))
                
    print("\nNetwork Accuracy:")
    print(np.mean(l1_rels_network))
    print(np.mean(l2_rels_network))
    print(" ")
