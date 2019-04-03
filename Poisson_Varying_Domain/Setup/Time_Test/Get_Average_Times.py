import numpy as np
import csv

n_filename = "Network_Times.csv"
ng_filename = "Network_Times_NO_GPU.csv"
f_filename = "FEniCS_Times.csv"
fc_filename = "FEniCS_Times_Coarse.csv"

n_times = []
ng_times = []
f_times = []
fc_times = []

with open(n_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csvreader:
        n, tt, bt, at = row
        n_times.append(at)

with open(ng_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csvreader:
        n, tt, bt, at = row
        ng_times.append(at)
        
with open(f_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csvreader:
        n, tt, at = row
        f_times.append(at)        

with open(fc_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csvreader:
        n, tt, at = row
        fc_times.append(at)        

n_times = np.array(n_times, dtype=np.float32)
ng_times = np.array(ng_times, dtype=np.float32)
f_times = np.array(f_times, dtype=np.float32)
fc_times = np.array(fc_times, dtype=np.float32)

#print(n_times)
#print(f_times)

print("\n [ AVERAGE TIME RESULTS ]\n")

print(" Average FEniCS Time:\t\t%.5f seconds" %(np.mean(f_times)))
print(" Average FEniCS Time (Coarse):\t%.5f seconds" %(np.mean(fc_times)))

print("\n Average Network Time (CPU):\t%.5f seconds" %(np.mean(ng_times)))
print(" Average Network Time (GPU):\t%.5f seconds\n" %(np.mean(n_times)))
