import numpy as np

def check_data(ID=0):
    mesh_dir = "./Meshes/"
    data_dir = "./Data/"
    soln_dir = "./Solutions/"

    mesh_file = mesh_dir + 'mesh_' + str(ID) + '.npy'
    mesh = np.load(mesh_file)

    data_file = data_dir + 'data_' + str(ID) + '.npy'
    data = np.load(data_file)
    data_min = np.min(data)
    data_max = np.max(data)
    #print("Data min/max:  {} / {}".format(data_min,data_max))

    soln_file = soln_dir + 'solution_' + str(ID) + '.npy'
    soln = np.load(soln_file)
    SCALING = 10.0
    soln = SCALING*soln
    soln_min = np.min(soln)
    soln_max = np.max(soln)
    #print("Solution min/max:  {} / {}".format(soln_min,soln_max))


    in_domain = (mesh > 0)
    domain_count = np.sum(in_domain)
    #print("Domain count: {}".format(domain_count))
    data_int = np.sum(data[in_domain])/domain_count
    soln_int = np.sum(soln[in_domain])/domain_count
    #soln_abs_int = np.sum(np.power(soln[in_domain],2))/domain_count
    #print("Data Integral: {}".format(data_int))
    #print("Solution Integral: {}".format(soln_int))
    print("{:2}  {:.4e} {:.4e}  {:.4e} {:.4e}  {:.4e} {:.4e}".format(ID,data_min,data_max,soln_min,soln_max,data_int,soln_int))
    #print("{:2}  {:.4e} {:.4e}  {:.4e} {:.4e}  {:.4e} {:.4e} {:.4e} ".format(ID,data_min,data_max,soln_min,soln_max,data_int,soln_int,soln_abs_int))

if __name__ == '__main__':
    print("\nID  Data min   Data max     Soln min   Soln max    Data int    Soln int\n"+"-"*74)
    for ID in range(0,15):
        check_data(ID=ID)
