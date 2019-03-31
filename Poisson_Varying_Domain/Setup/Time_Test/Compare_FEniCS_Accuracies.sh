#!/bin/bash

python Solve_Systems.py --time_count 10 --save_solutions
python Solve_Systems.py --time_count 10 --save_solutions --coarse_mesh
python Time_Test.py --save_solutions 
python check_accuracies.py
