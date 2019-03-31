#!/bin/bash
# data_count = 500   ==>  10000 solutions 
# data_count = 1000   ==>  20000 solutions 
#python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 0
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 10000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 20000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 30000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 40000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 50000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 60000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 70000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 80000
python Solve_Systems.py --cpu_count 8 --data_count 500  --data_start_count 90000

