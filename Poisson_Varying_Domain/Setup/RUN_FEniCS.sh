#!/bin/bash

# Specify whether a virtual environment should be
# used for writing the dataset to TFRecords files 
VENV=0
venv_command="source /home/nick/Documents/virtual_envs/tf/bin/activate"

# Specify which python command should be used
PYTHON=python3

# Specify whether the Cholesky factors must be recomputed
newchol=1

# Initialize failure state and begin data generation
failed=0
if [ $newchol -eq 1 ]; then
    if [ $failed -eq 0 ]; then
        "$PYTHON" Compute_Cholesky_Factors.py
        if [ $? -ne 0 ]; then
            failed=1
            fail_file="Compute_Cholesky_Factors.py"
        fi
    fi
fi
if [ $failed -eq 0 ]; then
    "$PYTHON" Generate_Samples.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Generate_Samples.py"
    fi
fi
if [ $failed -eq 0 ]; then
    "$PYTHON" Convert_Samples.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Convert_Samples.py"
    fi
fi
if [ $failed -eq 0 ]; then
    "$PYTHON" Generate_Meshes.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Generate_Meshes.py"
    fi
fi
if [ $failed -eq 0 ]; then
    "$PYTHON" Solve_Systems.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Solve_Systems.py"
    fi
fi


if [ $failed -ne 0 ]; then
    echo " "
    echo "Error encountered in '${fail_file}'"
    echo " "
fi




