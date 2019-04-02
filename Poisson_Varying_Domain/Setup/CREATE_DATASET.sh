#!/bin/bash

# Specify whether a virtual environment should be
# used for writing the dataset to TFRecords files 
VENV=1
venv_command="source ~/Documents/virtual_envs/tf/bin/activate"

# Specify whether the Cholesky factors must be recomputed
newchol=0

# Initialize failure state and begin data generation
failed=0
if [ $newchol -eq 1 ]; then
    if [ $failed -eq 0 ]; then
        python Compute_Cholesky_Factors.py
        if [ $? -ne 0 ]; then
            failed=1
            fail_file="Compute_Cholesky_Factors.py"
        fi
    fi
fi
if [ $failed -eq 0 ]; then
    python Generate_Samples.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Generate_Samples.py"
    fi
fi
if [ $failed -eq 0 ]; then
    python Convert_Samples.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Convert_Samples.py"
    fi
fi
if [ $failed -eq 0 ]; then
    python Generate_Meshes.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Generate_Meshes.py"
    fi
fi
if [ $failed -eq 0 ]; then
    python Solve_Systems.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Solve_Systems.py"
    fi
fi
if [ $failed -eq 0 ]; then
    python Preprocess_Data.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Preprocess_Data.py"
    fi
fi
if [ $failed -eq 0 ]; then
    if [ $VENV -eq 1 ]; then
        ${venv_command}
        python Write_TFRecords.py
        deactivate
    else
        python Write_TFRecords.py
    fi
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Write_TFRecords.py"
    fi
fi
if [ $failed -eq 0 ]; then
    python Clean_XML.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Clean_XML.py"
    fi
fi


if [ $failed -ne 0 ]; then
    echo " "
    echo "Error encountered in '${fail_file}'"
    echo " "
fi




