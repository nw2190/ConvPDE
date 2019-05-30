#!/bin/bash

# Specify whether a virtual environment should be
# used for writing the dataset to TFRecords files 
VENV=0
venv_command="source /home/nick/Documents/virtual_envs/tf/bin/activate"

# Specify which python command should be used
PYTHON=python3

# Initialize failure state and begin data generation
failed=0
if [ $failed -eq 0 ]; then
    "$PYTHON" Preprocess_Data.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Preprocess_Data.py"
    fi
fi
if [ $failed -eq 0 ]; then
    if [ $VENV -eq 1 ]; then
        ${venv_command}
        "$PYTHON" Write_TFRecords.py
        deactivate
    else
        "$PYTHON" Write_TFRecords.py
    fi
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Write_TFRecords.py"
    fi
fi
if [ $failed -eq 0 ]; then
    "$PYTHON" Clean_XML.py
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




