#!/bin/bash

count=1000

data_dir="./Data/"
mesh_dir="./Meshes/"
soln_dir="./Solutions/"


n="0"
while [ $n -lt $count ]
do
    filename="${data_dir}data_${n}.npy"
    if [ ! -f "${filename}" ]
    then
        echo "No data: ${n}"
    fi
    filename="${mesh_dir}mesh_${n}.npy"
    if [ ! -f "${filename}" ]
    then
        echo "No mesh: ${n}"
    fi
    filename="${soln_dir}solution_${n}.npy"
    if [ ! -f "${filename}" ]
    then
        echo "No solution: ${n}"
    fi
    n=$((n+1))
done
