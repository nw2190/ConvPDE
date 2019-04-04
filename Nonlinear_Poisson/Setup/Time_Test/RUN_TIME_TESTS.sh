#!/bin/bash

# Specify whether a virtual environment should be
# used for timing the GPU enabled network
VENV=1
venv_command="source /home/nick/Documents/virtual_envs/tf/bin/activate"

# Specify maximum number of increments
MAX="11"

filename="FEniCS_Times.csv"
if [ -f "${filename}" ]
then
   rm "${filename}"
fi

filename="FEniCS_Times_Coarse.csv"
if [ -f "${filename}" ]
then
   rm "${filename}"
fi

filename="Network_Times_NO_GPU.csv"
if [ -f "${filename}" ]
then
   rm "${filename}"
fi

filename="Network_Times.csv"
if [ -f "${filename}" ]
then
   rm "${filename}"
fi

n="1"   
while [ $n -lt ${MAX} ]
do
    python Time_FEniCS.py --time_count "${n}"
    n=$((n+1))
done

n="1"   
while [ $n -lt ${MAX} ]
do
    python Time_FEniCS.py --time_count "${n}" --coarse_mesh
    n=$((n+1))
done

n="1"   
while [ $n -lt ${MAX} ]
do
    python Time_Network.py --no_gpu --time_count "${n}"
    n=$((n+1))
done

if [ $VENV -eq 1 ]; then
    ${venv_command}
    n="1"   
    while [ $n -lt ${MAX} ]
    do
        python Time_Network.py --time_count "${n}"
        n=$((n+1))
    done
    deactivate
else
    n="1"   
    while [ $n -lt ${MAX} ]
    do
        python Time_Network.py --time_count "${n}"
        n=$((n+1))
    done
fi    

python Get_Average_Times.py
