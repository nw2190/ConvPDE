#!/bin/bash

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
    python Solve_Systems.py --time_count "${n}"
    n=$((n+1))
done

n="1"   
while [ $n -lt ${MAX} ]
do
    python Solve_Systems.py --time_count "${n}" --coarse_mesh
    n=$((n+1))
done

n="1"   
while [ $n -lt ${MAX} ]
do
    python Time_Test.py --no_gpu --time_count "${n}"
    n=$((n+1))
done


source /home/nick/Documents/virtual_envs/tf/bin/activate
n="1"   
while [ $n -lt ${MAX} ]
do
    python Time_Test.py --time_count "${n}"
    n=$((n+1))
done
deactivate

python Get_Average_Times.py
