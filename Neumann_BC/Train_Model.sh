#!/bin/bash

# Specify thread cacheing memory allocation preloader if applicable
LD_PRELOAD="/usr/lib/libtcmalloc.so.4"

# Check if an argument has been provided for specifying the model number
if ! [ ${#1} -gt 0 ]
then
    problem="1"
else
    problem="$1"
fi

# PROBABILISTIC NETWORK
if [ $problem -eq "1" ]
then
    LD_PRELOAD="${LD_PRELOAD}" python main.py --model_dir "Model_${problem}" --use_inception --learning_rate --factor --use_prob_loss --interpolate --training_steps 500000 --regularize --rotate --flip
 
# MSE NETWORK (BOUNDARY LOSS = 0.1)
elif [ $problem -eq "2" ]
then
    LD_PRELOAD="${LD_PRELOAD}" python main.py --model_dir "Model_${problem}" --use_inception --learning_rate 0.0001 --factor --interpolate --use_kl --training_steps 500000 --rotate --flip
   
# MSE NETWORK (BOUNDARY LOSS = 0.0)
elif [ $problem -eq "3" ]
then
    LD_PRELOAD="${LD_PRELOAD}" python main.py --model_dir "Model_${problem}" --use_inception --learning_rate 0.0001 --factor --interpolate --use_kl --training_steps 500000 --rotate --flip --bdry_weight 0.0
fi
