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
    LD_PRELOAD="${LD_PRELOAD}" python main.py --model_dir "Model_${problem}" --use_prob_loss --use_inception --factor --interpolate --regularize --use_log_implementation --training_steps 500000 --no_early_stopping --no_validation_checks --batch_size 32  # --rotate --flip 

# MSE NETWORK (BOUNDARY LOSS = 0.1)
elif [ $problem -eq "2" ]
then
    LD_PRELOAD="${LD_PRELOAD}" python main.py --model_dir "Model_${problem}" --use_inception --factor --interpolate --use_kl --lr_decay_rate 0.925 --training_steps 500000 --no_early_stopping --no_validation_checks --batch_size 32 # --rotate --flip 

# MSE NETWORK (BOUNDARY LOSS = 0.0)
elif [ $problem -eq "3" ]
then
    LD_PRELOAD="${LD_PRELOAD}" python main.py --model_dir "Model_${problem}" --use_inception --factor --interpolate --use_kl --lr_decay_rate 0.925 --training_steps 500000 --no_early_stopping --no_validation_checks --batch_size 32 --bdry_weight 0.0 # --rotate --flip  
fi
