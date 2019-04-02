#!/bin/bash

# Check if an argument has been provided
if ! [ ${#1} -gt 0 ]
then
    problem="1"
else
    problem="$1"
fi


# PROBABILISTIC NETWORK
if [ $problem -eq "1" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}" --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.95 --factor --dropout_rate 0.035 --network 1 --stopping_size 1000  --use_prob_loss --interpolate --use_kl --kl_weight 1.0 --training_steps 200000

# MSE NETWORK (BOUNDARY LOSS = 0.1)
elif [ $problem -eq "2" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}" --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.95 --factor --dropout_rate 0.035 --network 1 --stopping_size 1000   --interpolate --use_kl --kl_weight 1.0 --training_steps 500000

# MSE NETWORK (BOUNDARY LOSS = 0.0)
elif [ $problem -eq "3" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}" --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.95 --factor --dropout_rate 0.035 --network 1 --stopping_size 1000   --interpolate --use_kl --kl_weight 1.0 --training_steps 500000 --bdry_weight 0.0
fi
