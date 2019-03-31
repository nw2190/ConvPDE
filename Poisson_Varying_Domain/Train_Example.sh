#!/bin/bash

# Check if an argument has been provided
if ! [ ${#1} -gt 0 ]
then
    problem="1"
else
    problem="$1"
fi

if ! [ ${#2} -gt 0 ]
then
    data_files="32"
else
    data_files="$2"
fi

# PROBABILISTIC NETWORK
if [ $problem -eq "1" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}-${data_files}" --data_files ${data_files} --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.95 --factor --dropout_rate 0.045 --network 5 --stopping_size 1000  --use_prob_loss --interpolate --kl_weight 1.0 --training_steps 700000 --regularize --dataset_step 500 --rotate --flip --use_log_implementation --no_early_stopping --no_validation_checks

# MSE NETWORK (BOUNDARY LOSS = 0.1)
elif [ $problem -eq "2" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}-${data_files}" --data_files ${data_files} --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.925 --factor --dropout_rate 0.045 --network 5 --stopping_size 1000  --interpolate --use_kl --kl_weight 1.0 --training_steps 500000 --dataset_step 500 --rotate --flip

# MSE NETWORK (BOUNDARY LOSS = 0.0)
elif [ $problem -eq "3" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}-${data_files}" --data_files ${data_files} --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.925 --factor --dropout_rate 0.045 --network 5 --stopping_size 1000  --interpolate --use_kl --kl_weight 1.0 --training_steps 500000 --dataset_step 500 --rotate --flip --bdry_weight 0.0

# LAPLACE UNCERTAINTY MODEL
elif [ $problem -eq "4" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}-${data_files}" --data_files ${data_files} --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.95 --factor --dropout_rate 0.045 --network 5 --stopping_size 1000  --use_prob_loss --interpolate --use_kl --kl_weight 1.0 --training_steps 690000 --regularize --dataset_step 500 --rotate --flip --use_laplace

# CAUCHY UNCERTAINTY MODEL
elif [ $problem -eq "5" ]
then
    LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python main.py --model_dir "Model_${problem}-${data_files}" --data_files ${data_files} --use_inception --learning_rate 0.00075 --lr_decay_step 10000 --lr_decay_rate 0.95 --factor --dropout_rate 0.045 --network 5 --stopping_size 1000  --use_prob_loss --interpolate --use_kl --kl_weight 1.0 --training_steps 500000 --regularize --dataset_step 500 --rotate --flip --use_cauchy
    
fi
