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

model_dir="Model_${problem}-${data_files}/"
loss_file="${model_dir}evaluation_losses.csv"
skip_lines=2

clear
gnuplot -e "set terminal dumb size 120,30; set logscale y; set key off; set title 'Model ${problem}: Evaluation Loss'; plot '${loss_file}' every ::${skip_lines} using 1:2 smooth mcsplines with lines;"


## Plot UQ
#
#model_dir="Model_${problem}-${data_files}/"
#uq_file="${model_dir}evaluation_uncertainties.csv"
#skip_lines=2
#
#gnuplot -e "set terminal dumb size 120,30; set logscale y; set key off; set title 'Model ${problem}: Uncertainty Quantification'; plot '${uq_file}' every ::${skip_lines} using 1:2 smooth mcsplines with lines;"
