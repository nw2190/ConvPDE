#!/bin/bash

# Specify model folder
model_dir="../Model_2-32/"

# Check if an argument has been provided
if ! [ ${#2} -gt 0 ]
then
    view_angle="45"
else
    view_angle="$2"
fi
if ! [ ${#3} -gt 0 ]
then
    view_elev="30"
else
    view_elev="$3"
fi

python Plot_Prediction.py --model_dir ${model_dir} --show_error --ID $1 --save_plots --view_angle ${view_angle} --view_elev ${view_elev}
pdfcrop "Figures/Predictions_$1.pdf" "Figures/Predictions_$1.pdf"
convert -density 1000 "Figures/Predictions_$1.pdf" "Figures/Predictions_$1.png"
