#!/bin/bash

# Specify model folder
model_dir="../Model_4-32/"
#model_dir="../Model_5-32/"

# Specify dpi for output images
dpi=1000

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
if ! [ ${#4} -gt 0 ]
then
    error_view_angle="45"
else
    error_view_angle="$4"
fi
if ! [ ${#5} -gt 0 ]
then
    error_view_elev="30"
else
    error_view_elev="$5"
fi

# Remove pre-existing files
if [ -f "Figures/Predictions_$1.pdf" ]; then
    rm "Figures/Predictions_$1.pdf"
fi
if [ -f "Figures/Predictions_$1_config.txt" ]; then
    rm "Figures/Predictions_$1_config.txt"
fi


python Plot_Prediction.py --model_dir ${model_dir} --show_error --ID $1 --save_plots --view_angle ${view_angle} --view_elev ${view_elev} --error_view_angle ${error_view_angle} --error_view_elev ${error_view_elev} --plot_all
pdfcrop "Figures/Predictions_$1.pdf" "Figures/Predictions_$1.pdf"
convert -density "$dpi" "Figures/Predictions_$1.pdf" "Figures/Predictions_$1.png"

echo "./Save_Prediction.sh $1 ${view_angle} ${view_elev} ${error_view_angle} ${error_view_elev}" >> "Figures/Predictions_$1_config.txt"
echo "python Plot_Prediction.py --model_dir ${model_dir} --show_error --ID $1 --save_plots --view_angle ${view_angle} --view_elev ${view_elev} --error_view_angle ${error_view_angle} --error_view_elev ${error_view_elev}" >> "Figures/Predictions_$1_config.txt"
echo "dpi = $dpi" >> "Figures/Predictions_$1_config.txt"
