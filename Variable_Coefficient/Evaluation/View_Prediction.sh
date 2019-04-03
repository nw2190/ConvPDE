# Specify model folder
model_dir="../Model_26-32/"
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


python Plot_Prediction.py --model_dir ${model_dir} --show_error --ID $1 --view_angle ${view_angle} --view_elev ${view_elev} --error_view_angle ${error_view_angle} --error_view_elev ${error_view_elev}
