failed=0
if [ $failed -eq 0 ]; then
    python Solve_Systems.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Solve_Systems.py"
    fi
fi
if [ $failed -eq 0 ]; then
    python Preprocess_Data.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Preprocess_Data.py"
    fi
fi
if [ $failed -eq 0 ]; then
    python Write_TFRecords.py
    if [ $? -ne 0 ]; then
        failed=1
        fail_file="Write_TFRecords.py"
    fi
fi


if [ $failed -ne 0 ]; then
    echo " "
    echo "Error encountered in '${fail_file}'"
    echo " "
fi




