#!/bin/bash

# Check if correct number of arguments is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <question_number> [sub_part] [dataset_path]"
    exit 1
fi

# Assigning inputs to variables
question_number=$1
sub_part=$2
dataset_path=$3

# Function to run Q1 code
run_q1() {
    python q1.py
}

# Function to run Q2 code
run_q2() {
    python q2.py "$sub_part" "$dataset_path"
}

# Determine which question to run
case $question_number in
    1)
        run_q1
        ;;
    2)
        if [ -z "$sub_part" ] || [ -z "$dataset_path" ]; then
            echo "For question 2, both sub-part and dataset path are required."
            exit 1
        fi
        run_q2
        ;;
    *)
        echo "Invalid question number. Valid options are 1 or 2."
        exit 1
        ;;
esac