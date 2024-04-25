#!/bin/bash

train_nc_d1() {
    dataset_path=$1
    model_output_path=$2

    # Run trainer_nc_d1.py with python3
    python3 trainer_nc_d1.py $dataset_path $model_output_path
}

test_nc_d1() {
    model_path=$1
    test_dataset_path=$2
    output_labels_path=$3

    # Run tester_nc_d1.py with python3
    python3 tester_nc_d1.py $model_path $test_dataset_path $output_labels_path
}

train_nc_d2() {
    dataset_path=$1
    model_output_path=$2

    # Run trainer_nc_d2.py with python3
    python3 trainer_nc_d2.py $dataset_path $model_output_path
}

test_nc_d2() {
    model_path=$1
    test_dataset_path=$2
    output_labels_path=$3

    # Run tester_nc_d2.py with python3
    python3 tester_nc_d2.py $model_path $test_dataset_path $output_labels_path
}

if [[ "$1" == "train" ]]; then
    if [[ "$#" -ne 3 ]]; then
        echo "Usage: bash interface.sh train </path/to/dataset> </path/to/output/model/in>"
        exit 1
    fi
    if [[ $2 == *"NC_D1"* ]]; then
        train_nc_d1 "$2" "$3"
    elif [[ $2 == *"NC_D2"* ]]; then
        train_nc_d2 "$2" "$3"
    else
        echo "Unknown dataset type. Cannot determine which trainer script to use."
        exit 1
    fi
elif [[ "$1" == "test" ]]; then
    if [[ "$#" -ne 4 ]]; then
        echo "Usage: bash interface.sh test <path/to/model> </path/to/test/dataset> </path/to/output/labels.txt>"
        exit 1
    fi
    if [[ $3 == *"NC_D1"* ]]; then
        test_nc_d1 "$2" "$3" "$4"
    elif [[ $3 == *"NC_D2"* ]]; then
        test_nc_d2 "$2" "$3" "$4"
    else
        echo "Unknown dataset type. Cannot determine which tester script to use."
        exit 1
    fi
else
    echo "Invalid command. Please use 'train' or 'test'."
    echo "Usage:"
    echo "  $ bash interface.sh train </path/to/dataset> </path/to/output/model/in>"
    echo "  $ bash interface.sh test <path/to/model> </path/to/test/dataset> </path/to/output/labels.txt>"
    exit 1
fi
