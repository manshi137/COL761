#!/bin/bash
# module purge 
# conda install pyg pytorch-sparse -c pyg
# pip install ogb==1.3.6
# pip install pytorch_lightning==2.0.2
# pip install gdown==4.7.1
# module load pythonpackages/3.6.0/ucs4/gnu/447/pandas/0.18.1/intel
# # Function to display usage information
usage() {
    echo "Usage: $0 {train|test} <arguments>"
    echo "train: $0 train </path/to/dataset> </path/to/output/model/in>"
    echo "test: $0 test </path/to/model> </path/to/test/dataset> </path/to/output/labels.txt>"
    exit 1
}

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    usage
fi

# Perform action based on the first argument
case $1 in
    "train")
        if [ "$#" -ne 3 ]; then
            echo "Error: Incorrect number of arguments for train."
            usage
        fi
        # Run training script
        python3 d1.py "$2" "$3"
        ;;
    "test")
        if [ "$#" -ne 4 ]; then
            echo "Error: Incorrect number of arguments for test."
            usage
        fi
        # Run testing script
        python3 d1test.py "$2" "$3" "$4"
        ;;
    *)
        echo "Error: Invalid action. Please choose 'train' or 'test'."
        usage
        ;;
esac
