#!/bin/bash

module load compiler/python/3.9.13/ucs4/gnu/447
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load compiler/gcc/9.1.0
cd Q2
# Run make to compile the code
make

cd ..

# Check if two arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Store the arguments in variables
input_file=$1
output_file=$2
chmod +x Q2/q2
chmod +x Q2/fsg
# Run the compiled program with the provided arguments
./Q2/q2 "$input_file" "$output_file"
