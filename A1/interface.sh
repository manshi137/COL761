#!/bin/bash

# Check if both input and output paths are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mode> <input_dataset_path> <output_dataset_path>"
    exit 1
fi

# Assign input and output paths to variables
mode="$1"
input_dataset="$2"
output_dataset="$3"

module purge
module load compiler/gcc/9.1.0

if [ "$mode" = "C" ]; then
    # Compression mode
    ./main "$input_dataset" "$output_dataset"
    # Replace "compression_executable" with the actual compression command or executable
elif [ "$mode" = "D" ]; then
    # Decompression mode
    ./decompress "$input_dataset" "$output_dataset"
    # Replace "decompression_executable" with the actual decompression command or executable
else
    echo "Invalid mode. Modes: C (compression), D (decompression)"
    exit 1
fi
