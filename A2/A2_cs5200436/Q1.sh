#!/bin/bash
cd Q1
#make gaston
cd executables/gaston-1.1
make 
cd ../../

make 

# Check if the file path is provided as an argument
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file_path> <output_file_path>"
    exit 1
fi

chmod +x FormatFiles/Gaston_format
chmod +x FormatFiles/Gspan_format
chmod +x FormatFiles/FSG_format
chmod +x executables/gaston-1.1/gaston
chmod +x executables/gSpan-64
chmod +x executables/fsg

# Run the executable with the provided file path as an argument
./FormatFiles/Gaston_format "$1"
echo "Gaston_format execution completed."

./FormatFiles/Gspan_format "$1"
echo "Gspan_format execution completed."

./FormatFiles/FSG_format "$1"
echo "FSG_format execution completed."

python3 plot.py "$2"

cd ..
