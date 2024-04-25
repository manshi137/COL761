#!/bin/bash

if [ $# -eq 3 ]; then
    # 4 arguments
    if [[ $2 == *D1* ]]; then
        python Graph_D1.py "$1" "$2" "$3"
    elif [[ $2 == *D2* ]]; then
        python Graph_D2.py "$1" "$2" "$3"
    else
        echo "Invalid arguments"
    fi
elif [ $# -eq 4 ]; then
    # 5 arguments
    if [[ $3 == *D1* ]]; then
        python Graph_D1.py "$1" "$2" "$3" "$4"
    elif [[ $3 == *D2* ]]; then
        python Graph_D2.py "$1" "$2" "$3" "$4"
    else
        echo "Invalid arguments"
    fi
else
    echo "Invalid number of arguments"
fi