#!/bin/bash
module purge
module load compiler/gcc/9.1.0
make 
make decompressor

