#!/bin/bash

cd ivy
python3 -m pip install .
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

cd ..
cd demos

# install dependencies
python3 -m pip install -r requirements.txt >/dev/null 2>&1

# run test
echo "PATH"
echo "$1"
python3 new_tests/main.py "$1"
