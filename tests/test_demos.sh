#!/bin/bash

cd ivy
pip install -e .

cd ../demos
mkdir .ivy
touch .ivy/key.pem
echo -n "$1" > .ivy/key.pem

conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

# install dependencies
pip install -r requirements.txt >/dev/null 2>&1

# run test
echo "PATH"
echo "$2"
python3 new_tests/main.py "$2"
