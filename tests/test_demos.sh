#!/bin/bash

echo "NIGHTLY : $3"

if [ $3 == "true" ];
then
echo "NIGHTLY"
export VERSION=nightly
echo $VERSION
fi

cd ivy
pip install -e .

cd ../demos
mkdir .ivy
touch .ivy/key.pem
echo -n "$1" > .ivy/key.pem

# conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

# install dependencies
pip install -r requirements.txt >/dev/null 2>&1

# run test
echo "PATH : $2"
python3 tests/main.py "$2"
