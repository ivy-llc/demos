#!/bin/bash

# import ivy
export PYTHONPATH=$PYTHONPATH:'/ivy/ivy'

cd demos

# Compiler/Transpiler API keys
mkdir .ivy
touch .ivy/key.pem
echo -n "$1"> .ivy/key.pem
export IVY_ROOT='./.ivy'

# install dependencies
python3 -m pip install -r requirements.txt >/dev/null 2>&1

# run test
python3 tests/notebook_testing.py