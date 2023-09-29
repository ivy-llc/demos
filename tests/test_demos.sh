#!/bin/bash

python3 -m pip install ivy

cd demos

mkdir .ivy
touch .ivy/key.pem
echo -n "$1" > .ivy/key.pem

# install dependencies
python3 -m pip install -r requirements.txt >/dev/null 2>&1

# run test
python3 tests/notebook_testing.py "$2" "$3"