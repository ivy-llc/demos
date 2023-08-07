#!/bin/bash

# install dependencies
cd demos
python3 -m pip install -r requirements.txt

# run test
python3 tests/notebook_testing.py