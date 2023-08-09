#!/bin/bash

cd demos

# install dependencies
python3 -m pip install -r requirements.txt >/dev/null 2>&1

# run test
python3 tests/notebook_testing.py