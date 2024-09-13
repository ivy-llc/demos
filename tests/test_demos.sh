#!/bin/bash

if [ $2 == true ]; then
export VERSION=nightly
fi

cd ivy
pip install -e .

cd ../demos
mkdir .ivy

# install dependencies
pip install -r requirements.txt >/dev/null 2>&1

# get the binaries
VERSION=nightly python3 -c "import ivy; ivy.utils.cleanup_and_fetch_binaries(clean=True)"


# run test
if [ $3 == true ]; then
    echo "PATH : $1"
    python3 tests/main.py "$1"
else
    echo "Running the README tests"
    pip install -r tests/requirements.txt
    python3 -m pytest tests/test_README.py
fi
