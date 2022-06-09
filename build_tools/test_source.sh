#!/bin/bash

set -e
set -x

# Test the distribution on an isolated venv
# Directory structure is such that
# PWD=random_forest_run/random_forest_run/<repo>
cd ../../

python -m venv test_env
source test_env/bin/activate

python -m pip install pytest numpy
python -m pip install random_forest_run/random_forest_run/dist/*.tar.gz

sed -i -- "s/[^']\+test_data_sets/random_forest_run\/random_forest_run\/test_data_sets/" random_forest_run/random_forest_run/tests/*
pytest random_forest_run/random_forest_run/tests/*.py
