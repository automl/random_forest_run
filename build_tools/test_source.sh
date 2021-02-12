#!/bin/bash

set -e
set -x

# Test the distribution on an isolated venv
# Directory structure is such that
# PWD=random_forest_run/random_forest_run/<repo>
cd ../../

python -m venv test_env
source test_env/bin/activate

python -m pip install pytest "numpy<=1.19"
python -m pip install random_forest_run/random_forest_run/dist/*.tar.gz

pytest random_forest_run/random_forest_run/tests/pyrfr_unit_test_binary_regression_forest_transformed_data.py
