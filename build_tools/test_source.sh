#!/bin/bash

set -e
set -x

cd ../../

python -m venv test_env
source test_env/bin/activate

python -m pip install random_forest_run/build/python_package/dist/*.tar.gz
python -m pip install pytest pandas

pytest random_forest_run/tests/pyrfr_unit_test_binary_regression_forest_transformed_data.py
