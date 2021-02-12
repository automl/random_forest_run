#!/bin/bash

set -e
set -x

cd ../../
pwd
ls

python -m venv test_env
source test_env/bin/activate

pwd
ls
sudo apt-get install tree
tree
python -m pip install pytest "numpy<=1.19"
python -m pip install random_forest_run/random_forest_run/dist/*.tar.gz

pytest random_forest_run/random_forest_run/tests/pyrfr_unit_test_binary_regression_forest_transformed_data.py
