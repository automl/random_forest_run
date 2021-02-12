#!/bin/bash

set -e
set -x

pytest tests/pyrfr_unit_test_binary_regression_forest_transformed_data.py

# Test that there are no links to system libraries
python -m threadpoolctl -i pyrfr
