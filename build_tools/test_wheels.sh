#!/bin/bash
set -e  # Immediatly exit on any error
set -x  # Print each command before running

# The wheel builder will currently have the REPO cloned into /project.
# Also unittest from pyrfr are compiled and have local paths embedded.
# Modify such path so they are now pointing to /project
sed -i -- "s/[^']\+test_data_sets/\/project\/test_data_sets/" /project/tests/*
pytest -v /project/tests/*py

# Test that there are no links to system libraries
python -m threadpoolctl -i pyrfr
