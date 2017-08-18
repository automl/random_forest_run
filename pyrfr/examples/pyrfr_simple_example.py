import sys
sys.path.append("../build/")
import os
here = os.path.dirname(os.path.realpath(__file__))

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import pyrfr.regression as reg




data_set_prefix = '%(here)s/../test_data_sets/diabetes_' % {"here":here}
data = reg.default_data_container(10)
data.import_csv_files(data_set_prefix+'features.csv', data_set_prefix+'responses.csv')






rng = reg.default_random_engine()

# create an instance of a regerssion forest using binary splits and the RSS loss

the_forest = reg.binary_rss_forest()
the_forest.options.num_trees = 64
the_forest.options.num_data_points_per_tree = 200


print(the_forest.options.num_trees)
the_forest.fit(data, rng)

# you can save the forest to disk
the_forest.save_to_binary_file("/tmp/pyrfr_test.bin")


num_datapoints_old = the_forest.options.num_data_points_per_tree


# loading it works like that
the_forest = reg.binary_rss_forest()
the_forest.load_from_binary_file("/tmp/pyrfr_test.bin")

# you can save a LaTeX document that you can compile with pdflatex
the_forest.save_latex_representation("/tmp/rfr_test")


# the predict method will return a tuple containing the predicted mean and the standard deviation.
feature_vector = data.retrieve_data_point(0)
print(feature_vector)
print(the_forest.predict(feature_vector))

# it is possible to get the actual response values from the corresponding
# leaf in each tree that the given feature vector falls into
# The method returns a nested list
print(the_forest.all_leaf_values(feature_vector))

# quantile regression forest estimations
the_forest = reg.qr_forest()
the_forest.options.num_trees = 64
the_forest.options.num_data_points_per_tree = 200

the_forest.fit(data,rng)
alphas =[0.25, 0.5, 0.75]
print("="*50)
print(the_forest.predict_quantiles(feature_vector, alphas))
