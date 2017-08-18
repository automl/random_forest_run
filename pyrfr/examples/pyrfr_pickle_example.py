import sys
sys.path.append("..")
import os
here = os.path.dirname(os.path.realpath(__file__))

import pickle
import tempfile

import numpy as np
import pyrfr.regression



data_set_prefix = '%(here)s/../test_data_sets/diabetes_' % {"here":here}

features  = np.loadtxt(data_set_prefix+'features.csv', delimiter=",")
responses =  np.loadtxt(data_set_prefix+'responses.csv', delimiter=",")

data = pyrfr.regression.default_data_container(10)

data.import_csv_files(data_set_prefix+'features.csv', data_set_prefix+'responses.csv')


# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = pyrfr.regression.binary_rss_forest()

#reset to reseed the rng for the next fit
rng = pyrfr.regression.default_random_engine(42)
# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = pyrfr.regression.binary_rss_forest()

the_forest.options.num_trees = 16
# the forest's parameters
the_forest.options.compute_oob_error = True
the_forest.options.do_bootstrapping=True	# default: false
the_forest.options.num_data_points_per_tree=(data.num_data_points()//4)* 3 # means same number as data points
the_forest.options.tree_opts.max_features = data.num_features()//2 # 0 would mean all the features
the_forest.options.tree_opts.min_samples_to_split = 0	# 0 means split until pure
the_forest.options.tree_opts.min_samples_in_leaf = 0	# 0 means no restriction 
the_forest.options.tree_opts.max_depth=1024			# 0 means no restriction
the_forest.options.tree_opts.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon

the_forest.fit(data, rng)


predictions_1 = [ the_forest.predict(f.tolist()) for f in features]

with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
	fname = f.name
	pickle.dump(the_forest, f)


with open(fname, 'r+b') as fh:
	a_second_forest = pickle.load(fh)
os.remove(fname)


predictions_2 = [ a_second_forest.predict(f.tolist()) for f in features]


if (np.allclose(predictions_1, predictions_2)):
	print("successfully pickled/unpickled the forest")
else:
	print("something went wrong")


