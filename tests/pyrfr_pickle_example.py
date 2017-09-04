import sys
sys.path.append("..")
import os
here = os.path.dirname(os.path.realpath(__file__))

import pickle
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import pyrfr.regression


data_set_prefix = '%(here)s/../test_data_sets/diabetes_' % {"here":here}

# feature matrix contains one data point per row
features  = np.loadtxt(data_set_prefix+'features.csv', delimiter=",")

# the responses come in a 1d array
responses =  np.loadtxt(data_set_prefix+'responses.csv', delimiter=",")

# the types have the following meaning:
#	0 - this variable is continuous
#  >0 - the number of values from {0, 1, ...} this variable can take
types = np.zeros([features.shape[1]],dtype=np.uint)

# the data container to wrap the numpy arrays
# note:	no copy of the data is made on creation if it is C continuous
#		if the input is a sliced, or datapoints are added, a copy is made!
data = pyrfr.regression.numpy_data_container(features, responses, np.zeros([features.shape[1]],dtype=np.uint))


# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = pyrfr.regression.binary_rss()

the_forest.num_trees = 32

# the forest's parameters
the_forest.seed=12					# reset to reseed the rng for the next fit
the_forest.do_bootstrapping=True	# default: false
the_forest.num_data_points_per_tree=0 # means same number as data points
the_forest.max_features = features.shape[1]//2 # 0 would mean all the features
the_forest.min_samples_to_split = 0	# 0 means split until pure
the_forest.min_samples_in_leaf = 0	# 0 means no restriction 
the_forest.max_depth=1024			# 0 means no restriction
the_forest.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon


the_forest.fit(data)


predictions_1 = np.zeros_like(responses);
for i in range(features.shape[0]):
	sample = features[i]
	predictions_1[i] = the_forest.predict(sample)[0]


fname = None


with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
	fname = f.name
	pickle.dump(the_forest, f)


with open(fname, 'r+b') as fh:
	a_second_forest = pickle.load(fh)
os.remove(fname)


predictions_2 = np.zeros_like(responses);
for i in range(features.shape[0]):
	sample = features[i]
	predictions_2[i] = a_second_forest.predict(sample)[0]


if (np.allclose(predictions_1, predictions_2)):
	print("successfully pickled/unpickled the forest")
else:
	print("something went wrong")


