import sys
sys.path.append("..")

import os
here = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import matplotlib.pyplot as plt


import pyrfr.regression


data_set_prefix = '%(here)s/../test_data_sets/diabetes_' % {"here":here}

features  = np.loadtxt(data_set_prefix+'features.csv', delimiter=",")
responses =  np.loadtxt(data_set_prefix+'responses.csv', delimiter=",")
types = np.zeros([features.shape[1]],dtype=np.uint)

data1 = pyrfr.regression.numpy_data_container(features, responses, types)

# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = pyrfr.regression.binary_rss()

the_forest.num_trees = 16

# the forest's parameters
the_forest.seed=12					# reset to reseed the rng for the next fit
the_forest.do_bootstrapping=True	# default: false
the_forest.num_data_points_per_tree=0 # means same number as data points
the_forest.max_features = features.shape[1]//2 # 0 would mean all the features
the_forest.min_samples_to_split = 0	# 0 means split until pure
the_forest.min_samples_in_leaf = 0	# 0 means no restriction 
the_forest.max_depth=1024			# 0 means no restriction
the_forest.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon


the_forest.fit(data1)

# for continuous parameters the parameter configuration space consists of intervals for each variable.
pcs = list(zip( np.min(features, axis=0), np.max(features, axis=0) ))

# for categorical parameters, the tuple/list contains all allowed values


partitionings = the_forest.induced_partitionings(pcs)
print(len(partitionings))
print([len(p) for p in partitionings])
