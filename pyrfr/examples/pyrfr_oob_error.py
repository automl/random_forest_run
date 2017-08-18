import sys
sys.path.append("../build")

import os
here = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import matplotlib.pyplot as plt


import pyrfr.regression


data_set_prefix = '%(here)s/../test_data_sets/diabetes_' % {"here":here}

features  = np.loadtxt(data_set_prefix+'features.csv', delimiter=",")
responses =  np.loadtxt(data_set_prefix+'responses.csv', delimiter=",")
types = np.zeros([features.shape[1]],dtype=np.uint)


num_train_samples = 400


indices = np.array(range(features.shape[0]))
np.random.shuffle(indices)


features_train = features[indices[:num_train_samples]]
features_test  = features[indices[num_train_samples:]]

responses_train = responses[indices[:num_train_samples]]
responses_test  = responses[indices[num_train_samples:]]




data = pyrfr.regression.default_data_container(features_train.shape[1])

for f,r in zip(features_train, responses_train):
	data.add_data_point(f.tolist(), r)

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




predictions_test = [the_forest.predict(f.tolist()) for f in features_test]
print(np.sqrt(np.mean((predictions_test - responses_test) ** 2)))
print(the_forest.out_of_bag_error())
