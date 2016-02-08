import sys
sys.path.append("..")
import os
here = os.path.dirname(os.path.realpath(__file__))

import time
import numpy as np
import matplotlib.pyplot as plt

import pyrfr.regression


features = np.array([np.linspace(-1,1,100)]).transpose()

responses = np.exp(-np.power(features/0.3,2)).flatten() + 0.1*np.random.randn(features.shape[0])


print(features.shape, responses.shape)


# the types have the following meaning:
#	0 - this variable is continuous
#  >0 - the number of values from {0, 1, ...} this variable can take
types = np.zeros([features.shape[1]],dtype=np.uint)

# the data container to wrap the numpy arrays
# note:	no copy of the data is made on creation if it is C continuous
#		if the input is a sliced, or datapoints are added, a copy is made!
data1 = pyrfr.regression.numpy_data_container(features, responses, types)


# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = pyrfr.regression.binary_rss()

the_forest.num_trees = 16


# the forest's parameters
the_forest.seed=12					# reset to reseed the rng for the next fit
the_forest.do_bootstrapping=True	# default: false
the_forest.num_data_points_per_tree=features.shape[0]*5//10 # means same number as data points
the_forest.max_features = 1 # 0 would mean all the features
the_forest.min_samples_to_split = 0	# 0 means split until pure
the_forest.min_samples_in_leaf = 0	# 0 means no restriction 
the_forest.max_depth=1024			# 0 means no restriction
the_forest.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon

the_forest.fit(data1)



predictions = np.array([ the_forest.predict(x) for x in features])


print(features.shape, predictions.shape)

fig, ax = plt.subplots('121')

ax.fill_between(features[:,0], predictions[:,0] - predictions[:,1], predictions[:,0] + predictions[:,1], alpha=0.3)
ax.plot(features, predictions[:,0])
ax.scatter(features, responses)

ax.plot(features, np.exp(-np.power(features/0.3,2)).flatten(), color='red')

fig, ax = plt.subplots('122')

cov = np.array([the_forest.covariance(0, )])



plt.show()


