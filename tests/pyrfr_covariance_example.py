import sys
sys.path.append("..")
import os
here = os.path.dirname(os.path.realpath(__file__))

import time
import numpy as np
import matplotlib.pyplot as plt

import pyrfr.regression

sys.path.append("../../../github/george/")
import george


features = np.array([np.linspace(-1,1,51)]).transpose()
x2 = np.array([np.linspace(-1,1,100)]).transpose()
responses = np.exp(-np.power(features/0.3,2)).flatten() + 0.2*np.random.randn(features.shape[0])



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

the_forest.num_trees = 64


# the forest's parameters
the_forest.seed=12					# reset to reseed the rng for the next fit
the_forest.do_bootstrapping=True	# default: false
the_forest.num_data_points_per_tree=features.shape[0]*7//10 # means same number as data points
the_forest.max_features = 1 # 0 would mean all the features
the_forest.min_samples_to_split = 0	# 0 means split until pure
the_forest.min_samples_in_leaf = 1	# 0 means no restriction 
the_forest.max_depth=1024			# 0 means no restriction
the_forest.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon

the_forest.fit(data1)



predictions = np.array([ the_forest.predict(x) for x in x2])

fig, (ax1,ax2) = plt.subplots(2, sharex=True)

ax1.fill_between(x2[:,0], predictions[:,0] - predictions[:,1], predictions[:,0] + predictions[:,1], alpha=0.3)
ax1.plot(x2, predictions[:,0])
ax1.scatter(features, responses)

ax1.plot(x2, np.exp(-np.power(x2/0.3,2)).flatten(), color='red')

cov = np.array([the_forest.covariance(np.array([0], dtype=np.double), x) for x in x2])/the_forest.covariance(np.array([0], dtype=np.double), np.array([0],dtype=np.double)) 


ax2.plot(x2[:,0], cov)


plt.show()


