import sys
sys.path.append("..")
import os
here = os.path.dirname(os.path.realpath(__file__))

import time
import numpy as np
import matplotlib.pyplot as plt

import pyrfr.regression as reg


num_points = 8


features = np.array([np.linspace(-1,1,num_points)]).transpose()
x2 = np.array([np.linspace(-1,1,100)]).transpose()
responses = np.exp(-np.power(features/0.3,2)).flatten() + 0.05*np.random.randn(features.shape[0])




data = reg.default_data_container(1)

for f,r in zip(features, responses):
	data.add_data_point(f,r)

rng = reg.default_random_engine()

# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = reg.binary_rss_forest()
the_forest.options.num_trees = 64
the_forest.options.num_data_points_per_tree = num_points
the_forest.options.tree_opts.min_samples_in_leaf = 1


the_forest.fit(data, rng)

fig, (ax1,ax2, ax3) = plt.subplots(3, sharex=True)


predictions = np.array([ the_forest.predict_mean_var(x) for x in x2])
ax1.fill_between(x2[:,0], predictions[:,0] - predictions[:,1], predictions[:,0] + predictions[:,1], alpha=0.3)
ax1.plot(x2, predictions[:,0])
ax1.scatter(features, responses)
ax1.plot(x2, np.exp(-np.power(x2/0.3,2)).flatten(), color='red')


cov = np.array([the_forest.covariance(np.array([0], dtype=np.double), x) for x in x2])/the_forest.covariance(np.array([0], dtype=np.double), np.array([0],dtype=np.double)) 
ax2.plot(x2[:,0], cov)

kernel = np.array([the_forest.kernel([0], x.tolist()) for x in x2])
ax3.plot(x2, kernel)

plt.show()


