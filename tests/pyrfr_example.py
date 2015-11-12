import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import pyrfr.regression


data_set_prefix = '../test_data_sets/diabetes_'

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
data1 = pyrfr.regression.numpy_data_container(features, responses, np.zeros([features.shape[1]],dtype=np.uint))

# this is how you add a data point
data1.add_data_point(np.random.rand(10), 1)

print("number of features: {}".format(data1.num_features()))
print("number of data points: {}".format(data1.num_data_points()))

# how to get data points out of the container. Negative indices are supported, too!
data1.retrieve_data_point(-2)

# this container grants access to the data arrays directly:
data1.features
data1.responses
data1.types

# A second container living completely in the C++ code.
# The only argument to the constructor  is the number of features
data2 = pyrfr.regression.mostly_continuous_data_container(features.shape[1])

# set the types of each feature, before any data is added!
# you can set the type of each feature like that:
data2.set_type_of_feature(0,5) #arguments are "feature index" and "type"

# how to get the type of a feature out of the container
print("feature 0 is now of type {}".format(data2.get_type_of_feature(0)))

# define each type before you add any data points
data2 = pyrfr.regression.mostly_continuous_data_container(features.shape[1])

# besides adding data points as above (add_data_point method), this 
# container can import numpy arrays (a copy of the data will be made!)
data2.import_numpy_arrays(features, responses);



if np.allclose(data2.export_responses(),responses) and  np.allclose(data2.export_features(),features):
	print("Import of data into data2 was successful")


# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = pyrfr.regression.binary_rss()

the_forest.num_trees = 2


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


# you can save a LaTeX document that you can compile with pdflatex
#the_forest.save_latex_representation("/tmp/rfr_test")


# the predict method will return a tuple containing the predicted mean and the standard deviation.
print(the_forest.predict(features[0]))



# it is possible to get the actual response values from the corresponding
# leaf in each tree that the given feature vector falls into
# The method returns a nested list
print(the_forest.all_leaf_values(features[0]))



# let's play around a bit and train different numbers of trees and compare the speed to scikit learn
# Note: the dataset here is pretty small, so the results may not by representative
times_rfr = []
times_scikit=[]
num_trees = [1, 2, 4, 8, 16, 32, 64, 128]

for nt in num_trees:
	print("training {} trees".format(nt))
	the_forest.num_trees=nt

	start = time.time()

	# this is how you fit the forest, it should work with any data container
	the_forest.fit(data2)

	end = time.time()
	times_rfr.append(end-start)

	sk_forest = RandomForestRegressor(nt, max_features=the_forest.max_features)
	start = time.time()
	sk_forest.fit(features, responses)
	end = time.time()
	times_scikit.append(end-start)


predictions_rfr = np.zeros_like(responses);
predictions_scikit = np.zeros_like(responses);



for i in range(features.shape[0]):
	sample = features[i]
	predictions_rfr[i] = the_forest.predict(sample)[0]
	predictions_scikit[i] = sk_forest.predict(sample)



plt.plot(predictions_rfr, label="rfr predictions")
plt.plot(predictions_scikit, label="scikit predictions")
plt.plot(responses, label="responses")
plt.xlabel("index")
plt.ylabel("response value")


plt.legend()


plt.figure()
plt.plot(np.sort((responses-predictions_rfr)/responses), label="rfr")
plt.plot(np.sort((responses-predictions_scikit)/responses), label="scikit")
plt.xlabel("sorted index")
plt.ylabel("relative error")


plt.legend()
plt.figure()



plt.xlabel("number of trees")
plt.ylabel("time to train")
plt.scatter(num_trees, times_rfr, color='blue', label="rfr")
plt.scatter(num_trees, times_scikit, color='red', label="scikit")

plt.legend()
plt.show()
