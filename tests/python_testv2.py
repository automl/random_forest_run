import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import pyrfr.data_containers


data_set_prefix = '../test_data_sets/diabetes_'



# feature matrix contains one data point per row
features  = np.loadtxt(data_set_prefix+'features.csv', delimiter=",")

# the responses come in a 1d array
responses =  np.loadtxt(data_set_prefix+'responses.csv', delimiter=",")

# the types have the following meaning:
#	0 - this variable is continuous
#  >0 - the number of values from {1, 2, ...} this variable can take
types = np.zeros([features.shape[1]],dtype=np.uint)





# the simplest data container to wrap the numpy arrays
# note: no copy of the data is made, so be sure that the numpy matrices
#		still exist when the random forest is fitted
#data1 = pyrfr.data_containers.numpy_container_regression(features, responses, types)
data1 = pyrfr.data_containers.numpy_container_regression(np.loadtxt(data_set_prefix+'features.csv', delimiter=","),  np.loadtxt(data_set_prefix+'responses.csv', delimiter=","), np.zeros([features.shape[1]],dtype=np.uint))

print(data1.num_features(), data1.num_data_points())
print(np.allclose(data1.retrieve_data_point(0)- features[0],0))
data1.add_data_point(np.random.rand(10), 1)
print(data1.num_features(), data1.num_data_points())
print(data1.retrieve_data_point(-2))



# the third container, that lives completely in the C++ code
# the only argument the constructor takes is the number of features
data2 = pyrfr.data_containers.mostly_continuous_data_regression(features.shape[1])

# you can set the type of each feature like that:
data2.set_type_of_feature(0,5) #arguments are "feature index" and "type"
print(data2.get_type_of_feature(0))
# define each type before you add any data points



# the container can import numpy arrays (a copy of the data will be made)
data3.import_numpy_arrays(features, responses);



# the other way of feeding data points into the container is this:
data4 = rfr.data_container.mostly_continuous_data_regression(features.shape[1])
for i in range(features.shape[0]):
	data4.add_data_point( features[i], responses[i])



if np.allclose(data3.export_responses(),responses) and  np.allclose(data3.export_features(),features):
	print("Import of data into data3 was successful")

if np.allclose(data4.export_responses(),responses) and  np.allclose(data4.export_features(),features):
	print("Import of data into data4 was successful")


# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = rfr.regression.binary_rss()

the_forest.num_trees = 2


# the forest's parameters
the_forest.seed=12					# reset to reseed the rng for the next fit
the_forest.do_bootstrapping=True	# default: false


the_forest.num_data_points_per_tree=0 # means same number as data points

the_forest.max_features_per_split = features.shape[1]//2 # 0 would mean all the features
the_forest.min_samples_to_split = 0	# 0 means split until pure
the_forest.min_samples_in_leaf = 0	# 0 means no restriction 
the_forest.max_depth=0				# 0 means no restriction
the_forest.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon



the_forest.fit(data1)

# you can save a LaTeX document that you can compile with pdflatex
the_forest.save_latex_representation("/tmp/rfr_test")


# the predict method will return a tuple containing the predicted mean and the standard deviation.
print(the_forest.predict(features[0]))




# let's play around a bit and train different numbers of trees and compare the speed to scikit learn
# Note: the dataset here is pretty small, so the results may not by representative
times_rfr = []
times_scikit=[]
num_trees = [1, 2, 4, 8, 16, 32, 64, 128,256]

for nt in num_trees:

	the_forest.num_trees=nt

	start = time.time()

	# this is how you fit the forest, it should work with any data container
	the_forest.fit(data3)

	end = time.time()
	times_rfr.append(end-start)

	sk_forest = RandomForestRegressor(nt, max_features=the_forest.max_features_per_split)
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
