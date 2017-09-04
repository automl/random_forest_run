import sys
sys.path.append("../build")
import os
here = os.path.dirname(os.path.realpath(__file__))

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import pyrfr.regression as reg




data_set_prefix = '%(here)s/../test_data_sets/diabetes_' % {"here":here}
data = reg.data_container()
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

exit(0)

# it is possible to get the actual response values from the corresponding
# leaf in each tree that the given feature vector falls into
# The method returns a nested list
print(the_forest.all_leaf_values(features[0]))

# quantile regression forest estimations
#alphas =[0.25, 0.5, 0.75]
#print(the_forest.quantile_rf(features[0], alphas))


# let's play around a bit and train different numbers of trees and compare the speed to scikit learn
# Note: the dataset here is pretty small, so the results may not by representative
times_rfr=[]
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
