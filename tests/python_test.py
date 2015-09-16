
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import rfr


#data_set_prefix = '../../test_data_sets/toy_data_set_'
#types = np.array([0,4],dtype=np.uint32)


data_set_prefix = '../test_data_sets/diabetes_'




features  = np.loadtxt(data_set_prefix+'features.csv', delimiter=",")
responses =  np.loadtxt(data_set_prefix+'responses.csv', delimiter=",")



types = np.zeros([features.shape[1]],dtype=np.uint32)




print(features.shape)

the_forest = rfr.regression.binary_rss()



the_forest.seed=12

the_forest.do_bootstrapping=True
the_forest.num_data_points_per_tree=0
the_forest.max_features_per_split = 10
the_forest.min_samples_to_split = 2
the_forest.min_samples_in_leaf = 1
the_forest.max_depth=0
the_forest.epsilon_purity = 1e-8


times_rfr = []
times_scikit=[]
num_trees = [1, 2, 4, 8]#, 16, 32, 64, 128,256]






for nt in num_trees:
	the_forest.num_trees=nt

	start = time.time()
	the_forest.fit(features,  responses, types)
	end = time.time()
	times_rfr.append(end-start)


	sk_forest = RandomForestRegressor(nt, max_features=the_forest.max_features_per_split)
	start = time.time()
	sk_forest.fit(features, responses)
	end = time.time()
	times_scikit.append(end-start)



the_forest.save_latex_representation("/tmp/rfr_test")


predictions_rfr = np.zeros_like(responses);
predictions_scikit = np.zeros_like(responses);



for i in range(features.shape[0]):
	sample = features[i]
	predictions_rfr[i] = the_forest.predict(sample)[0]
	predictions_scikit[i] = sk_forest.predict(sample)



plt.plot(predictions_rfr, label="rfr predictions")
plt.plot(predictions_scikit, label="scikit predictions")
plt.plot(responses, label="responses")

plt.legend()

plt.figure()
plt.plot(np.sort((responses-predictions_rfr)/responses), label="rfr")

plt.plot(np.sort((responses-predictions_scikit)/responses), label="scikit")

plt.legend()
plt.figure()


plt.plot(num_trees, times_rfr, label="rfr")
plt.plot(num_trees, times_scikit, label="scikit")

plt.legend()

plt.show()
