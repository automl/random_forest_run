import sys
import os
import csv
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../build")
import pyrfr.regression

print(pyrfr.regression.__file__)
here = os.path.dirname(os.path.realpath(__file__))
data_set_folder = '%(here)s/../test_data_sets/' % {"here":here}


with open(os.path.join(data_set_folder, 'sat_saps_configurations.csv')) as fh:
	reader = csv.reader(fh)
	configurations = np.array(([l[5:] for l in list(reader)[1:]]), dtype=np.float)


print("read {} configurations".format(len(configurations)))


instance_dict = {}
with open(os.path.join(data_set_folder, 'sat_instance_features.csv')) as fh:
	reader = csv.reader(fh)
	#skip header
	next(reader)

	for row in reader:
		instance_dict[row[0]] = np.array(row[1:],dtype=np.double)
	num_instance_features = len(row)-1
	

print("read {} instances with {} features each".format(len(instance_dict), num_instance_features))


performance_matrix_dict = {}
with open(os.path.join(data_set_folder, 'sat_performance_matrix.csv')) as fh:
	reader = csv.reader(fh)
	
	for row in reader:
		performance_matrix_dict[row[0]]=np.array(row[1:],dtype=np.double)


#print([performance_matrix_dict[i] for i in instance_names])


s1 = set(instance_dict)
s2 = set(performance_matrix_dict.keys())
common_instances = s1 & s2

print("{} of the instances have features and runtime data".format(len(common_instances)))


instances = np.zeros([len(common_instances), num_instance_features], dtype=np.double)
performance_matrix = np.zeros([len(configurations),len(common_instances)], dtype=np.double)

print(configurations.shape, instances.shape, performance_matrix.shape)

i=0
for instance in common_instances:
	instances[i] = instance_dict[instance]
	performance_matrix[:,i] = performance_matrix_dict[instance]
	i+=1


data = pyrfr.regression.default_data_container_with_instances(configurations.shape[1], instances.shape[1])


for c in configurations:
	data.add_configuration(c.tolist())

for i in instances:
	data.add_instance(i.tolist())




for i in range(1000):
	i = np.random.randint(performance_matrix.shape[0])
	j = np.random.randint(performance_matrix.shape[1])
	data.add_data_point(i,j, performance_matrix[i,j])


#reset to reseed the rng for the next fit
rng = pyrfr.regression.default_random_engine(42)
# create an instance of a regerssion forest using binary splits and the RSS loss
the_forest = pyrfr.regression.binary_rss_forest()

the_forest.options.num_trees = 16
# the forest's parameters
the_forest.options.do_bootstrapping=True	# default: false
the_forest.options.num_data_points_per_tree=data.num_data_points() # means same number as data points
the_forest.options.tree_opts.max_features = data.num_features()//2 # 0 would mean all the features
the_forest.options.tree_opts.min_samples_to_split = 0	# 0 means split until pure
the_forest.options.tree_opts.min_samples_in_leaf = 0	# 0 means no restriction 
the_forest.options.tree_opts.max_depth=1024			# 0 means no restriction
the_forest.options.tree_opts.epsilon_purity = 1e-8	# when checking for purity, the data points can differ by this epsilon


the_forest.fit(data, rng)
