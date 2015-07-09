import numpy as np
import itertools



features = np.loadtxt("toy_data_set_features.csv", delimiter=",")
responses= np.loadtxt("toy_data_set_responses.csv", delimiter=",")

indices = np.arange(len(responses))

def calc_loss(l_indices, r_indices):
	l1 = np.var(responses[l_indices])*len(l_indices)
	l2 = np.var(responses[r_indices])*len(r_indices)
	return(l1+l2)




# find best split for the continuous first feature
def loss1 (split_value):
	left_indices = indices[(features[:,0] <= split_value)]
	right_indices= indices[(features[:,0] > split_value)]
	if (len(left_indices)*len(right_indices) == 0):
		return np.inf
	return(calc_loss(left_indices, right_indices))
	
losses = list(map(loss1, features[:,0]))
best_index = (np.argmin(losses))

print(losses[best_index], features[best_index,0])
tmp = np.array(features[:,0] <= features[best_index,0],dtype=np.int)
print("Reference for the () operator")
print(",".join(map(str, tmp)))


# find best split for the categorical second feature

print(list(map(lambda i: np.mean(responses[indices[features[:,1] == i]]), range(1,4))))
print("The way the C++ implementation works, the category with the smallest mean will always fall into the left child!")
print("here this is category 2")

for conf in (itertools.product([0,1], [0], [0,1])):
	left_indices = np.array([], dtype=np.int);
	right_indices = np.array([], dtype=np.int);
	print(conf)
	
	if not any(conf) or all(conf):
		print("skip")
		continue
	
	for i in range(0,3):
		if conf[i] == 1:
			left_indices = np.hstack([left_indices, indices[features[:,1]==i+1] ])
		else:
			right_indices = np.hstack([right_indices, indices[features[:,1]==i+1] ])
	print(calc_loss(left_indices, right_indices))

print("So the final split set has to contain 1 and 2")
tmp = np.array(features[:,1] <= 2,dtype=np.int)
print("Reference for the () operator")
print(",".join(map(str, tmp)))



