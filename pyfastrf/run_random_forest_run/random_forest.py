'''
Created on May 28, 2015

@author: Aaron Klein
'''

import numpy as np

from tree import Tree


class RandomForest(object):

    def __init__(self, seed=42):
        self._seed = seed
        self._trees = []

    def train(self, X, y, do_bootstrapping=True, ratio_features=0.7, num_trees=20, split_min=10):
        """
            Trains the random forest on the given X and y matrix

            Parameters:
            ----------

            X (np.array) : The input data, where each row stands for a data point and each column represents a feature
            y (np.array) : The response values
            do_bootstrapping (bool) : Do bootstrapping in each tree or not
            ratio_features (float) : Ratio of features that will be considered for the split
            num_trees (int) : Number of trees in the random forest
            split_min (int) : Minimum number of data points in a leaf in order to make a new split
        """

        assert X.shape[0] == y.shape[0]

        for i in range(num_trees):
            t = Tree(X, y, do_bootstrapping, ratio_features, split_min)
            self._trees.append(t)

    def train_with_instances(self, Theta, X, idx, y, do_bootstrapping=True, ratio_features=0.7, num_trees=20, split_min=10):
        """
            Trains the random forest on the given Theta, X and y matrix

            Parameters:
            ----------

            Theta (np.array) : The configurations, where each row stands for a configuration and each column represents a parameter
            X (np.array) : The instance features, where each row stands for a instance and each column represents a feature
            idx (np.array) : (N x 2) dimensional matrix that concatenates the Theta matrix with the X matrix. The first entry in each row indicates the row in Theta and
                               the second entry indicates the row in X
            y (np.array) : The response values
            do_bootstrapping (bool) : Do bootstrapping in each tree or not
            ratio_features (float) : Ratio of features that will be considered for the split
            num_trees (int) : Number of trees in the random forest
            split_min (int) : Minimum number of data points in a leaf in order to make a new split
        """

        assert idx.shape[0] == y.shape[0]

        data = np.concatenate((Theta[idx[:, 0]], X[idx[:, 1]]), axis=1)
        self.train(data, y, do_bootstrapping, ratio_features, num_trees, split_min)

    def predict(self, x):
        """
            Returns for a test point x the empirical mean and variance of the single tree predictions

            Parameters:
            ----------

            x (np.array) : The test data point

            Returns:
            -------

            mean (np.array) : The empirical mean
            var (np.array) : The empirical variance

        """

        predictions = self.predict_each_tree(x)
        return np.mean(predictions), np.var(predictions)

    def predict_with_instances(self, theta, x):
        """
            Returns for a test configuration theta and an instance feature x the empirical mean and variance of the single tree predictions

            Parameters:
            ----------

            theta (np.array) : The test configuration
            x (np.array) : The instance feature

            Returns:
            -------

            mean (np.array) : The empirical mean
            var (np.array) : The empirical variance
        """

        return self.predict(np.concatenate((theta, x), axis=0))

    def predict_each_tree(self, x):
        """
            Returns for a test point x the single tree predictions

            Parameters:
            ----------

            x (np.array) : The test data point

            Returns:
            -------

            predictions (np.array) : A (num_trees) dimensional vector where the i'th entry is the prediction of i'th tree

        """
        predictions = np.zeros([len(self._trees)])
        for idx, tree in enumerate(self._trees):
            predictions[idx] = tree.predict(x)
        return predictions

    def predict_each_tree_with_instances(self, theta, x):
        """
            Returns for a test configuration theta and an instance feature x the single tree predictions

            Parameters:
            ----------

            theta (np.array) : The test configuration
            x (np.array) : The instance feature

            Returns:
            -------

            predictions (np.array) : A (num_trees) dimensional vector where the i'th entry is the prediction of i'th tree

        """
        return self.predict_each_tree(np.concatenate((theta, x), axis=0))
