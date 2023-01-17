import sys
sys.path.append("${CMAKE_BINARY_DIR}")

import os
import pickle
import tempfile
import unittest
import math

import pyrfr.regression as reg


class TestBinaryRssRegressionForest(unittest.TestCase):

	def setUp(self):
		data_set_prefix = '${CMAKE_SOURCE_DIR}/test_data_sets/'
		self.data = reg.default_data_container(64)
		self.data.import_csv_files(data_set_prefix+'features13.csv', data_set_prefix+'responses13.csv')
		

		self.forest = reg.binary_rss_forest()
		self.forest.options.num_trees = 64
		self.forest.options.do_bootstrapping = True
		self.forest.options.num_data_points_per_tree = 200

		self.assertEqual(self.forest.options.num_trees, 64)
		self.assertTrue (self.forest.options.do_bootstrapping)
		self.assertEqual(self.forest.options.num_data_points_per_tree, 200)

		self.rng = reg.default_random_engine(1)
	
	def tearDown(self):
		self.data = None
		self.forest = None
	
	def test_prediction(self):
		# doesn't really do anything, but calling the fit and predict methods
		self.forest.fit(self.data, self.rng)
		self.forest.predict( self.data.retrieve_data_point(0))
		
	
	def test_first_nearest_neighbor(self):
		# if no bootstrapping is done, the tree gets all the data points,
		# all features are used for every split and all datapoints are unique,
		# a single tree will perfectly recall the datapoints
		self.forest.options.num_trees = 1
		self.forest.options.do_bootstrapping = False
		self.forest.options.num_data_points_per_tree = self.data.num_data_points()
		self.forest.options.tree_opts.max_features = self.data.num_features()

		self.forest.fit(self.data, self.rng)

		self.assertEqual(self.forest.num_trees(), 1)
		for i in range(self.data.num_data_points()):
			self.assertEqual( self.forest.predict( self.data.retrieve_data_point(i)), self.data.response(i))


	def test_oob_error(self):
		self.forest.options.compute_oob_error=True
		self.forest.options.num_data_points_per_tree = self.data.num_data_points()
		self.forest.fit(self.data, self.rng)

		# not really a test for correctness here, but the toy dataset is too
		# small to get reliable OOB and test errors
		self.assertFalse(math.isnan(self.forest.out_of_bag_error()))
		

	def test_covariance(self):
		self.forest.options.compute_oob_error=False
		self.forest.options.num_data_points_per_tree = self.data.num_data_points()
		self.forest.fit(self.data, self.rng)

		for i in range(self.data.num_data_points()):	
			datum =  self.data.retrieve_data_point(i)
			m, v = self.forest.predict_mean_var(datum)
			cov = self.forest.covariance(datum, datum)
			# need to round to get Greater or Allmost equal
			self.assertGreaterEqual( round(v, 5), round(cov,5))

			kernel = self.forest.kernel(datum, datum)
			self.assertEqual(kernel, 1)

	def test_contents(self):
		self.forest.options.num_trees = 10
		self.forest.fit(self.data, self.rng)
		trees = self.forest.get_all_trees()
		self.assertEqual(len(trees), self.forest.options.num_trees)
		self.assertAlmostEqual(trees[0].get_node(0).get_num_split_value(), -0.20652545999380345, 2)
		self.assertAlmostEqual(trees[1].get_node(1).get_num_split_value(), -0.8857842741236657, 2)

	def test_marginalized_over_instances(self):
		def flatten(lists):
			return [i for l in lists for i in l]

		self.forest.options.num_trees = 10
		self.forest.fit(self.data, self.rng)

		data = [self.data.retrieve_data_point(i) for i in range(4)]
		leaf_entries = [self.forest.all_leaf_values(x) for x in data]
		leaf_entries = list(zip(*leaf_entries))  # Entries per tree
		leaf_entries = [flatten(e) for e in leaf_entries]
		gt_means = [sum(entries) / len(entries) for entries in leaf_entries]  # Ground truth

		mean_per_tree = self.forest.predict_marginalized_over_instances(data, False)
		self.assertEqual(len(mean_per_tree), 10)
		for mean_test, mean_gt in zip(mean_per_tree, gt_means):
			self.assertAlmostEqual(mean_test, mean_gt, 6)

	def test_pickling(self):
		
		the_forest = reg.binary_rss_forest()
		the_forest.options.num_trees = 16
		the_forest.options.do_bootstrapping = True
		the_forest.options.num_data_points_per_tree = self.data.num_data_points()		

		self.assertEqual(the_forest.options.num_trees, 16)

		the_forest.fit(self.data,self.rng)
		
		with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
			fname = f.name
			pickle.dump(the_forest, f)


		with open(fname, 'r+b') as fh:
			a_second_forest = pickle.load(fh)
		os.remove(fname)
		
		for i in range(self.data.num_data_points()):
			d = self.data.retrieve_data_point(i)
			self.assertEqual( the_forest.predict(d), a_second_forest.predict(d))


if __name__ == '__main__':
	unittest.main()
