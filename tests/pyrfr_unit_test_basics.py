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
	

	def test_nondynamic_objects(self):

		def try_set_min_depth(d):
			self.forest.options.tree_opts.min_depth = d
		self.assertRaises(Exception, try_set_min_depth, 5)

		def set_num_features_wrongly(n):
			self.forest.options.num_features = n
		self.assertRaises(Exception, set_num_features_wrongly, 5)


		def add_foo_attr():
			self.forest.foo=5
		self.assertRaises(Exception, add_foo_attr)



	def test_data_container(self):
		data = reg.default_data_container(10)

		data.add_data_point([1]*10, 2)
		data.retrieve_data_point(0)


if __name__ == '__main__':
	unittest.main()
