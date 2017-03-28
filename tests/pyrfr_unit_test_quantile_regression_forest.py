import sys
sys.path.append("${CMAKE_BINARY_DIR}")

import os
import pickle
import tempfile
import unittest

import pyrfr.regression as reg


class TestQuantileRegressionForest(unittest.TestCase):

	def setUp(self):
		data_set_prefix = '${CMAKE_SOURCE_DIR}/test_data_sets/'
		self.data = reg.default_data_container(64)
		self.data.import_csv_files(data_set_prefix+'features13.csv', data_set_prefix+'responses13.csv')
		
		self.rng = reg.default_random_engine(1)
		self.forest_constructor = reg.qr_forest
	
	def tearDown(self):
		self.data = None
		
	def test_options_constructor(self):
		fopts = reg.forest_opts()
		fopts.num_trees = 16
		fopts.num_data_points_per_tree = self.data.num_data_points()		

		the_forest = self.forest_constructor(fopts)
		the_forest.fit(self.data, self.rng)


	def test_options_member(self):

		the_forest = self.forest_constructor()
		
		fopts = reg.forest_opts()
		the_forest.options.num_trees = 7
		the_forest.options.num_data_points_per_tree = self.data.num_data_points()		

		the_forest.fit(self.data, self.rng)

		self.assertEqual(the_forest.num_trees(), 7)



if __name__ == '__main__':
	unittest.main()
