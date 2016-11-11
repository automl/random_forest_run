import unittest

import sys
sys.path.append("${CMAKE_BINARY_DIR}")

import pyrfr.regression as reg


class TestBinaryRssRegressionForest(unittest.TestCase):

	def setUp(self):
		data_set_prefix = '${CMAKE_SOURCE_DIR}/test_data_sets/'
		self.data = reg.data_container()
		self.data.import_csv_files(data_set_prefix+'features13.csv', data_set_prefix+'responses13.csv')
		
		self.rng = reg.default_random_engine(1)
	
	def tearDown(self):
		self.data = None
	
	def test_prediction(self):
		the_forest = reg.binary_rss_forest()

		the_forest.options.num_trees = 64
		the_forest.options.do_bootstrapping = True
		the_forest.options.num_data_points_per_tree = 200
		
		self.assertEqual(the_forest.options.num_trees, 64)
		self.assertTrue (the_forest.options.do_bootstrapping)
		self.assertEqual(the_forest.options.num_data_points_per_tree, 200)
		
		the_forest.fit(self.data, self.rng)
		
		the_forest.predict( self.data.retrieve_data_point(0))
	
	def test_first_nearest_neightbor(self):
		# if no bootstrapping is done, the tree gets all the data points,
		# all features are used for every split and all datapoints are unique,
		# a single tree will perfectly recall the datapoints
		the_forest = reg.binary_rss_forest()
		the_forest.options.num_trees = 1
		the_forest.options.do_bootstrapping = False
		the_forest.options.num_data_points_per_tree = self.data.num_data_points()
		the_forest.options.tree_opts.max_features = self.data.num_features()

		the_forest.fit(self.data, self.rng)
		
		for i in range(self.data.num_data_points()):
			self.assertEqual( the_forest.predict( self.data.retrieve_data_point(i)), self.data.response(i))
	
	def test_pickling(self):
		import pickle
		import tempfile
		
		the_forest = reg.binary_rss_forest()
		the_forest.options.num_trees = 16
		the_forest.options.do_bootstrapping = True
		the_forest.options.num_data_points_per_tree = self.data.num_data_points()		

		the_forest.fit(self.data,self.rng)
		
		with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
			fname = f.name
			pickle.dump(the_forest, f)


		with open(fname, 'r+b') as fh:
			a_second_forest = pickle.load(fh)
		os.remove(fname)
		
		for i in range(self.data.num_data_points()):
			d = self.data.retrieve_data_point(i)
			self.assertEqual( the_forest.predict(d), a_second_forest(predict(d)))
		


if __name__ == '__main__':
	unittest.main()
