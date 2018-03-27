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
		self.X = [
			[0., 0., 0.],
			[0., 0., 0.],
			[0., 0., 0.],
			[0., 0., 1.],
			[0., 0., 1.],
			[0., 0., 1.],
			[0., 1., 0.],
			[0., 1., 0.],
			[0., 1., 0.],
			[0., 1., 1.],
			[0., 1., 1.],
			[0., 1., 1.],
			[1., 0., 0.],
			[1., 0., 0.],
			[1., 0., 0.],
			[1., 0., 1.],
			[1., 0., 1.],
			[1., 0., 1.],
			[1., 1., 0.],
			[1., 1., 0.],
			[1., 1., 0.],
			[1., 1., 1.],
			[1., 1., 1.],
			[1., 1., 1.]
		]
		self.y = [
			[50],
			[50],
			[50],
			[.2],
			[.2],
			[.2],
			[9],
			[9],
			[9],
			[9.2],
			[9.2],
			[9.2],
			[500],
			[500],
			[500],
			[10.2],
			[10.2],
			[10.2],
			[109.],
			[109.],
			[109.],
			[100],
			[100],
			[100]
		]
		self.y_dual = list(map(lambda x: [math.log10(x[0]), x[0]], self.y))
		bounds = [(0, float('nan')), (0, float('nan')), (0, float('nan'))]
		def init_data(X, y, bounds):
			data = reg.default_data_container(len(X[0]))

			for i, (mn, mx) in enumerate(bounds):
				if math.isnan(mx):
					data.set_type_of_feature(i, mn)
				else:
					data.set_bounds_of_feature(i, mn, mx)

			for row_X, row_y in zip(X, y):
				data.add_data_point(row_X, row_y)
			return data
		self.data = init_data(self.X, self.y, bounds)
		self.data_dual = init_data(self.X, self.y_dual, bounds)

		self.forest = reg.binary_rss_forest()
		self.forest.options.num_trees = 64
		self.forest.options.do_bootstrapping = True
		self.forest.options.num_data_points_per_tree = 200
		self.forest.options.compute_law_of_total_variance = True

		self.assertEqual(self.forest.options.num_trees, 64)
		self.assertTrue (self.forest.options.do_bootstrapping)
		self.assertEqual(self.forest.options.num_data_points_per_tree, 200)
		self.assertTrue(self.forest.options.compute_law_of_total_variance)

		self.rng = reg.default_random_engine(1)
	
	def tearDown(self):
		self.data = None
		self.data_dual = None
		self.forest = None
	
	def test_prediction(self):
		# doesn't really do anything, but calling the fit and predict methods
		self.forest.fit(self.data, self.rng)
		self.forest.predict( self.data.retrieve_data_point(0))

	def test_prediction_dual_fit(self):
		self.forest.fit(self.data_dual, self.rng)
		# Forest is completely overfit thus pred will have to equal pred_val.
		pred = self.forest.predict(self.data_dual.retrieve_data_point(0))
		fit_val = self.data_dual.response(0)
		pred_val = self.data_dual.predict_value(0)
		self.assertEqual(pred, pred_val)
		self.assertNotEqual(fit_val, pred_val)
		
	
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
			self.assertGreaterEqual( round(v, 4), round(cov, 4))

			kernel = self.forest.kernel(datum, datum)
			self.assertEqual(kernel, 1)
	


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
