#ifndef RFR_REGRESSION_FOREST_HPP
#define RFR_REGRESSION_FOREST_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <functional>  // std::bind


#include "cereal/cereal.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <iostream>
#include <sstream>



#include "rfr/trees/tree_options.hpp"
#include "rfr/forests/forest_options.hpp"


namespace rfr{ namespace forests{

typedef cereal::PortableBinaryInputArchive iarch_type;
typedef cereal::PortableBinaryOutputArchive oarch_type;



template <typename tree_type, typename rng_type, typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class regression_forest{
  private:
	forest_options<num_type, response_type, index_type> forest_opts;
	std::vector<tree_type> the_trees;
	index_type num_features;

  public:

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( forest_opts, the_trees);
	}


	regression_forest(){}


	regression_forest(forest_options<num_type, response_type, index_type> forest_opts): forest_opts(forest_opts){
		the_trees.resize(forest_opts.num_trees);
	}



	/* \brief growing the random forest for a given data set
	 * 
	 * \param data a filled data container
	 * \param rng the random number generator to be used
	 */
	void fit(const rfr::data_containers::data_container_base<num_type, response_type, index_type> &data, rng_type &rng){

		if ((!forest_opts.do_bootstrapping) && (data.num_data_points() < forest_opts.num_data_points_per_tree)){
			std::cout<<"You cannot use more data points per tree than actual data point present without bootstrapping!";
			return;
		}

		std::vector<index_type> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);
		std::vector<index_type> data_indices_to_be_used( forest_opts.num_data_points_per_tree);
	  
		num_features = data.num_features();
		
		for (auto &tree : the_trees){
			// prepare the data(sub)set
			if (forest_opts.do_bootstrapping){
				std::uniform_int_distribution<index_type> dist (0,data.num_data_points()-1);
				auto dice = std::bind(dist, rng);
				std::generate_n(data_indices_to_be_used.begin(), data_indices_to_be_used.size(), dice);
			}
			else{
				std::shuffle(data_indices.begin(), data_indices.end(), rng);
				data_indices_to_be_used.assign(data_indices.begin(), data_indices.begin()+ forest_opts.num_data_points_per_tree);
			}
			tree.fit(data, forest_opts.tree_opts, data_indices_to_be_used, rng);
		}
	}


	/* \brief combines the prediction of all trees in the forest
	 *
	 * Every random tree makes an individual prediction. From that, the mean and the standard
	 * deviation of those predictions is calculated. (See Frank's PhD thesis section 11.?)
	 *
	 * \param feature_vector a valid (size and values) array containing features
	 *
	 * \return std::pair<num_type, num_type> mean and total variance (= mean of variances + variance of means )
	 */
	std::pair<num_type, num_type> predict_mean_var( num_type * feature_vector){

		num_type sum_mean(0), sum_var(0);
		std::vector<num_type> means;
		means.reserve(the_trees.size());
		
		for (auto &tree: the_trees){
			num_type m , v;
			index_type n;

			std::tie(m, v, n) = tree.predict_mean_var_N(feature_vector);
			
			sum_mean += m;
			sum_var += v;
			means.push_back(m);
		}
		
		num_type N = num_type(means.size());
		num_type mean_p = sum_mean / N;

		num_type mean_of_vars = sum_var/N;
		
		num_type var_of_means = 0;
		for (auto &m: means)
			var_of_means += (m - mean_p);
		var_of_means /= N; 
		
		return(std::pair<num_type, num_type> (mean_p, std::max<num_type>(0, mean_of_vars + var_of_means)));
	}


	/* \brief combines the prediction of all trees in the forest
	 *
	 * Every random tree makes an individual prediction. From that, the mean and the standard
	 * deviation of those predictions is calculated. (See Frank's PhD thesis section 11.?)
	 *
	 * \param feature_vector a valid (size and values) array containing features
	 *
	 * \return std::pair<num_type, num_type> mean and sqrt(total variance = mean of variances + variance of means )
	 */
	std::pair<num_type, num_type> predict_mean_std( num_type * feature_vector){
		auto p = predict_mean_var(feature_vector);
		p.second = sqrt(p.second);
		return(p);
	}


	/* \brief predict the mean and the variance deviation for a configuration marginalized over a given set of partial configurations
	 * 
	 * This function will be mostly used to predict the mean over a given set of instances, but could be used to marginalize over any discrete set of partial configurations.
	 * 
	 * \param features a (partial) configuration where unset values should be set to NaN
	 * \param set_features a array containing the (partial) assignments used for the averaging. Every NaN value will be replaced by the corresponding value from features.
	 * \param set_size number of feature vectors in set_features
	 * 
	 * \return std::pair<num_type, num_type> mean and variance prediction
	 */
	std::pair<num_type, num_type> predict_mean_var_marginalized_over_set (num_type *features, num_type* set_features, index_type set_size){
		
		num_type fv[num_features];
		
		num_type sum_mean(0), sum_var(0);
		std::vector<num_type> means;
		means.reserve(set_size);
		
		for (auto i=0u; i < set_size; ++i){
			// construct the single feature vector
			std::copy_n(features,num_features, fv);
			for (auto j=0u; j <num_features; ++j){
				if (!isnan(set_features[i*num_features + j]))
					fv[j] = set_features[i*num_features + j];
			}
			num_type m , v;
			index_type n;
			std::tie(m, v, n) = predict_mean_var(fv);
			
			sum_mean += m;
			sum_var += v;
			means.emplace_back(m);
		}
		
		// compute the mean and the parts of the total variance
		num_type N = num_type(set_size);
		num_type mean_p = sum_mean / N;

		num_type mean_of_vars = sum_var/N;
		
		num_type var_of_means = 0;
		for (auto &m: means)
			var_of_means += (m - mean_p);
		var_of_means /= N; 
		
		return(std::pair<num_type, num_type> (mean_p, std::max<num_type>(0, mean_of_vars + var_of_means)));
	}


	/* \brief estimates the covariance of two feature vectors
	 * 
	 * The covariance between to input vectors contains information about the
	 * feature space. For other models, like GPs, this is a natural quantity
	 * (e.g., property of the kernel). Here, we try to estimate it using the
	 * total covariance of the individual tree's prediction and the assumption
	 * that cov(X,Y) = 0, if X and Y fall into different leafs, and cov(X,Y) = var(X) = var(Y)
	 * otherwise.
	 * 
	 * \param f1 a valid feature vector (no sanity checks are performed!)
	 * \param f2 a second feature vector (no sanity checks are performed!)
	 */
	num_type covariance (num_type* f1, num_type* f2){
		std::vector<num_type> means1, means2;
		means1.reserve(the_trees.size());
		means2.reserve(the_trees.size());
		
		num_type sum_mean1(0), sum_mean2(0), sum_cov(0); 
		
		for (auto &t: the_trees){
			auto l1 = t.find_leaf(f1);
			auto l2 = t.find_leaf(f2);
			
			num_type m , v;
			index_type n;

			std::tie(m, v, n) = t.predict_mean_var_N(f1);
			sum_mean1 += m;
			means1.push_back(m);

			std::tie(m, v, n) = t.predict_mean_var_N(f2);
			sum_mean2 += m;
			means2.push_back(m);
			
			// assumption here: cov = 0 if the leafs are different, and cov = var if both feature vectors fall into the same leaf
			if (l1 == l2)
				sum_cov += v;
		}
		
		
		num_type N = num_type (the_trees.size());
		num_type mean_covs = sum_cov/N;
		num_type m1 = sum_mean1/N, m2 = sum_mean2/N;
		
		num_type cov_means = 0;
		
		for (auto i=0u; i < the_trees.size(); ++i)
			cov_means += (means1-m1)*(means2-m2);

		cov_means /= N; 

		return(mean_covs + cov_means);
	}


	std::vector< std::vector<num_type> > all_leaf_values (num_type * feature_vector){
		std::vector< std::vector<num_type> > rv;
		rv.reserve(the_trees.size());

		for (auto &t: the_trees){
			rv.push_back(t.leaf_entries(feature_vector));
		}
		return(rv);
	}


	void save_to_binary_file(const std::string filename){
		std::ofstream ofs(filename, std::ios::binary);
		oarch_type oarch(ofs);
		serialize(oarch);
	}


	void load_from_binary_file(const std::string filename){
		std::ifstream ifs(filename, std::ios::binary);
		std::cout<<"opening file "<<filename<<std::endl;
		iarch_type iarch(ifs);
		serialize(iarch);
	}


	std::string save_into_string(){
		std::stringstream oss;
		oarch_type oarch(oss);
		serialize(oarch);
		return(oss.str());
	}

	void load_from_string( std::string str){
		std::stringstream iss;
		iss.str(str);
		iarch_type iarch(iss);
		serialize(iarch);
	}


	forest_options<num_type, response_type, index_type> get_forest_options(){return(forest_opts);}

	/* \brief stores a latex document for every individual tree
	 * 
	 * \param filename_template a string to specify the location and the naming scheme. Note the directory is not created, so make sure it exists.
	 * 
	 */
	void save_latex_representation(const std::string filename_template){
		for (auto i = 0u; i<the_trees.size(); i++){
			std::stringstream filename;
			filename << filename_template<<i<<".tex";
			the_trees[i].save_latex_representation(filename.str().c_str());
		}
	}

	void print_info(){
		for (auto t: the_trees){
			t.print_info();
		}
	}
};


}}//namespace rfr::forests
#endif
