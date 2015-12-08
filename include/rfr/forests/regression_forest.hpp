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



#include "rfr/trees/tree_options.hpp"
#include "rfr/forests/forest_options.hpp"


namespace rfr{ namespace forests{


template <typename tree_type, typename rng_type, typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class regression_forest{
  private:
	forest_options<num_type, response_type, index_type> forest_opts;

	std::vector<tree_type> the_trees;

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
	 * \return std::pair<num_type, num_type> mean and sqrt( total variance = mean of variances + variance of means )
	 */
	std::pair<num_type, num_type> predict_mean_std( num_type * feature_vector){

		num_type sum_mean=0;
		num_type sum_squared_mean = 0;

		num_type sum_var=0;

		for (auto &tree: the_trees){
			num_type m , v;
			index_type n;

			std::tie(m, v, n) = tree.predict_mean_var_N(feature_vector);
			
			sum_mean += m;
			sum_squared_mean += m*m;
			
			sum_var += v;
		}
		
		unsigned int N = the_trees.size();
		return(std::pair<num_type, num_type> (sum_mean/N, sqrt( sum_var/N + sum_squared_mean/N - (sum_mean/N)*(sum_mean/N)    )));
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
		cereal::PortableBinaryOutputArchive oarch(ofs);
		serialize(oarch);
	}


	void load_from_binary_file(const std::string filename){
		std::ifstream ifs(filename, std::ios::binary);
		std::cout<<"opening file "<<filename<<std::endl;
		cereal::PortableBinaryInputArchive iarch(ifs);
		serialize(iarch);
	}


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
