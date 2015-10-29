#ifndef RFR_CLASSIFICATION_FOREST_HPP
#define RFR_CLASSIFICATION_FOREST_HPP

#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>


#include "rfr/trees/tree_options.hpp"
#include "rfr/forests/forest_options.hpp"

namespace rfr{ namespace forests{


template <typename tree_type, typename rng_type, typename num_type = float, typename response_type = unsigned int, typename index_type = unsigned int>
class classification_forest{
  private:
	forest_options<num_type, response_type, index_type> forest_opts;

	std::vector<tree_type> the_trees;

  public:

	classification_forest(forest_options<num_type, response_type, index_type> forest_opts): forest_opts(forest_opts){
		the_trees.resize(forest_opts.num_trees);
	}

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
	 * Every random tree makes an individual prediction. We want the estimated probability of membership
	 * in each of the classes.
	 */
	std::tuple<num_type, num_type> predict_class( num_type * feature_vector){
		int max_class =  *std::max_element(begin(data.response), end(data.response));
    		int min_class =  *std::min_element(begin(data.response), end(data.response));
		int length = max_class - min_class+1;
		std::vector<index_type> classvector(length, 0);
		std::vector<num_type> class_probs(length, 0);
		int N = 0;
		for (auto &tree: the_trees){
			num_type c;

			c = tree.predict_class(feature_vector);
			// recompute the sum and the sum of squared response values 
			classvector[c - min_class] += 1 ;
			N++;
		}
		for (int i = 0; i <length; i++){
			class_probs[i] = classvector[i]/N;
		}
		return(std::tuple<float, float>(class_probs);
	}

	void save_latex_representation(const char* filename_template){
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


}}//namespace rfr
#endif

