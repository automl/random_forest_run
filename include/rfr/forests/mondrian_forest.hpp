#ifndef RFR_MONDRIAN_FOREST_HPP
#define RFR_MONDRIAN_FOREST_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>


#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>

#include <iostream>
#include <sstream>



#include "rfr/trees/tree_options.hpp"
#include "rfr/forests/forest_options.hpp"
#include "rfr/util.hpp"

namespace rfr{ namespace forests{

typedef cereal::PortableBinaryInputArchive binary_iarch_t;
typedef cereal::PortableBinaryOutputArchive binary_oarch_t;

typedef cereal::JSONInputArchive ascii_iarch_t;
typedef cereal::JSONOutputArchive ascii_oarch_t;



template <typename tree_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_t=std::default_random_engine>
class mondrian_forest{
  protected:
	std::vector<tree_t> the_trees;
	index_t num_features;

	std::vector<std::vector<num_t> > bootstrap_sample_weights;
	
	num_t oob_error = NAN;
	
	// the forest needs to remember the data types on which it was trained
	std::vector<index_t> types;

  public:
  	std::vector<tree_t> get_trees()const {return the_trees;}

	forest_options<num_t, response_t, index_t> options;
	index_t internal_index = 0;
	std::string name = "";
	//forest_options<num_t, response_t, index_t> options;


  	/** \brief serialize function for saving forests with cerial*/
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( options, the_trees, num_features, bootstrap_sample_weights, oob_error, types/*, bounds*/);
	}

	mondrian_forest(): options()	{}
	
	//mondrian_forest(mondrian_forest_options<num_t, response_t, index_t> opts): options(opts){}
	mondrian_forest(forest_options<num_t, response_t, index_t> opts): options(opts){}


	virtual ~mondrian_forest()	{};

	/**\brief growing the random forest for a given data set
	 * 
	 * \param data a filled data container
	 * \param rng the random number generator to be used
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_t &rng){

		if (options.num_trees <= 0)
			throw std::runtime_error("The number of trees has to be positive!");

		if ((!options.do_bootstrapping) && (data.num_data_points() < options.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");

		the_trees.resize(options.num_trees);


		std::vector<index_t> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);

		types.resize(data.num_features());
		//bounds.resize(data.num_features());

		num_features = data.num_features();
		
		// catch some stupid things that will make the forest crash when fitting
		if (options.num_data_points_per_tree == 0)
			throw std::runtime_error("The number of data points per tree is set to zero!");
		
		if (options.tree_opts.max_features == 0)
			throw std::runtime_error("The number of features used for a split is set to zero!");
		
		bootstrap_sample_weights.clear();

		for (auto &tree : the_trees){
            std::vector<num_t> bssf (data.num_data_points(), 0); // BootStrap Sample Frequencies
			// prepare the data(sub)set
			if (options.do_bootstrapping){
                std::uniform_int_distribution<index_t> dist (0,data.num_data_points()-1);
				index_t die = dist(rng);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i){
					die = dist(rng);
                    ++bssf[die];
				}
			}
			else{
				std::shuffle(data_indices.begin(), data_indices.end(), rng);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i)
                    ++bssf[data_indices[i]];
			}
			
			tree.fit(data, options.tree_opts, bssf, rng);
			
			// record sample counts for later use
			if (options.compute_oob_error)
				bootstrap_sample_weights.push_back(bssf);
		}
		
		oob_error = NAN;
		index_t amount_obb_test = 0;
		num_t pred;
		bool bootstrapable = false;
		if (options.compute_oob_error){
			
			rfr::util::running_statistics<num_t> oob_error_stat;
			
			for (auto i=0u; i < data.num_data_points(); i++){

				rfr::util::running_statistics<num_t> prediction_stat;

				for (auto j=0u; j<the_trees.size(); j++){
					// only consider data points that were not part of that bootstrap sample
					if (bootstrap_sample_weights[j][i] == 0){
						amount_obb_test++;
						pred = the_trees[j].predict( data.retrieve_data_point(i));
						prediction_stat.push(pred);
						bootstrapable = true;
					}
				}
				if(bootstrapable){
					// compute squared error of prediction
					oob_error_stat.push(std::pow(prediction_stat.mean() - data.response(i), 2));	
					oob_error = std::sqrt(oob_error_stat.mean());
				}
				bootstrapable = false;
				
			}
			oob_error = std::sqrt(oob_error_stat.mean());
		}
	}

   /* \brief makes a prediction for the mean and a variance estimation
    *
    *
    * 
    * Every tree returns the mean and the variance of the leaf the feature vector falls into.
    * These are combined to the forests mean prediction (mean of the means) and a variance estimate
    * (mean of the variance + variance of the means).
    * 
    * Use weighted_data = false if the weights assigned to each data point were frequencies, not importance weights.
    * Use this if you haven't assigned any weigths, too.
    * 
	* \param feature_vector a valid feature vector
	* \param weighted_data whether the data had importance weights
	* \return std::pair<response_t, num_t> mean and variance prediction
    */
    std::pair<num_t, num_t> predict_mean_var( const std::vector<num_t> &feature_vector){

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> var_stats, mean_stats;
		for (auto &tree: the_trees){
			auto mv = tree.predict_mean_var(feature_vector);
			mean_stats.push(mv.first);
			var_stats.push(mv.second);
		}

		return(std::pair<num_t, num_t>(mean_stats.mean(), var_stats.mean()));
	}

	response_t predict(const std::vector<num_t> &feature_vector) const{

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats;
		for (auto &tree: the_trees)
			mean_stats.push(tree.predict(feature_vector));
		return(mean_stats.mean());
	}
    

	response_t predict_median( const std::vector<num_t> &feature_vector){

		// collect the predictions of individual trees
		response_t pred;
		std::vector<response_t> preds, means, sds;
		for (index_t i = 0; i<the_trees.size(); i++){
			pred = the_trees[i].predict(feature_vector);
			preds.emplace_back(pred);

		}
		std::sort (preds.begin(), preds.end()); 

		if(the_trees.size()%2){
			return ((preds[the_trees.size()/2] + preds[the_trees.size()/2 + 1])/2);
		}
		else{
			return(preds[the_trees.size()/2 + 1]);
		}
	}


	virtual void partial_fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_t &rng, index_t point){

		if (options.num_trees <= 0)
			throw std::runtime_error("The number of trees has to be positive!");

		if ((!options.do_bootstrapping) && (data.num_data_points() < options.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");

		the_trees.resize(options.num_trees);


		std::vector<index_t> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);

		types.resize(data.num_features());
		//bounds.resize(data.num_features());

		num_features = data.num_features();
		
		// catch some stupid things that will make the forest crash when fitting
		if (options.num_data_points_per_tree == 0)
			throw std::runtime_error("The number of data points per tree is set to zero!");
		
		if (options.tree_opts.max_features == 0)
			throw std::runtime_error("The number of features used for a split is set to zero!");
		
		for (auto &tree : the_trees){
			tree.partial_fit(data, options.tree_opts, point, rng);
		}
		oob_error = NAN;
	}
    

	// check

    

	num_t out_of_bag_error(){return(oob_error);}

	/* \brief writes serialized representation into a binary file
	 * 
	 * \param filename name of the file to store the forest in. Make sure that the directory exists!
	 */
	void save_to_binary_file(const std::string filename){
		std::ofstream ofs(filename, std::ios::binary);
		binary_oarch_t oarch(ofs);
		serialize(oarch);
	}

	/* \brief deserialize from a binary file created by save_to_binary_file
	 *
	 * \param filename name of the file in which the forest is stored. 
	 */
	void load_from_binary_file(const std::string filename){
		std::ifstream ifs(filename, std::ios::binary);
		binary_iarch_t iarch(ifs);
		serialize(iarch);
	}

	/* serialize into a string; used for Python's pickle.dump
	 * 
	 * \return std::string a JSON serialization of the forest
	 */
	std::string ascii_string_representation(){
		std::stringstream oss;
		{
			ascii_oarch_t oarch(oss);
			serialize(oarch);
		}
		return(oss.str());
	}

	/* \brief deserialize from string; used for Python's pickle.load
	 * 
	 * \return std::string a JSON serialization of the forest
	 */
	void load_from_ascii_string( std::string const &str){
		std::stringstream iss;
		iss.str(str);
		ascii_iarch_t iarch(iss);
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
		for (auto &t: the_trees){
			t.print_info();
		}
	}


	virtual unsigned int num_trees (){ return(the_trees.size());}
};


}}//namespace rfr::forests
#endif
