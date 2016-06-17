#ifndef RFR_BINARY_SPLIT_RSS_HPP
#define RFR_BINARY_SPLIT_RSS_HPP

#include <vector>
#include <bitset>
#include <array>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>


#include "cereal/cereal.hpp"
#include <cereal/types/bitset.hpp>
#include <cereal/types/memory.hpp>
#include <memory>
//#include <cereal/types/polymorphic.hpp>
#include "rfr/data_containers/data_container_base.hpp"
#include "rfr/splits/split_base.hpp"
#include "rfr/data_containers/data_container_utils.hpp"
namespace rfr{ namespace splits{



template <typename rng_type, typename num_type = float, typename response_type=float, typename index_type = unsigned int, unsigned int max_num_categories = 128>
class binary_split_one_feature_rss_loss: public rfr::splits::k_ary_split_base<2,rng_type, num_type, response_type, index_type> {
  private:
	
	index_type feature_index;	//!< split needs to know which feature it uses
	num_type num_split_value;
	std::bitset<max_num_categories> cat_split_set;
	
  public:
  	
  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive){
		archive( feature_index, num_split_value, cat_split_set); 
	}
  	
  	
	/** \brief the implementation to find the best binary split using only one feature minimizing the RSS loss
	 *
	 * The best binary split is determined among all allowed features. For a continuous feature the split is a single value.
	 * For catergoricals, the split criterion is a "set" (actual implementation might use a different datatype for performance).
	 * In both cases the split is computed as efficiently as possible exploiting properties of the RSS loss (optimal split for categoricals
	 * can be found in polynomial rather than exponential time in the number of possible values).
	 * The constructor assumes that the data should be split. Testing whether the number of points and their values allow further splitting is checked by the tree
	 * 
	 * \param data the container holding the training data
	 * \param features_to_try a vector with the indices of all the features that can be considered for this split
	 * \param indices a vector containing the subset of data point indices to be considered (output!)
	 * \param an iterator into this vector that says where to split the data for the two children
	 *
	 * \return num_type loss of the best found split
	 */
	 virtual num_type find_best_split(	const rfr::data_containers::data_container_base<num_type, response_type, index_type> &data,
										const std::vector<index_type> &features_to_try,
										std::vector<index_type> & indices,
										std::array<typename std::vector<index_type>::iterator, 3> &split_indices_it,
										rng_type &rng){

				
		// gather all the responses into one vector and precompute mean and variance
		num_type sum = 0; num_type sum2= 0;
		std::vector<response_type> responses(indices.size());
		for (auto tmp1=0u; tmp1< indices.size(); tmp1++){
			response_type res = data.response(indices[tmp1]);
			responses[tmp1] = res;
			sum  += res; sum2 += res*res;
		}
		
		
		// tmp vectors to hold the features of the current data-subset and the best so far
		std::vector<num_type> best_features (responses.size());
		num_type best_loss = std::numeric_limits<num_type>::infinity();

		

		for (index_type fi : features_to_try){ //! > uses C++11 range based loop

			num_type loss;
			num_type num_split_copy;
			std::bitset<max_num_categories> cat_split_copy;


			std::vector<num_type> current_features = data.features(fi, indices);

			index_type ft = data.get_type_of_feature(fi);
			// feature_type zero means that it is a continous variable
			if (ft == 0){
				// find best split for the current feature_index
				loss = best_split_continuous(current_features, responses, num_split_copy, sum, sum2, rng);
			}
			// a positive feature type encodes the number of possible values
			if (ft > 0){
				// find best split for the current feature_index
				loss = best_split_categorical(current_features, ft, responses, cat_split_copy, sum, sum2, rng);
			}

			// check if this split is the best so far
			if (loss < best_loss){
				best_loss = loss;
				best_features.swap(current_features);
				feature_index = fi;
				
				if (ft == 0){
					num_split_value = num_split_copy;
				}
				else{
					num_split_value = NAN;
					cat_split_set = cat_split_copy;
				}
					
			}
		}

		if (best_loss < std::numeric_limits<num_type>::infinity()){
			
			// now we have to rearrange the indices based on which leaf they fall into

			// the default values for the two split iterators
			split_indices_it[0] = indices.begin();
			split_indices_it[2] = indices.end();    

			// adapted from http://www.cplusplus.com/reference/algorithm/partition/
			// because std::partition is not usable in the use case here
			auto i_first = indices.begin();
			auto i_last  = indices.end();
			auto f_first = best_features.begin();
			auto f_last  = best_features.end();                     

			while (i_first != i_last){
					while ( !operator()(*f_first)){
							++f_first; ++i_first;
							if (i_first == i_last){
									split_indices_it[1] = i_first;
									return(best_loss);
							}
					}
					do{
							--f_last; -- i_last;
							if (i_first == i_last){
									split_indices_it[1] = i_first;
									return(best_loss);
							}                       
					} while (operator()(*f_last));
					std::iter_swap(i_first, i_last);
					++f_first; ++i_first;
			}
			split_indices_it[1] = i_first;
		}
		return(best_loss);
	}


	/** \brief this operator tells into which child the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return int whether the feature_vector falls into the left (false) or right (true) child
	 */
	virtual index_type operator() (num_type *feature_vector) { return(operator()(feature_vector[feature_index]));}
	
	/** \brief overloaded operator for just the respective feature value instead of the complete vector
	 * 
	 */
	virtual index_type operator() (num_type &feature_value) {
		// categorical feature
		if (std::isnan(num_split_value))
			return(!bool(cat_split_set[ int(feature_value)]));
		// standard numerical feature
		return(feature_value > num_split_value);
	}



	/** \brief member function to find the best possible split for a single (continuous) feature
	 * 
	 * 
	 * \param features a vector with the values for the current feature
	 * \param responses the corresponding response values
	 * \param split_criterion a reference to store the split criterion
	 * \param S_y_right the sum of all the response values
	 * \param S_y2_right the sum of all squared response values
	 * \param rng an pseudo random number generator instance
	 * 
	 * \return float the loss of this split
	 */
	virtual num_type best_split_continuous(	const std::vector<num_type> & features,
									const std::vector<response_type> & responses,
									num_type &split_value,
									num_type S_y_right, num_type S_y2_right,
									rng_type &rng){

		// find the best split by looking at any meaningful value for the feature
		// first some temporary variables
		num_type S_y_left(0), S_y2_left(0);
		num_type N_left(0), N_right(features.size());
		num_type loss, best_loss = std::numeric_limits<num_type>::infinity();;

		std::vector<index_type> tmp_indices(features.size());
		std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

		std::sort(	tmp_indices.begin(), tmp_indices.end(),
					[&features](index_type a, index_type b){return features[a] < features[b];}		//! > uses C++11 lambda function, how exciting :)
		);


		// now we can increase the splitting value to move data points from the right to the left child
		// this way we do not consider a split with everything in the right child
		auto tmp_i = 0u;
		while (tmp_i != tmp_indices.size()){
			num_type psv = features[tmp_indices[tmp_i]]+ 1e-6; // potential split value add small delta for numerical inaccuracy
			// combine data points that are very close
			do {
				// change the Sum(y) and Sum(y^2) for left and right accordingly
				response_type res = responses[tmp_indices[tmp_i]];
				S_y_left  += res;
				S_y_right -= res;

				S_y2_left += res*res;
				S_y2_right-= res*res;
				N_right--;
				N_left++;
				tmp_i++;
			} while ((tmp_i != tmp_indices.size()) && (features[tmp_indices[tmp_i]] - psv <= 0));
			
			// stop if all data points are now in the left child as this is not a meaningful split
			if (N_right == 0) {break;}

			// compute the loss
			loss = (S_y2_left  - (S_y_left *S_y_left )/N_left) 
			     + (S_y2_right - (S_y_right*S_y_right)/N_right);

			// store the best split
			if (loss < best_loss){
				std::uniform_real_distribution<num_type> dist(0.0,1.0);
				best_loss = loss;
				split_value = psv + dist(rng)*(features[tmp_indices[tmp_i]] - psv);
			}
		}
		return(best_loss);
	}

	/** \brief member function to find the best possible split for a single (categorical) feature
	 * 
	 * \param features a vector with the values for the current feature
	 * \param num_categories the feature type (number of different values)
	 * \param responses the corresponding response values
	 * \param split_criterion a reference to store the split criterion
	 * \param S_y_right the sum of all the response values
	 * \param S_y2_right the sum of all squared response values
	 * \param rng an pseudo random number generator instance
	 * 
	 * \return float the loss of this split
	 */
	virtual num_type best_split_categorical(const std::vector<num_type> & features,
									index_type num_categories,
									const std::vector<response_type> & responses,
									std::bitset<max_num_categories> &split_set,
									num_type S_y_right, num_type S_y2_right, rng_type &rng){
		// auxiliary variables
		std::vector<index_type> category_ranking(num_categories);
		std::iota(category_ranking.begin(), category_ranking.end(),0);
		std::vector<index_type> N_points_in_category(num_categories,0);
		std::vector<num_type> S_y(num_categories, 0);
		std::vector<num_type> S_y2(num_categories, 0);

		for (auto i = 0u; i < features.size(); i++){
			// find the category for each entry as a proper int
			//! >assumes that the features for categoricals have been properly rounded so casting them to ints results in the right value!
			int cat = features[i];
			// collect all the data to compute the loss
			S_y[cat]  += responses[i];
			S_y2[cat] += responses[i]*responses[i];
			N_points_in_category[cat] += 1;
		}

		// take care b/c certain categories might not be encountered (maybe there was a split on the same variable further up the tree...)
		// sort the categories by whether there were samples or not
		// std::partition rearranges the data using a boolean predicate into all that evaluate to true in front of all evaluating to false.
		// it even returns an iterator pointing to the first element where the predicate is false, how convenient :)
		auto empty_categories_it = std::partition(category_ranking.begin(), category_ranking.end(),
						[&](index_type a){return(N_points_in_category[a] > 0);});
		
		// sort the categories by their individual mean. only consider the ones with actual specimen here
		std::sort(	category_ranking.begin(), empty_categories_it,
					[&](index_type a, index_type b){return ( (S_y[a]/N_points_in_category[a]) < (S_y[b]/N_points_in_category[b]) );});		// C++11 lambda function, how exciting :)

		//more auxiliary variables
		num_type S_y_left = 0, S_y2_left = 0;
		index_type N_left = 0, N_right= features.size();
		num_type current_loss = 0, best_loss = 0;

		// put one category in the left node
		auto it_best_split = category_ranking.begin();
		S_y_left  = S_y[*it_best_split];
		S_y2_left = S_y2[*it_best_split];

		S_y_right  -= S_y[*it_best_split];
		S_y2_right -= S_y2[*it_best_split];
		
		N_left    = N_points_in_category[*it_best_split];
		N_right   -= N_left;  
		it_best_split++;

		// it can happen that the node is not pure wrt the response, but the
		// feature at hand takes only one value in this node. By setting the
		// best_loss to the largest possible value, this split will not be chosen.
		
		if ( (N_right == 0) || (N_left == 0) )
			best_loss = std::numeric_limits<num_type>::infinity();
		else
			best_loss = (S_y2_right - (S_y_right*S_y_right)/N_right)
							+ (S_y2_left - (S_y_left*S_y_left)/N_left);

		current_loss = best_loss;

		// now move one category at a time to the left child and recompute the loss
		for (auto it1 = it_best_split; it1 != empty_categories_it; it1++){
			S_y_left  += S_y[*it1];
			S_y_right -= S_y[*it1];

			S_y2_left  += S_y2[*it1];
			S_y2_right -= S_y2[*it1];

			N_left  += N_points_in_category[*it1];
			N_right -= N_points_in_category[*it1];


			// catch divide by zero as they are invalid splits anyway
			// becomes important if only one or two categories have specimen here!
			if ( (N_right != 0) && (N_left != 0) ){
				current_loss 	= (S_y2_right - (S_y_right*S_y_right)/N_right)
								+ (S_y2_left - (S_y_left*S_y_left)/N_left);

				// keep the best split
				if (current_loss < best_loss){
					best_loss = current_loss;
					it_best_split = it1;
					it_best_split++;
				}
			}
		}

		// store the split set for the left leaf
		split_set.reset();
		for (auto it1 = category_ranking.begin(); it1 != it_best_split; it1++)
			split_set.set(*it1);

		// add unobserved values randomly to the split_set
		if (empty_categories_it != category_ranking.end()){
			std::bernoulli_distribution dist;

			for (auto it1 = empty_categories_it; it1 != category_ranking.end(); it1++){
				if (dist(rng))
					split_set.set(*it1);
			}
		}
		return(best_loss);
	}


	virtual void print_info(){
		if(std::isnan(num_split_value)){
			std::cout<<"split: f_"<<feature_index<<" in {";
			for (size_t i = 0; i < max_num_categories; i++)
				if (cat_split_set[i]) std::cout<<i<<", ";
			std::cout<<"\b\b}\n";			
		}
		else
			std::cout<<"split: f_"<<feature_index<<" <= "<<num_split_value<<"\n";
	}

	/** \brief member function to create a string representing the split criterion
	 * 
	 * \return std::string a label that characterizes the split
	 */	
	virtual std::string latex_representation(){
		std::stringstream str;

		if (std::isnan(num_split_value)){
			auto i = 0u;
			while (cat_split_set[i] == 0)
				i++;
			str << "$f_{" << feature_index << "} \\in \\{"<<i;
			
			for (i++; i < max_num_categories; i++){
				if (cat_split_set[i])
					str<<i<<" ";
			}
			str << "\\}$";
		}
		else
			str << "$f_{" << feature_index << "}<=" << num_split_value << "$";
		return(str.str());
	}
	
	index_type get_feature_index() {return(feature_index);}
	num_type get_num_split_value() {return(num_split_value);}
	std::bitset<max_num_categories> get_cat_split_set() {return(cat_split_set);}

	std::array<std::vector< std::vector<num_type> >, 2> compute_subspaces( std::vector< std::vector<num_type> > subspace){
	
		//std::array
	
		// if feature is numerical
		if (std::isnan(num_split_value)){
		
		
		}
		else{
			
		}
	}

};


}}//namespace rfr::splits
#endif
