#ifndef RFR_BINARY_SPLIT_RSS_HPP
#define RFR_BINARY_SPLIT_RSS_HPP

#include <vector>
#include <array>
#include <algorithm>

#include "data_containers/data_container_base.hpp"
#include "splits/split_base.hpp"

namespace rfr{


template <typename num_type = float, typename index_type = unsigned int>
class binary_split_one_feature_rss_loss: public rfr::k_ary_split_base<2,num_type, index_type> {
  private:
	
	index_type feature_index;	//!< split needs to know which feature it uses
	
	//!< The split criterion contains its type (first element = 0 for numerical, =1 for categoricals), and the split value in the second/ the categories that fall into the left child respectively
	std::vector<num_type> split_criterion; //!< one could consider to use a dynamically sized array here to save some memory (vector stores size and capacity + it might allocate more memory than needed!)
  public:
  	
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
	 */
	 virtual num_type find_best_split(	const rfr::data_container_base<num_type, index_type> &data,
									const std::vector<index_type> &features_to_try,
									std::vector<index_type> & indices,
									std::array<typename std::vector<index_type>::iterator, 3> &split_indices_it){
		
		print_vector(indices);
		std::vector<index_type> indices_copy(indices);
		num_type best_loss = std::numeric_limits<num_type>::infinity();
		
		for (index_type fi : features_to_try){ //! > uses C++11 range based loop

			std::vector<num_type> split_criterion_copy;
			typename std::vector<index_type>::iterator split_indices_it_copy = indices_copy.begin();
			num_type loss;

			// sort the indices by the value in feature vector
			std::sort(	indices_copy.begin(), indices_copy.end(),
						[&](index_type a, index_type b){return data.feature(fi,a) < data.feature(fi, b);}		//! > uses C++11 lambda function, how exciting :)
			);

			index_type ft = data.get_type_of_feature(fi);
			// feature_type zero means that it is a continous variable
			if (ft == 0){
				split_criterion_copy.assign(2, 0);
				// find best split for the current feature_index
				loss = best_split_continuous(data, fi, split_criterion_copy, indices_copy, split_indices_it_copy);
			}
			// a positive feature type encodes the number of possible values
			if (ft > 0){
				split_criterion_copy.assign(1,ft);
				// find best split for the current feature_index
				loss = best_split_categorical(data,fi, ft, split_criterion_copy, indices_copy, split_indices_it_copy);
			}
			// check if this split is the best so far
			if (loss < best_loss){
				best_loss = loss;
				feature_index = fi;
				split_criterion.swap(split_criterion_copy);
				indices.swap(indices_copy);
				split_indices_it.at(1) = split_indices_it_copy;
			}
		}
		split_indices_it[0] = indices.begin();
		split_indices_it[2] = indices.end();
		std::sort(++split_criterion.begin(), split_criterion.end());
		//split_criterion.shrink_to_fit();
		return(best_loss);
	}


	/** \brief this operator tells into which child the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return int whether the feature_vector falls into the left (true) or right (false) child
	 */
	virtual index_type operator() (num_type *feature_vector) {
		auto it = split_criterion.begin();
		
		// handle categorical features
		if (*it > (num_type) 0){
			// check if the value is contained in the split 'set'
			it++;
			//it = std::find(it, split_criterion.end(), feature_vector[feature_index]);
			//return( it != split_criterion.end());
			return(std::binary_search(it, split_criterion.end(), feature_vector[feature_index]));
		}
		
		// simple case of a numerical feature
		return(feature_vector[feature_index] <= split_criterion[1]);
	}


	/** \brief member function to find the best possible split for a single (continuous) feature
	 * 
	 * 
	 * \param data pointer to the the data container
	 * \param fi the index of the feature to be used
	 * \param split_criterion_copy a reference to store the split criterion
	 * \param indices_copy a const reference to the indices (const b/c it has already been sorted)
	 * \param split_indices_it_copy an iterator that will point to the first element of indices_copy that would go into the right child
	 * 
	 * \return float the loss of this split
	 */
	num_type best_split_continuous( const rfr::data_container_base<num_type, index_type> &data,
									const index_type & fi,
									std::vector<num_type> &split_criterion_copy,
									std::vector<index_type> &indices_copy,
									typename std::vector<index_type>::iterator &split_indices_it_copy){

		// find the best split by looking at any meaningful value for the feature
		// first some temporary variables
		num_type S_y_left(0), S_y2_left(0);
		num_type S_y_right(0), S_y2_right(0);
		num_type N_left(0), N_right(indices_copy.size());
		num_type loss, best_loss = std::numeric_limits<num_type>::infinity();;

		// we start out with everything in the right child
		// so we compute the mean and the variance for that case
		for (auto it = indices_copy.begin(); it != indices_copy.end(); it++){
			S_y_right  += data.response(*it);
			S_y2_right += data.response(*it)*data.response(*it);
		}
		typename std::vector<index_type>::iterator psii = indices_copy.begin();	// potential split index iterator

		// now we can increase the splitting value to move data points from the right to the left child
		// this way we do not consider a split with everything in the right child
		while (psii != indices_copy.end()){
			num_type psv = data.feature(fi, *psii) + 1e-10; // potential split value add small delta for numerical inaccuracy
			// combine data points that are very close
			while ((psii != indices_copy.end()) && (data.feature(fi,*psii) - psv <= 0)){
				// change the Sum(y) and Sum(y^2) for left and right accordingly
				S_y_left  += data.response(*psii);
				S_y_right -= data.response(*psii);

				S_y2_left += data.response(*psii)*data.response(*psii);
				S_y2_right-= data.response(*psii)*data.response(*psii);
				N_right--;
				N_left++;
				psii++;
			}
			// stop if all data points are now in the left child as this is not a meaningful split
			if (N_right == 0) break;

			// compute the loss
			loss = (S_y2_left  - (S_y_left *S_y_left )/N_left) 
			     + (S_y2_right - (S_y_right*S_y_right)/N_right);

			// store the best split
			if (loss < best_loss){
				best_loss = loss;
				split_criterion_copy[1] = psv;
				split_indices_it_copy = psii;
			}
		}
		return(best_loss);
	}

	/** \brief member function to find the best possible split for a single (categorical) feature
	 * 
	 * 
	 * \param data pointer to the the data container
	 * \param fi the index of the feature to be used
	 * \param num_categories how many different values this variable can take
	 * \param split_criterion_copy a reference to store the split criterion
	 * \param indices_copy a const reference to the indices (const b/c it has already been sorted)
	 * \param split_indices_it_copy an iterator that will point to the first element of indices_copy that would go into the right child
	 * 
	 * \return float the loss of this split
	 */
	num_type best_split_categorical(const rfr::data_container_base<num_type, index_type> &data,
									const index_type &fi,
									const index_type &num_categories,
									std::vector<num_type> &split_criterion_copy,
									std::vector<index_type> &indices_copy,
									typename std::vector<index_type>::iterator &split_indices_it_copy){

		// auxiliary variables
		std::vector<index_type> category_ranking(num_categories);
		std::iota(category_ranking.begin(), category_ranking.end(),0);
		std::vector<index_type> N_points_in_category(num_categories,0);
		std::vector<num_type> S_y(num_categories, 0);
		std::vector<num_type> S_y2(num_categories, 0);

		for (size_t i = 0; i< data.num_data_points() ; i++){
			// find the category for each entry as a proper int
			//! >assumes that the features for categoricals have been properly rounded so casting them to ints results in the right value!
			int cat = data.feature(fi,i)-1;	// subtracked a 1 to accomodate for the categorical values starting at 1, but vector indices at zero
			// collect all the data to compute the loss
			S_y[cat]  += data.response(i);
			S_y2[cat] += data.response(i)*data.response(i);
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
		num_type S_y_left = 0, S_y2_left = 0, N_left = 0;
		num_type S_y_right = 0, S_y2_right = 0, N_right= 0;
		num_type current_loss = 0, best_loss = 0;

		// put one category in the left node
		auto it_best_split = category_ranking.begin();
		S_y_left  = S_y[*it_best_split];
		S_y2_left = S_y2[*it_best_split];
		N_left    = N_points_in_category[*it_best_split];
		it_best_split++;

		// the rest goes into the right node
		for (auto it1 = it_best_split; it1!=empty_categories_it; it1++){
			S_y_right  += S_y[*it1];
			S_y2_right += S_y2[*it1];
			N_right    += N_points_in_category[*it1];
		}

		best_loss = (S_y2_right - (S_y_right*S_y_right)/N_right)
						+ (S_y2_left - (S_y_left*S_y_left)/N_left);


		// now move one category at a time to the left child and recompute the loss
		// beware: keep at least one category in the right child
		auto tmp_it = empty_categories_it; tmp_it--;
		for (auto it1 = it_best_split; it1 != tmp_it; it1++){
			S_y_left  += S_y[*it1];
			S_y_right -= S_y[*it1];

			S_y2_left  += S_y2[*it1];
			S_y2_right -= S_y2[*it1];

			N_left  += N_points_in_category[*it1];
			N_right -= N_points_in_category[*it1];

			current_loss 	= (S_y2_right - (S_y_right*S_y_right)/N_right)
							+ (S_y2_left - (S_y_left*S_y_left)/N_left);
			// keep the best split
			if (current_loss < best_loss){
				best_loss = current_loss;
				it_best_split = it1;
				it_best_split++;
			}
		}

		// store the split set for the left leaf
		for (auto it1 = category_ranking.begin(); it1 != it_best_split; it1++)
			split_criterion_copy.push_back(*it1+1);	// add a 1 to accomodate for the categorical values starting at 1, but vector indices at zero

		// add unobserved values randomly to the split_set
		//!<  TODO: consider using one RNG across everything by passing it along.
		if (empty_categories_it != category_ranking.end()){
			std::default_random_engine rng;
			std::bernoulli_distribution dist;

			for (auto it1 = empty_categories_it; it1 != category_ranking.end(); it1++){
				if (dist(rng)){
					split_criterion_copy.push_back(*it1+1);	// add a 1 to accomodate for the categorical values starting at 1, but vector indices at zero
				}
			}
		}

		//rearrange indices according to their category and in which leaf they go
		// std::partition is again exactly what we need here :)
		split_indices_it_copy = std::partition( 
					indices_copy.begin(), indices_copy.end(),
					[&](size_t i){return(std::find(split_criterion_copy.begin(), split_criterion_copy.end(), data.feature(fi, i)) == split_criterion_copy.end());});
		return(best_loss);
	}

	std::vector<num_type> get_split_criterion(){return(split_criterion);}

};


}//namespace rfr
#endif
