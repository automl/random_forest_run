#ifndef RFR_BINARY_SPLIT_RSS_V2_HPP
#define RFR_BINARY_SPLIT_RSS_V2_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <sstream>
#include <stack>
#include <tuple>
#include <utility>


#include "rfr/data_containers/data_container_base.hpp"
#include "rfr/splits/split_base.hpp"
#include "rfr/data_containers/data_container_utils.hpp"
namespace rfr{ namespace splits{



template <typename ft, typename it>
void print_helper(ft feats, it indices){
	for (auto i=0u; i< indices.size(); i++){
		std::cout<<indices[i]<<" ";
	}
	std::cout<<std::endl;
		for (auto i=0u; i< indices.size(); i++){
		std::cout<<feats[indices[i]]<<" ";
	}
	std::cout<<std::endl;
}

template<typename i_t, typename f_t>
inline void my_qsort (std::vector<i_t> &indices, const std::vector<f_t> &features){
	
	// to store the indices of the subpartitions
	std::stack<i_t> index_stack;

	f_t pivot_f;
	i_t pivot_i;
	int l,r,L=0,R=features.size()-1, m;
	
	while(true){
		//std::cout<<index_stack.size()<<std::endl;
		// handle small arrays with insertion sort
		if ((R-L)< 8){
			//print_helper(features, indices);		
			for(r=L+1; r<=R; r++){
				pivot_i = indices[r];
				pivot_f = features[indices[r]];
				for(l = r-1; l >= L; l--){
					//std::cout<<l <<" "<<r<<std::endl;
					if (features[indices[l]] < pivot_f) break;
					indices[l+1] = indices[l];
					//print_helper(features, indices);
				}
				indices[l+1] = pivot_i;
			}
			if (index_stack.empty()) break;
			R = index_stack.top(); index_stack.pop();
			L = index_stack.top(); index_stack.pop();
		}
		else{
			// pick pivot as the median of 3, and put sentinels in place
			m = (L+R)/2;
			std::swap(indices[m], indices[L+1]);
			if (features[indices[L]] > features[indices[R]])
				std::swap(indices[L], indices[R]);
			if (features[indices[L+1]] > features[indices[R]])
				std::swap(indices[L+1], indices[R]);
			if (features[indices[L]] > features[indices[L+1]])
				std::swap(indices[L], indices[L+1]);
				
			l = L+1;
			r = R;
			
			pivot_f = features[indices[L+1]];
			pivot_i = indices[L+1];
			// the actual partitioning
			while(true){
				do l++; while (features[indices[l]] < pivot_f);
				do r--; while (features[indices[r]] > pivot_f);
				if (l>r) break;
				std::swap(indices[l], indices[r]);
			}
			indices[L+1] = indices[r];
			indices[r] = pivot_i;
			
			// put the larger range on the stack
			if (R-l+1 >= r-L){
				index_stack.push(l);
				index_stack.push(R);
				R = r-1;
			}
			else{
				index_stack.push(L);
				index_stack.push(r-1);
				L = l;
			}
		}
	}
}





template <typename rng_type, typename num_type = float, typename response_type=float, typename index_type = unsigned int>
class binary_split_one_feature_rss_loss_v2: public rfr::splits::k_ary_split_base<2,rng_type, num_type, response_type, index_type> {
  private:
	
	index_type feature_index;	//!< split needs to know which feature it uses
	
	//!< The split criterion contains its type (first element = 0 for numerical, >=1 for categoricals), and the split value in the second/ the categories that fall into the left child respectively
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
	 * \return num_type 
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
			std::vector<num_type> split_criterion_copy;
			std::vector<num_type> current_features = data.features(fi, indices);

			index_type ft = data.get_type_of_feature(fi);
			// feature_type zero means that it is a continous variable
			if (ft == 0){
				split_criterion_copy.assign(2, 0);

				// find best split for the current feature_index
				loss = best_split_continuous(current_features, responses, split_criterion_copy, sum, sum2);
			}
			// a positive feature type encodes the number of possible values
			if (ft > 0){
				split_criterion_copy.assign(1,ft);
				// find best split for the current feature_index
				loss = best_split_categorical(current_features, ft, responses, split_criterion_copy, sum, sum2, rng);
			}

			// check if this split is the best so far
			if (loss < best_loss){
				best_loss = loss;
				best_features.swap(current_features);
				feature_index = fi;
				split_criterion.swap(split_criterion_copy);
			}
		}

		if (best_loss < std::numeric_limits<num_type>::infinity()){
			
			split_criterion.shrink_to_fit();
			
			// now we have to rearrange the indices based on which leaf they fall into
			
			// first for a continuous variable
			if (split_criterion[0] == 0){
				
				auto i_it1 = indices.begin();
				auto i_it2 = --indices.end();
				auto f_it1 = best_features.begin();
				auto f_it2 = --best_features.end();
				
				while (i_it1 != i_it2){
					// find the left most entry which should go to the right child
					while ((*f_it1 <= split_criterion[1]) && (i_it1 != i_it2))
						{
							i_it1++; f_it1++;
						}
					// find the right most entry that should go into the left child
					while ((*f_it2 > split_criterion[1]) && (i_it1 != i_it2))
						{
							i_it2--; f_it2--;
						}
					// swap the indices (don't worry about the feature vector as no index will be visited twice
					std::iter_swap(i_it1, i_it2);
					
					// advance the left iterator (if necessary)
					if (i_it1 != i_it2)
						{i_it1++; f_it1++;}
					// advance the right iterator (if necessary)
					if (i_it1 != i_it2)
						{ i_it2--; f_it2--;}
					
					if ((i_it1 == indices.end()) || (i_it2 == indices.begin())){
						exit(1);
					}
				}
				
				split_indices_it[1] = i_it1;
			}
			// and then for a categorical feature
			else{
				std::sort(++split_criterion.begin(), split_criterion.end());
				auto i_it1 = indices.begin();
				auto i_it2 = --indices.end();
				auto f_it1 = best_features.begin();
				auto f_it2 = --best_features.end();

				while (i_it1 != i_it2){
					// find the left most entry which should go to the right child
					while ((std::binary_search(++split_criterion.begin(), split_criterion.end(), *f_it1)) && (i_it1 != i_it2))
						{
							i_it1++; f_it1++;
						}
					// find the right most entry that should go into the left child
					while (!(std::binary_search(++split_criterion.begin(), split_criterion.end(), *f_it2)) && (i_it1 != i_it2))
						{
							i_it2--; f_it2--;
						}
					// swap the indices (don't worry about the feature vector as no index will be visited twice
					std::iter_swap(i_it1, i_it2);
					
					// advance the left iterator (if necessary)
					if (i_it1 != i_it2)
						{i_it1++; f_it1++;}
					// advance the right iterator (if necessary)
					if (i_it1 != i_it2)
						{ i_it2--; f_it2--;}
					
					if ((i_it1 == indices.end()) || (i_it2 == indices.begin())){
						exit(1);
					}
				}
				split_indices_it[1] = i_it1;
			}
			// the default values for the two split iterators
			split_indices_it[0] = indices.begin();
			split_indices_it[2] = indices.end();			
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
		auto it = split_criterion.begin();
		
		// handle categorical features
		if (*it > (num_type) 0){
			// check if the value is contained in the split 'set'
			it++;
			//it = std::find(it, split_criterion.end(), feature_vector[feature_index]);
			//return( it != split_criterion.end());
			return(!std::binary_search(it, split_criterion.end(), feature_value));
		}
		// simple case of a numerical feature
		return(feature_value > split_criterion[1]);
	}



	/** \brief member function to find the best possible split for a single (continuous) feature
	 * 
	 * 
	 * \param features a vector with the values for the current feature
	 * \param responses the corresponding response values
	 * \param split_criterion a reference to store the split criterion
	 * \param S_y_right the sum of all the response values
	 * \param S_y2_right the sum of all squared response values
	 * 
	 * \return float the loss of this split
	 */
	virtual num_type best_split_continuous(	const std::vector<num_type> & features,
									const std::vector<response_type> & responses,
									std::vector<num_type> &split_criterion,
									num_type S_y_right, num_type S_y2_right){

		// find the best split by looking at any meaningful value for the feature
		// first some temporary variables
		num_type S_y_left(0), S_y2_left(0);
		num_type N_left(0), N_right(features.size());
		num_type loss, best_loss = std::numeric_limits<num_type>::infinity();;

		std::vector<index_type> tmp_indices(features.size());
		std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

		//print_helper(features, tmp_indices);

		my_qsort<index_type, num_type>(tmp_indices, features);

		//print_helper(features, tmp_indices);

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
				best_loss = loss;
				split_criterion[1] = psv;
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
	 * 
	 * \return float the loss of this split
	 */
	virtual num_type best_split_categorical(const std::vector<num_type> & features,
									index_type num_categories,
									const std::vector<response_type> & responses,
									std::vector<num_type> &split_criterion,
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
			int cat = features[i]-1;	// subtracked a 1 to accomodate for the categorical values starting at 1, but vector indices at zero
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
			best_loss = std::numeric_limits<num_type>::max();
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
		for (auto it1 = category_ranking.begin(); it1 != it_best_split; it1++)
			split_criterion.push_back(*it1+1);	// add a 1 to accomodate for the categorical values starting at 1, but vector indices at zero

		// add unobserved values randomly to the split_set
		if (empty_categories_it != category_ranking.end()){
			std::bernoulli_distribution dist;

			for (auto it1 = empty_categories_it; it1 != category_ranking.end(); it1++){
				if (dist(rng)){
					split_criterion.push_back(*it1+1);	// add a 1 to accomodate for the categorical values starting at 1, but vector indices at zero
				}
			}
		}
		return(best_loss);
	}


	virtual void print_info(){
		if (split_criterion[0] == 0)
			std::cout<<"split: f_"<<feature_index<<" <= "<<split_criterion[1]<<"\n";
		else{
			std::cout<<"split: f_"<<feature_index<<" in {";
			for (size_t i = 1; i < split_criterion.size(); i++)
				std::cout<<split_criterion[i]<<", ";
			std::cout<<"\b\b}\n";
		}
	}
	/** \brief member function to create a string representing the split criterion
	 * 
	 * \return std::string a label that characterizes the split
	 */	
	virtual std::string latex_representation(){
		std::stringstream str;
		if (split_criterion[0] == 0){
			str << "$f_{" << feature_index << "}<=" << split_criterion[1] << "$";
		}
		else{
			str << "$f_{" << feature_index << "} \\in \\{"<< split_criterion[1];
			for (size_t i = 2; i < split_criterion.size(); i++){
				str<<","<<split_criterion[i];
			}
			str << "\\}$";
		}
		return(str.str());
	}
	
	std::vector<num_type> get_split_criterion(){return(split_criterion);}

};


}}//namespace rfr::splits
#endif
