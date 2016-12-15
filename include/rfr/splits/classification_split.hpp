#ifndef CLASSIFICATION_SPLIT_HPP
#define CLASSIFICATION_SPLIT_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <sstream>

#include "rfr/data_containers/data_container_base.hpp"
#include "rfr/splits/split_base.hpp"
#include "rfr/data_containers/data_container_utils.hpp"
namespace rfr{ namespace splits{


template <typename rng_t, typename num_t = float, typename response_t=unsigned int, typename index_t = unsigned int>
/** \brief OUTDATED: Needs to be adapted to the new internal API for splitting!
 */
class binary_split_one_feature_gini: public rfr::splits::k_ary_split_base<2,rng_t, num_t, response_t, index_t> {
  private:

	index_t feature_index;       //!< split needs to know which feature it uses

	//!< The split criterion contains its type (first element = 0 for numerical, =1 for categoricals), and the split value in the second/ the categories that fall into the left child respectively
	std::vector<num_t> split_criterion; //!< one could consider to use a dynamically sized array here to save some memory (vector stores size and capacity + it might allocate more memory than needed!)
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
	 * \param rng a pseudo random number generator instance
	 * 
	 */
	 virtual num_t find_best_split(	const rfr::data_containers::data_container_base<num_t, response_t, index_t> &data,
						const std::vector<index_t> &features_to_try,
						std::vector<index_t> & indices,
						std::array<typename std::vector<index_t>::iterator, 3> &split_indices_it,
						rng_t &rng){

		std::vector<index_t> indices_copy(indices);
		num_t best_gini = std::numeric_limits<num_t>::infinity();

		for (index_t fi : features_to_try){ //! > uses C++11 range based loop


			std::vector<num_t> split_criterion_copy;
			typename std::vector<index_t>::iterator split_indices_it_copy = indices_copy.begin();
			num_t gini;

			// sort the indices by the value in feature vector
			std::sort(      indices_copy.begin(), indices_copy.end(),
						[&](index_t a, index_t b){return data.feature(fi,a) < data.feature(fi, b);}               //! > uses C++11 lambda function, how exciting :)
			);

			index_t ft = data.get_type_of_feature(fi);
			// feature_type zero means that it is a continous variable
			if (ft == 0){
				split_criterion_copy.assign(2, 0);
				// find best split for the current feature_index
				gini = best_split_continuous(data, fi, split_criterion_copy, indices_copy, split_indices_it_copy);
			}
			// a positive feature type encodes the number of possible values
			if (ft > 0){
				split_criterion_copy.assign(1,ft);
				// find best split for the current feature_index
				gini = best_split_categorical(data,fi, ft, split_criterion_copy, indices_copy, split_indices_it_copy, rng);
			}

			// check if this split is the best so far
			if (gini < best_gini){
				best_gini = gini;
				feature_index = fi;
				split_criterion.swap(split_criterion_copy);
				indices.swap(indices_copy);
				split_indices_it.at(1) = split_indices_it_copy;
			}
		}
		if (best_gini < std::numeric_limits<num_t>::infinity()){
			split_indices_it[0] = indices.begin();
			split_indices_it[2] = indices.end();
			std::sort(++split_criterion.begin(), split_criterion.end());
			split_criterion.shrink_to_fit();
		}
		return(best_gini);
	}

	/** \brief this operator tells into which child the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return int whether the feature_vector falls into the left (true) or right (false) child
	 */
	virtual index_t operator() (num_t *feature_vector) {
		auto it = split_criterion.begin();

		// handle categorical features
		if (*it > (num_t) 0){
			// check if the value is contained in the split 'set'
			it++;
			//it = std::find(it, split_criterion.end(), feature_vector[feature_index]);
			//return( it != split_criterion.end());
			return(!std::binary_search(it, split_criterion.end(), feature_vector[feature_index]));
		}

		// simple case of a numerical feature
		return(feature_vector[feature_index] > split_criterion[1]);
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
	 * \return the gini criterion of this split
	 */
	num_t best_split_continuous( const rfr::data_containers::data_container_base<num_t, response_t, index_t> &data,
									const index_t & fi,
									std::vector<num_t> &split_criterion_copy,
									std::vector<index_t> &indices_copy,
									typename std::vector<index_t>::iterator &split_indices_it_copy){
		

		// Splitting criteria = Gini criterion
		// first some temporary variables
		num_t N_L(0), N_R(indices_copy.size());
		num_t S_y_left(0), S_y_right(0);
		num_t gini, best_gini = std::numeric_limits<num_t>::infinity();
		int max_class =  *std::max_element(begin(data.response), end(data.response));
    	int min_class =  *std::min_element(begin(data.response), end(data.response));
		int length = max_class - min_class+1;
		std::vector<index_t> classvector_r (length, 0);
		std::vector<index_t> classvector_l (length, 0);
		std::vector<num_t> p_kr (length);
		std::vector<num_t> p_kl (length);
		num_t gini_l = 0, gini_r = 0;

		// we start out with everything in the right child and calculate the gini for total dataset
		for (auto it = indices_copy.begin(); it != indices_copy.end(); it++){
			S_y_right = data.response(*it);
			classvector_r[S_y_right - min_class] += 1;
		}
		// distribution (proportion) of all classes in the node:
		for (int i = 0; i <length; i++){
			p_kr[i] = classvector_r[i]/N_R;
		}
		// calculating the gini of this node
		for ( int i = 0; i < length; i++){
			gini += p_kr[i]*(1-p_kr[i]);
		}
		// resulting gini index for that node:
		gini = N_R * gini;

		// splits (best slit is the one with the highest purity/gini)
		
		typename std::vector<index_t>::iterator psii = indices_copy.begin(); // potential split index iterator
		// now we can increase the splitting value to move data points from the right to the left child
		// this way we do not consider a split with everything in the right child
		while (psii != indices_copy.end()){
			num_t psv = data.feature(fi, *psii) + 1e-10; // potential split value add small delta for numerical inaccuracy
			// combine data points that are very close
			while ((psii != indices_copy.end()) && (data.feature(fi,*psii) - psv <= 0)){

				S_y_left  += data.response(*psii);
				S_y_right -= data.response(*psii);
				classvector_r[S_y_right - min_class] -= 1;
				classvector_l[S_y_left - min_class] += 1;
				N_R--;
				N_L++;
				// count classes for the right and left node and divide them by the overall classes of node
				for (int i = 0; i <length; i++){
					p_kr[i] = classvector_r[i]/N_R;
					p_kl[i] = classvector_l[i]/N_L;
				}
				// calculate the ginis for each node (summing up proportions of all classes)
				for ( int i = 0; i < length; i++){
					gini_r += p_kr[i]*(1-p_kr[i]);
					gini_l += p_kl[i]*(1-p_kl[i]);
				}
				
				psii++;
			}
			// stop if all data points are now in the left child as this is not a meaningful split
			if (N_R == 0) break;

			// compute the main gini index for both sides (the most important values)
			gini = (N_L * gini_l) + (N_R * gini_r);

			// store the best split
			if (gini < best_gini){
				best_gini = gini;
				split_criterion_copy[1] = psv;
				split_indices_it_copy = psii;
			}
		}
		return(best_gini);		
		
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
	num_t best_split_categorical(const rfr::data_containers::data_container_base<num_t, response_t, index_t> &data,
									const index_t &fi,
									const index_t &num_categories,
									std::vector<num_t> &split_criterion_copy,
									std::vector<index_t> &indices_copy,
									typename std::vector<index_t>::iterator &split_indices_it_copy,
									rng_t &rng){

		// auxiliary variables
		std::vector<index_t> category_ranking(num_categories);
		std::iota(category_ranking.begin(), category_ranking.end(),0);
		std::vector<index_t> N_points_in_category(num_categories,0);
		std::vector<num_t> S_y(num_categories, 0);

		for (auto i: indices_copy){
			// find the category for each entry as a proper int
			//! >assumes that the features for categoricals have been properly rounded so casting them to ints results in the right value!
			int cat = data.feature(fi,i)-1; // subtracked a 1 to accomodate for the categorical values starting at 1, but vector indices at zero
			// collect all the data to compute the loss
			S_y[cat]  += data.response(i);
			N_points_in_category[cat] += 1;
		}
		// take care b/c certain categories might not be encountered (maybe there was a split on the same variable further up the tree...)
		// sort the categories by whether there were samples or not
		// std::partition rearranges the data using a boolean predicate into all that evaluate to true in front of all evaluating to false.
		// it even returns an iterator pointing to the first element where the predicate is false, how convenient :)
		auto empty_categories_it = std::partition(category_ranking.begin(), category_ranking.end(),
						[&](index_t a){return(N_points_in_category[a] > 0);});

		// sort the categories by their individual mean. only consider the ones with actual specimen here
		std::sort(      category_ranking.begin(), empty_categories_it,
					[&](index_t a, index_t b){return ( (S_y[a]/N_points_in_category[a]) < (S_y[b]/N_points_in_category[b]) );});              // C++11 lambda function, how exciting :)

		//more auxiliary variables
		num_t S_y_left = 0, S_y2_left = 0, N_left = 0;
		num_t S_y_right = 0, S_y2_right = 0, N_right= 0;
		num_t current_gini = 0, best_gini = 0;
		int max_class =  *std::max_element(begin(data.response), end(data.response));
    	int min_class =  *std::min_element(begin(data.response), end(data.response));
		int length = max_class - min_class+1;
		std::vector<index_t> classvector_r (length, 0);
		std::vector<index_t> classvector_l (length, 0);
		std::vector<num_t> p_kr (length);
		std::vector<num_t> p_kl (length);
		float gini_l = 0;
		float gini_r = 0;
		float gini = 0;	

		// put one category in the left node
		auto it_best_split = category_ranking.begin();
		S_y_left  = S_y[*it_best_split];
		N_left    = N_points_in_category[*it_best_split];
		// distribution (proportion) of all classes in the node:
		for (int i = 0; i <length; i++){
			p_kl[i] = classvector_l[i]/N_left;
		}
		// calculate the gini for left node (summing up proportions of all classes)
		for ( int i = 0; i < length; i++){
			gini_l += p_kl[i]*(1-p_kl[i]);
		}
		// resulting gini index for left node:
		gini = N_left * gini_l;
	    
		it_best_split++;


		// the rest goes into the right node
		for (auto it1 = it_best_split; it1!=empty_categories_it; it1++){
			S_y_right  += S_y[*it1];
			N_right    += N_points_in_category[*it1];
		}
		// distribution (proportion) of all classes in the node:
		for (int i = 0; i <length; i++){
			p_kr[i] = classvector_r[i]/N_right;
		}
		for ( int i = 0; i < length; i++){
				gini_r += p_kr[i]*(1-p_kr[i]);
		}
		// gini index for that node:
		gini = N_right * gini_r;           

		// it can happen that the node is not pure wrt the response, but the
		// feature at hand takes only one value in this node. By setting the
		// best_gini to the largest possible value, this split will not be chosen.
		// It also yields 
		if ( (N_right == 0) || (N_left == 0) )
			best_gini = std::numeric_limits<num_t>::max();
		else
			best_gini = (N_left * gini_l) + (N_right * gini_r);

		current_gini = best_gini;

		// now move one category at a time to the left child and recompute the gini index
		for (auto it1 = it_best_split; it1 != empty_categories_it; it1++){
			S_y_left  += S_y[*it1];
			S_y_right -= S_y[*it1];
			N_left  += N_points_in_category[*it1];
			N_right -= N_points_in_category[*it1];
			classvector_r[S_y_right - min_class] -= 1;
			classvector_l[S_y_left - min_class] += 1;
			for (int i = 0; i <length; i++){
				p_kr[i] = classvector_r[i]/N_right;
				p_kl[i] = classvector_l[i]/N_left;
			}
			for ( int i = 0; i < length; i++){
				gini_r += p_kr[i]*(1-p_kr[i]);
				gini_l += p_kl[i]*(1-p_kl[i]);
			}



			// catch divide by zero as they are invalid splits anyway
			// becomes important if only one or two categories have specimen here!
			if ( (N_right != 0) && (N_left != 0) ){
				current_gini = (N_left * gini_l) + (N_right * gini_r);

				// keep the best split
				if (current_gini < best_gini){
					best_gini = current_gini;
					it_best_split = it1;
					it_best_split++;
				}
			}
		}

		// store the split set for the left leaf
		for (auto it1 = category_ranking.begin(); it1 != it_best_split; it1++)
			split_criterion_copy.push_back(*it1+1); // add a 1 to accomodate for the categorical values starting at 1, but vector indices at zero

		// add unobserved values randomly to the split_set
		if (empty_categories_it != category_ranking.end()){
			std::bernoulli_distribution dist;

			for (auto it1 = empty_categories_it; it1 != category_ranking.end(); it1++){
				if (dist(rng)){
					split_criterion_copy.push_back(*it1+1); // add a 1 to accomodate for the categorical values starting at 1, but vector indices at zero
				}
			}
		}

		// rearrange indices according to their category and in which leaf they go
		// std::partition is again exactly what we need here :)
		split_indices_it_copy = std::partition(
					indices_copy.begin(), indices_copy.end(),
					[&](size_t i){return(std::find(++split_criterion_copy.begin(), split_criterion_copy.end(), data.feature(fi, i)) != split_criterion_copy.end());});
		
		return(best_gini);
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

	std::vector<num_t> get_split_criterion(){return(split_criterion);}




};

}}//namespace rfr::splits
#endif
