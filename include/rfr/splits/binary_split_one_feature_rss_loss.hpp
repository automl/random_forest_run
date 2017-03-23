#ifndef RFR_BINARY_SPLIT_RSS_HPP
#define RFR_BINARY_SPLIT_RSS_HPP

#include <vector>
#include <bitset>
#include <array>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>


#include <cereal/cereal.hpp>
#include <cereal/types/bitset.hpp>
#include <rfr/util.hpp>
#include <rfr/data_containers/data_container.hpp>
#include <rfr/splits/split_base.hpp>
#include <rfr/data_containers/data_container_utils.hpp>
namespace rfr{ namespace splits{



template <	typename num_t = float,
			typename response_t=float,
			typename index_t = unsigned int,
			typename rng_t = std::default_random_engine,
			unsigned int max_num_categories = 128>
class binary_split_one_feature_rss_loss: public rfr::splits::k_ary_split_base<2, num_t, response_t, index_t, rng_t> {
  private:
	
	index_t feature_index;	//!< split needs to know which feature it uses
	num_t num_split_value;	//!< value of a numerical split
	std::bitset<max_num_categories> cat_split_set;	//!< set of values for a categorical split

  public:
  	

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive){
		archive( feature_index, num_split_value, cat_split_set); 
	}
  	
	/* get the number of split categories.
	 * 
	 * SF: that is definitely wrong .. 
	 * 
	*/
	index_t get_num_categories() const { return max_num_categories; }//return cat_split_set.size(); }
  	
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
	 * \param infos_begin iterator to the first (relevant) element in a vector containing the minimal information in tuples
	 * \param infos_end iterator beyond the last (relevant) element in a vector containing the minimal information in tuples
	 * \param info_split_its iterators into this vector saying where to split the data for the two children
	 * \param rng a random number generator instance
	 * \return num_t loss of the best found split
	 */
	 virtual num_t find_best_split(	const rfr::data_containers::base<num_t, response_t, index_t> &data,
									const std::vector<index_t> &features_to_try,
									typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator infos_begin,
									typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator infos_end,
									std::array<typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator, 3> &info_split_its,
									rng_t &rng){

				
		// precompute mean and variance of all responses
		rfr::util::weighted_running_statistics<num_t> total_stat;
		for (auto it = infos_begin; it != infos_end; ++it){
			total_stat.push(it->response, it->weight);
		}
		
		num_t best_loss = std::numeric_limits<num_t>::infinity();

		for (index_t fi : features_to_try){ //! > uses C++11 range based loop

			num_t loss;
			num_t num_split_copy = NAN;
			std::bitset<max_num_categories> cat_split_copy;

			for (auto it = infos_begin; it != infos_end; ++it){
				it->feature = data.feature( fi, it->index);
			}

			index_t ft = data.get_type_of_feature(fi);
			// feature_type zero means that it is a continous variable
			if (ft == 0){
				// find best split for the current feature_index
				loss = best_split_continuous(infos_begin, infos_end, num_split_copy, total_stat ,rng);
			}
			// a positive feature type encodes the number of possible values
			if (ft > 0){
				// find best split for the current feature_index
				loss = best_split_categorical(infos_begin, infos_end, ft, cat_split_copy, total_stat, rng);
			}

			// check if this split is the best so far
			if (loss < best_loss){
				best_loss = loss;
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
		// now we have to rearrange the indices based on which leaf they fall into
		if (best_loss < std::numeric_limits<num_t>::infinity()){
			// the default values for the two split iterators
			info_split_its[0] = infos_begin;
			info_split_its[2] = infos_end;

			info_split_its[1] = std::partition (infos_begin, infos_end,
				[this,&data] (rfr::splits::data_info_t<num_t, response_t, index_t> &arg){
					return !(this->operator()(data.feature(this->feature_index, arg.index)));
				});
		}
		return(best_loss);
	}


	/** \brief this operator tells into which child the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return int whether the feature_vector falls into the left (false) or right (true) child
	 */
	virtual index_t operator() (const std::vector<num_t> &feature_vector) const { return(operator()(feature_vector[feature_index]));}
	
	/** \brief overloaded operator for just the respective feature value instead of the complete vector
	 * 
	 */
	virtual index_t operator() (const num_t &feature_value) const {
		// categorical feature
		if (std::isnan(num_split_value))
			return(! bool(cat_split_set[ int(feature_value)]));
		// standard numerical feature
		return(feature_value > num_split_value);
	}



	/** \brief member function to find the best possible split for a single (continuous) feature
	 * 
	 * 
	 * \param infos_begin iterator to the first (relevant) element in a vector containing the minimal information in tuples
	 * \param infos_end iterator beyond the last (relevant) element in a vector containing the minimal information in tuples
	 * \param split_value a reference to store the split (numerical) criterion
	 * \param right_stat a weighted_runnin_statistics object containing the statistics of all responses
	 * \param rng a pseudo random number generator instance
	 * 
	 * \return float the loss of this split
	 */
	virtual num_t best_split_continuous(
					typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator infos_begin,
					typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator infos_end,
					num_t &split_value,
					rfr::util::weighted_running_statistics<num_t> right_stat,
					rng_t &rng){

		// first, sort the info vector by the feature
		std::sort(infos_begin, infos_end,
			[] (const rfr::splits::data_info_t<num_t, response_t, index_t> &a, const rfr::splits::data_info_t<num_t, response_t, index_t> &b) {return (a.feature < b.feature) ;});

		// first some temporary variables
		rfr::util::weighted_running_statistics<num_t> left_stat;
		num_t best_loss = std::numeric_limits<num_t>::infinity();


		// now we can increase the splitting value to move data points from the right to the left child
		// this way we do not consider a split with everything in the left or right child
		auto it = infos_begin;
		while (it != infos_end-1){
			num_t psv = (*it).feature + 1e-6; // potential split value add small delta for numerical inaccuracy
			// combine data points that are very close
			do {
				left_stat.push((*it).response, (*it).weight);
                try{// substraction of statistics can lead to negative squared distances from the mean, especially if the right
                    // child contains samples with the same response value. This happens all the time if a categorical feature is
                    // not specified as such.
                    right_stat.pop((*it).response, (*it).weight);
                }
                catch (const std::runtime_error& e){
                    // Just recompute the statistic from scratch, which should always work!
                    right_stat = rfr::util::weighted_running_statistics<num_t> ();
                    for (auto tmp_it = it+1; tmp_it != infos_end; tmp_it++)
                        right_stat.push((*tmp_it).response, (*tmp_it).weight);
                }
				++it;
			} while ((it != infos_end-1) && ((*it).feature <= psv));
			
			// stop if all data points are now in the left child as this is not a meaningful split
			if (right_stat.sum_of_weights() == 0) {break;}

			// compute the loss
			num_t loss = 	left_stat.squared_deviations_from_the_mean() +
							right_stat.squared_deviations_from_the_mean();

			// store the best split
			if (loss < best_loss){
				std::uniform_real_distribution<num_t> dist(0.0,1.0);
				best_loss = loss;
				split_value = psv + dist(rng)*((*it).feature - psv);
			}
		}
		return(best_loss);
	}

	/** \brief member function to find the best possible split for a single (categorical) feature
	 * 
	 * \param infos_begin iterator to the first (relevant) element in a vector containing the minimal information in tuples
	 * \param infos_end iterator beyond the last (relevant) element in a vector containing the minimal information in tuples	 * 
	 * \param num_categories the feature type (number of different values)
	 * \param split_set a reference to store the split criterion
	 * \param right_stat the statistics of the reponses of all remaining data points
	 * \param rng an pseudo random number generator instance
	 * 
	 * \return float the loss of this split
	 */
	virtual num_t best_split_categorical(
									typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator infos_begin,
									typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator infos_end,
									index_t num_categories,
									std::bitset<max_num_categories> &split_set,
									rfr::util::weighted_running_statistics<num_t> right_stat,
									rng_t &rng){
		// auxiliary variables
		num_t best_loss = std::numeric_limits<num_t>::infinity();
		rfr::util::weighted_running_statistics<num_t> left_stat;

		std::vector<rfr::util::weighted_running_statistics<num_t> > cat_stats (num_categories);
		for (auto it = infos_begin; it != infos_end; ++it){
			// find the category for each entry as a proper int
			//! >assumes that the features for categoricals have been properly rounded so casting them to ints results in the right value!
			int cat = it->feature;
			// collect all the data to compute the loss
			cat_stats[cat].push(it->response, it->weight);
		}

		// take care b/c certain categories might not be encountered (maybe there was a split on the same variable further up the tree...)
		// sort the categories by whether there were samples or not
		// std::partition rearranges the data using a boolean predicate into all that evaluate to true in front of all evaluating to false.
		// it even returns an iterator pointing to the first element where the predicate is false, how convenient :)
		std::vector<index_t> cat_ranking(num_categories);
		std::iota(cat_ranking.begin(), cat_ranking.end(), 0);
		
		auto empty_cat_ranking_it = std::partition(cat_ranking.begin(), cat_ranking.end(),
						[cat_stats](index_t &index){return(cat_stats[index].sum_of_weights() > 0);});
		

		// it can happen that the node is not pure wrt the response, but the
		// feature at hand takes only one value in this node. By returning
		// best_loss = inf, this split will not be chosen.
		unsigned int active_categories = std::distance(cat_ranking.begin(), empty_cat_ranking_it);		
		if (active_categories == 1) return(best_loss);

		// sort the categories by their individual mean. only consider the ones with actual specimen here
		std::sort(	cat_ranking.begin(), empty_cat_ranking_it,
					[cat_stats](const index_t a, const index_t &b){return ( cat_stats[a].mean() < cat_stats[b].mean() );});		// C++11 lambda function, how exciting :)


		auto it1 = cat_ranking.begin();
		auto it_best_split = cat_ranking.begin()+1;

		// now move one category at a time to the left child and recompute the loss
		do{
			left_stat += cat_stats[*it1];
			right_stat-= cat_stats[*it1];

			num_t loss = left_stat.squared_deviations_from_the_mean() +
						right_stat.squared_deviations_from_the_mean();

				// keep the best split
				if (loss < best_loss){
					best_loss = loss;
					it_best_split = it1 + 1;
				}
			++it1;
			--active_categories;
		} while (active_categories > 1);

		// store the split set for the left leaf
		split_set.reset();
		for (auto it1 = cat_ranking.begin(); it1 != it_best_split; it1++)
			split_set.set(*it1);

		// add unobserved values randomly to the split_set
		if (empty_cat_ranking_it != cat_ranking.end()){
			std::bernoulli_distribution dist;

			for (auto it1 = empty_cat_ranking_it; it1 != cat_ranking.end(); it1++){
				if (dist(rng))
					split_set.set(*it1);
			}
		}
		return(best_loss);
	}


	virtual void print_info() const {
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
	virtual std::string latex_representation() const {
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
	
	index_t get_feature_index() const {return(feature_index);}
	num_t get_num_split_value() const {return(num_split_value);}
	std::bitset<max_num_categories> get_cat_split_set() const {return(cat_split_set);}
	
	/* \brief takes a subspace and returns the 2 corresponding subspaces after the split is applied
	 *
	 * This is an essential function for the fANOVA. Every split
	 * constraints one of the parameters in each of the children.
	 */
	std::array<std::vector< std::vector<num_t> >, 2> compute_subspaces(const std::vector< std::vector<num_t> > &subspace) const {
		
	
		std::array<std::vector<std::vector<num_t> >, 2> subspaces = {subspace, subspace};

		// if feature is numerical
		if (! std::isnan(num_split_value)){
			// for the left child, the split value is the new upper bound
			subspaces[0][feature_index][1] = num_split_value;
			// for the right child the split value is the new lower bound
			subspaces[1][feature_index][0] = num_split_value;
		}
		else{
			// every element in the split set should go to the left -> remove from right
			auto it = std::partition (subspaces[0][feature_index].begin(), subspaces[0][feature_index].end(),
										[this] (int i) {return((bool) this->cat_split_set[i]);});
		
			// replace the values in the 'right subspace'
			subspaces[1][feature_index].assign(it, subspaces[0][feature_index].end());
			
			// delete all values in the 'left subspace'
			subspaces[0][feature_index].resize(std::distance(subspaces[0][feature_index].begin(),it));
		}
		return(subspaces);
	}

	bool can_be_split(const std::vector<num_t> &feature_vector) const {
		return(!std::isnan(feature_vector[feature_index]));
	}
};


}}//namespace rfr::splits
#endif
