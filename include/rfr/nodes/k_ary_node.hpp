#ifndef RFR_BINARY_NODES_HPP
#define RFR_BINARY_NODES_HPP

#include <vector>
#include <deque>
#include <array>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <random>

#include "rfr/data_containers/data_container.hpp"
#include "rfr/data_containers/data_container_utils.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/util.hpp"
#include "rfr/splits/split_base.hpp"

#include "cereal/cereal.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>


#include <iostream>


namespace rfr{ namespace nodes{



template <int k, typename split_type, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_t = std::default_random_engine>
class k_ary_node_minimal{
  protected:
	index_t parent_index;

	// for leaf nodes
	rfr::util::weighted_running_statistics<num_t> response_stat;

	// for internal_nodes
	std::array<index_t, k> children;
	std::array<num_t, k> split_fractions;
	split_type split;
	
  public:

	virtual ~k_ary_node_minimal () {};

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive) {
		archive( parent_index, children, split_fractions, split, response_stat); 
	}

  
	/** \brief If the temporary node should be split further, this member turns this node into an internal node.
	*
	* 
	* \param tmp_node a temporary_node struct containing all the important information. It is not changed in this function.
	* \param data a refernce to the data object that is used
	* \param features_to_try vector of allowed features to be used for this split
	* \param num_nodes number of already created nodes
	* \param tmp_nodes a deque instance containing all temporary nodes that still have to be checked
    * \param rng a RNG instance
	*
	* \return num_t the loss of the split
	*/ 
	num_t make_internal_node(const rfr::nodes::temporary_node<num_t, response_t, index_t> &tmp_node,
                             const rfr::data_containers::base<num_t, response_t, index_t> &data,
							 std::vector<index_t> &features_to_try,
							 index_t num_nodes,
							 std::deque<rfr::nodes::temporary_node<num_t, response_t, index_t> > &tmp_nodes,
							 rng_t &rng){
		parent_index = tmp_node.parent_index;
		std::array<typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t> >::iterator, k+1> split_indices_it;
		num_t best_loss = split.find_best_split(data, features_to_try, tmp_node.begin, tmp_node.end, split_indices_it,rng);
	
		//check if a split was found
		// note: if the number of features to try is too small, there is a chance that the data cannot be split any further
		if (best_loss <  std::numeric_limits<num_t>::infinity()){
			// create a tmp node for each child
            num_t total_weight = 0;
			for (index_t i = 0; i < k; i++){
				tmp_nodes.emplace_back(num_nodes+i, tmp_node.node_index, tmp_node.node_level+1, split_indices_it[i], split_indices_it[i+1]);
                split_fractions[i]=tmp_nodes.back().total_weight();
                total_weight += split_fractions[i];
				children[i] = num_nodes + i;
			}
			for (auto &sf: split_fractions)
                sf /= total_weight;

		}
		else
			make_leaf_node(tmp_node, data);
		return(best_loss);
	}
	
	/** \brief  turns this node into a leaf node based on a temporary node.
	*
	* \param tmp_node the internal representation for a temporary node.
	* \param data a data container instance
	*/
	void make_leaf_node(const rfr::nodes::temporary_node<num_t, response_t, index_t> &tmp_node,
						const rfr::data_containers::base<num_t, response_t, index_t> &data){
		parent_index = tmp_node.parent_index;
		children.fill(0);
		split_fractions.fill(NAN);
		
		for (auto it = tmp_node.begin; it != tmp_node.end; ++it){
			push_response_value((*it).response, (*it).weight);
		}
	}	

	/* \brief function to check if a feature vector can be splitted */
	bool can_be_split(const std::vector<num_t> &feature_vector) const {
		if (is_a_leaf()) return(false);
		return(split.can_be_split(feature_vector));
	}

	/** \brief returns the index of the child into which the provided sample falls
	 * 
	 * \param feature_vector a feature vector of the appropriate size (not checked!)
	 *
	 * \return index_t index of the child
	 */
	index_t falls_into_child(const std::vector<num_t> &feature_vector) const {
		if (is_a_leaf()) return(0);
		return(children[split(feature_vector)]);
	}


	/** \brief adds an observation to the leaf node
	 *
	 * This function can be used for pseudo updates of a tree by
	 * simply adding observations into the corresponding leaf
	 */
	virtual void push_response_value ( response_t r, num_t w){
		response_stat.push(r,w);
	}

	/** \brief removes an observation from the leaf node
	 *
	 * This function can be used for pseudo updates of a tree by
	 * simply removing observations from the corresponding leaf
	 */
	virtual void pop_response_value (response_t r, num_t w){
		response_stat.pop(r,w);
	}

	/** \brief helper function for the fANOVA
	 *
	 * 	See description of rfr::splits::binary_split_one_feature_rss_loss::compute_subspace.
	 */
	std::array<std::vector< std::vector<num_t> >, 2> compute_subspaces( const std::vector< std::vector<num_t> > &subspace) const {
		return(split.compute_subspaces(subspace));
	}

	/* \brief returns a running_statistics instance for computations of mean, variance, etc...
	 *
	 * See description of rfr::util::weighted_running_statistics for more information
	 */
	rfr::util::weighted_running_statistics<num_t> const & leaf_statistic()  const { return (response_stat);}

	/** \brief to test whether this node is a leaf */
	bool is_a_leaf() const {return(children[0] == 0);}
	/** \brief get the index of the node's parent */
	index_t parent() const {return(parent_index);}
	
	/** \brief get indices of all children*/
	std::array<index_t, k> get_children() const {return(children);}
	index_t get_child_index (index_t idx) const {return(children[idx]);};

	std::array<num_t, k> get_split_fractions() const {return(split_fractions);}
	num_t get_split_fraction (index_t idx) const {return(split_fractions[idx]);};

	const split_type & get_split() const {return(split);}

	/** \brief prints out some basic information about the node*/
	virtual void print_info() const {
		if (is_a_leaf()){
			std::cout << "N = "<<response_stat.sum_of_weights()<<std::endl;
			std::cout <<"mean = "<< response_stat.mean()<<std::endl;
			std::cout <<"variance = " << response_stat.variance_unbiased_frequency()<<std::endl;
		}
		else{
			std::cout<<"status: internal node\n";
			std::cout<<"children: ";
			for (auto i=0; i < k; i++)
				std::cout<<children[i]<<" ";
			std::cout<<std::endl;
		}
	}

	/** \brief generates a label for the node to be used in the LaTeX visualization*/
	virtual std::string latex_representation( int my_index) const {
		std::stringstream str;
			
		if (is_a_leaf()){
			str << "{i = " << my_index << ": ";

			str << "N = "<<response_stat.sum_of_weights();
			str <<", mean = "<< response_stat.mean();
			str <<", variance = " << response_stat.variance_unbiased_frequency()<<"}";
			
		}
		else{
			str << "{ i = " << my_index << "\\nodepart{two} {";
			str << split.latex_representation() << "}},rectangle split,rectangle split parts=2,draw";
			
		}
		return(str.str());
	}
};



/** \brief The node class for regular k-ary trees.
 * 
 * In a regular k-ary tree, every node has either zero (a leaf) or exactly k-children (an internal node).
 * In this case, one can try to gain some speed by replacing variable length std::vectors by std::arrays.
 * 
 */
template <int k, typename split_type, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_t = std::default_random_engine>
class k_ary_node_full: public k_ary_node_minimal<k, split_type, num_t, response_t, index_t, rng_t>{
  protected:
	// additional info for leaf nodes
	std::vector<response_t> response_values;
	std::vector<num_t> response_weights;
	typedef k_ary_node_minimal<k, split_type, num_t, response_t, index_t, rng_t> super;
	
  public:

	virtual ~k_ary_node_full () {};

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive) {
		archive(response_values, response_weights);
		super::serialize(archive);
	}


	/** \brief adds an observation to the leaf node
	 *
	 * This function can be used for pseudo updates of a tree by
	 * simply adding observations into the corresponding leaf
	 */
	virtual void push_response_value ( response_t r, num_t w){
		super::push_response_value(r,w);
		response_values.push_back(r);
		response_weights.push_back(w);
	}

	/** \brief removes the last added observation from the leaf node
	 *
	 * This function can be used for pseudo updates of a tree by
	 * simply adding observations into the corresponding leaf
	 *
	 * \param r ignored
	 * \param w ignored
	 * 
	 */
	virtual void pop_response_value (response_t r, num_t w){

		super::pop_response_value(response_values.back(), response_weights.back());
		response_values.pop_back();
		response_weights.pop_back();
	}

	/** \brief get reference to the response values*/	
	std::vector<response_t> const &responses () const { return( (std::vector<response_t> const &) response_values);}

	/** \brief get reference to the response values*/	
	std::vector<num_t> const &weights () const { return( (std::vector<num_t> const &) response_weights);}

	/** \brief prints out some basic information about the node*/
	virtual void print_info() const {
		super::print_info();
		if (super::is_a_leaf()){
			rfr::print_vector(response_values);
		}
	}

};


}} // namespace rfr::nodes
#endif
