#ifndef RFR_BINARY_MONDRIAN_NODES_CPP
#define RFR_BINARY_MONDRIAN_NODES_CPP

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


template <int k, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_t = std::default_random_engine>
class k_ary_mondrian_node_minimal{
  protected:
	// for internal_nodes
	std::array<index_t, k> children;
	index_t depth;

	//average, variance, etc
	rfr::util::weighted_running_statistics<num_t> response_stat;   //TODO: needs to be serialized!
	
  public:

	virtual ~k_ary_mondrian_node_minimal () {
	}

	k_ary_mondrian_node_minimal () {
		depth = 0;
		children[0] = 0;
		children[1] = 0;
	};
	k_ary_mondrian_node_minimal (index_t level) {
		depth = level;
		children[0] = 0;
		children[1] = 0;
	};

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive) {
		archive(children, depth); 
	}

	/** \brief to test whether this node is a leaf */
	bool is_a_leaf() const {return(children[0] == children[1] && children[0] == 0);}
	/** \brief get the index of the node's parent */
	
	/** \brief get indices of all children */
	std::array<index_t, k> get_children() const {return(children);}
	index_t get_child_index (index_t idx) const {return(children[idx]);};

	index_t get_depth() const {return(depth);}

	rfr::util::weighted_running_statistics<num_t> get_response_stat() const {return(response_stat);};

	/** \brief returns the index of the child into which the provided sample falls
	 * 
	 * \param feature_vector a feature vector of the appropriate size (not checked!)
	 *
	 * \return index_t index of the child
	 */
	index_t falls_into_child(const std::vector<num_t> &feature_vector) const {
		if (is_a_leaf()) return(0);
		return(1);
	}

	void set_child (index_t idx, index_t child) { children[idx] = child;}
	void set_depth(index_t new_depth) {depth = new_depth;}
	void set_response_stat(rfr::util::weighted_running_statistics<num_t> r_s) {response_stat = r_s;}

	/** \brief prints out some basic information about the node*/
	virtual void print_info() const {
		if (is_a_leaf()){
		}
		else{
			std::cout<<"status: internal node\n";
			std::cout<<"children: ";
			for (auto i=0; i < k; i++)
				std::cout<<children[i]<<" ";
			std::cout<<std::endl;
		}
		std::cout << "N = "<<response_stat.sum_of_weights()<<std::endl;
		std::cout <<"mean = "<< response_stat.mean()<<std::endl;
		std::cout <<"variance polutation = " << response_stat.variance_population()<<std::endl;
		std::cout <<"variance without noise = " << get_response_stat().sum_of_squares() / get_response_stat().sum_of_weights() <<std::endl;

		
		std::cout <<"depth = " << get_depth()<<std::endl;
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
		}
		return(str.str());
	}
};


template <int k, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_t = std::default_random_engine>
class k_ary_mondrian_node_full: public k_ary_mondrian_node_minimal<k, num_t, response_t, index_t, rng_t>{
  protected:
	num_t sum_E;
	num_t split_cost;
	int parent_index;//index_t may have not negative numbers
	num_t split_time;
	index_t split_dimension;
	num_t split_value;
	num_t variance;
	num_t mean;
	std::vector<std::pair<num_t,num_t>> min_max;//
	std::array<typename std::vector<index_t>::iterator, 3> info_split_its;//
	std::array<index_t, 3> info_split_its_index;//
	int number_of_points;
  public:
	virtual ~k_ary_mondrian_node_full () {};
	
	k_ary_mondrian_node_full (){};

	k_ary_mondrian_node_full (int parent, index_t depth, std::array<typename std::vector<index_t>::iterator, 3> info_split):
		k_ary_mondrian_node_minimal<k, num_t, response_t, index_t, rng_t>::k_ary_mondrian_node_minimal(depth),
		parent_index(parent), info_split_its(info_split)
		{};

	k_ary_mondrian_node_full (int parent, index_t depth, std::array<index_t, 3> info_split_index):
		k_ary_mondrian_node_minimal<k, num_t, response_t, index_t, rng_t>::k_ary_mondrian_node_minimal(depth),
		parent_index(parent), info_split_its_index(info_split_index)
		{};

	k_ary_mondrian_node_full (int parent, index_t depth):
		k_ary_mondrian_node_minimal<k, num_t, response_t, index_t, rng_t>::k_ary_mondrian_node_minimal(depth),
		parent_index(parent)
		{};

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive) {
		archive(sum_E, split_cost, parent_index, split_time, split_dimension, split_value, variance,
			mean,/* min_max,*/ number_of_points);
		//super::serialize(archive);
	}
	
	/** \brief get reference to the response values*/	
	//std::vector<response_t> const &responses () const { return( (std::vector<response_t> const &) response_values);}
	/** \brief get the sum of the mx-min intervals fo the node*/	
	num_t const get_sum_of_Min_Max_intervals () const { return( sum_E);}
	int const get_parent_index () const { return( parent_index);}
	num_t const get_split_time () const { return( split_time);}
	index_t const get_split_dimension () const { return( split_dimension);}
	num_t const get_split_value () const { return( split_value);}
	std::vector<std::pair<num_t,num_t>> const get_min_max () const { return( min_max);}
	std::array<typename std::vector<index_t>::iterator, 3> const get_info_split_its () const { return( info_split_its);}
	std::array<index_t, 3> const get_info_split_its_index () const { return( info_split_its_index);}
	num_t const get_variance() const { return(variance);}
	num_t const get_mean() const { return(mean);}
	int const get_number_of_points() const { return(number_of_points);}
	//std::vector<index_t> const get_points() const { return(points);}
	num_t const get_split_cost() const { return(split_cost);}
	//std::vector<response_t> const  get_responses () const { return response_values;}

	void set_sum_of_Min_Max_intervals (num_t sum){ sum_E = sum;}
	void set_parent_index (index_t parent){ parent_index = parent;}
	void set_split_time (num_t time_s){ split_time = time_s;}
	void set_split_dimension (index_t split_d){ split_dimension = split_d;}
	void set_split_value (num_t split_v){ split_value = split_v;}
	void set_min_max (std::vector<std::pair<num_t,num_t>> m_m){ min_max = m_m;}
	void set_info_split_its (std::array<typename std::vector<index_t>::iterator, 3> info_split){ info_split_its = info_split;}
	void set_info_split_its_index (std::array<index_t, 3> info_split){ info_split_its_index = info_split;}
	void set_variance (num_t var){ variance = var;}
	void set_mean (num_t m){ mean = m;}
	//void set_points (std::vector<index_t> p){ points = p;}
	void set_number_of_points (int p){ number_of_points = p;}
	
	void set_split_cost (num_t sc){ split_cost =sc;}
	

	void add_response (response_t response, num_t weight){ 
		//response_values.emplace_back(response);
		k_ary_mondrian_node_minimal<k, num_t, response_t, index_t, rng_t>::response_stat.push(response, weight);
	}

	/** \brief prints out some basic information about the node*/
	virtual void print_info() const {
		k_ary_mondrian_node_minimal<k, num_t, response_t, index_t, rng_t>::print_info();
		std::cout << "parent index:" << parent_index << std::endl;
		
		std::cout << "Split dimension:" << get_split_dimension() << std::endl;
		std::cout << "Split time:" << split_time << std::endl;
		std::cout << "Split value:" << split_value << std::endl;
		std::cout << "Split cost:" << get_split_cost() << std::endl;
		std::cout << "Min max:[" << min_max[0].first << " " << min_max[0].second <<"]" << std::endl;
		// if (k_ary_mondrian_node_minimal<k, num_t, response_t, index_t, rng_t>::is_a_leaf()){
		// 	rfr::print_vector(response_values);
		// }
		std::cout << "pred_variance: " << variance << std::endl;
		std::cout << "points: " << number_of_points << std::endl;
		//rfr::print_vector(points);
	}

};


}} // namespace rfr::nodes
#endif



