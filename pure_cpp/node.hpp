#include "split.hpp"

namespace rfr{

/* simple leaf node class that only stores the data that falls into in
 * for huge data and special purposes, one could replace this class with
 * one that only stores the mean or other desired quantities
 */
template<class num_type>
class leaf_node{
  public:
	std::vector<num_type> data;
	
  public:
	leaf_node (int n, num_type v){
		data = std::vector<num_type>(n,v);
	}
	
};


/* class containing all information an internal node needs
 * the split is a boost variant to cover continuous and categorical
 * values with the same interface.
 */
template<class num_type>
class internal_node{
  public:
	split<num_type> split_criterion;
	int left_child, right_child;
  public:
	internal_node(split<num_type> S, int lc, int rc): split_criterion(S), left_child(lc), right_child(rc) {};
};

}// namespace rfr
