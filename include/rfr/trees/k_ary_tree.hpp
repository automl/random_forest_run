#ifndef RFR_K_ARY_TREE_HPP
#define RFR_K_ARY_TREE_HPP

#include<vector>
#include<deque>
#include<stack>
#include<utility>       // std::pair
#include<algorithm>     // std::shuffle
#include<numeric>       // std::iota
#include<cmath>         // abs
#include<iterator>      // std::advance
#include<fstream>


#include "rfr/data_containers/data_container_base.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_base.hpp"
#include "rfr/trees/tree_options.hpp"


namespace rfr{ namespace trees{

template <const int k,typename split_type, typename rng_type, typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class k_ary_random_tree : public rfr::trees::tree_base<rng_type, num_type, response_type, index_type> {
  private:
	std::vector< rfr::nodes::k_ary_node<k, split_type, rng_type, num_type, response_type, index_type> > the_nodes;
	index_type num_leafs;
	index_type actual_depth;
	
  public:

	/** \brief fits a randomized decision tree to a subset of the data
	 * 
	 * At each node, if it is 'splitworthy', a random subset of all features is considered for the
	 * split. Depending on the split_type provided, greedy or randomized choices can be
	 * made. Just make sure the max_features in tree_opts to a number smaller than the number of features!
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options opject that controls certain aspects of "growing" the tree
	 * \param data_indices vector containing the indices of all allowed datapoints to be used (to implement subsampling, no checks are done here!)
	 */
	virtual void fit(const rfr::data_containers::data_container_base<num_type, response_type, index_type> &data,
			 rfr::tree_options<num_type, response_type, index_type> tree_opts,
			 std::vector<index_type> &data_indices,
			 rng_type &rng){
		
		tree_opts.adjust_limits_to_data(data);
		
		// storage for all the temporary nodes
		std::deque<temporary_node<num_type, index_type> > tmp_nodes;
		
		std::vector<index_type> feature_indices(data.num_features());
		std::iota(feature_indices.begin(), feature_indices.end(), 0);
		
		// add the root to the temporary nodes to get things started
		tmp_nodes.emplace_back(0, 0, 0, data_indices.begin(), data_indices.end());

		// initialize the private variables in case the tree is refitted!
		the_nodes.clear();
		num_leafs = 0;
		actual_depth = 0;
		
		// as long as there are potentially splittable nodes
		while (!tmp_nodes.empty()){

			// resize 'the_nodes' if necessary
			if (tmp_nodes.back().node_index >= the_nodes.size())
				the_nodes.resize( tmp_nodes.back().node_index+1);

			bool is_not_pure = false;
			// check if the node is pure!
			{
				num_type ref = data.response(tmp_nodes.front().data_indices[0]);
				
				for(auto it = ++tmp_nodes.front().data_indices.begin(); it!= tmp_nodes.front().data_indices.end(); it++){
							if (abs(data.response(*it)- ref) > tree_opts.epsilon_purity){
									is_not_pure = true;
									break;
							}
				}
			}
			
			// check if it should be split
			if ((tmp_nodes.front().node_level < tree_opts.max_depth) &&                     // don't grow the tree to deep!
				(tmp_nodes.front().data_indices.size() >= tree_opts.min_samples_to_split)&& // are enough sample left in the node?
				(is_not_pure) &&                                                            // are not all the values the same?
				(the_nodes.size() <= tree_opts.max_num_nodes-k)                             // don't have more nodes than the user specified number
				){
				// generate a subset of the features to try
				std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
				std::vector<index_type> feature_subset(feature_indices.begin(), std::next(feature_indices.begin(), tree_opts.max_features));

				//split the node
				
				num_type best_loss = the_nodes[tmp_nodes.front().node_index].make_internal_node(
										tmp_nodes.front(), data, feature_subset,
										the_nodes.size(), tmp_nodes,rng);
				

				// if no split was produces, the node turns itself into a leaf, so nothing else has to be done
				if (best_loss <  std::numeric_limits<num_type>::infinity()){

					// Now, we have to check whether the split was legal
					bool illegal_split = false;
					auto tmp_it = tmp_nodes.end();
					std::advance(tmp_it, -k);
				
					for (; tmp_it != tmp_nodes.end(); tmp_it++){
						if ( (*tmp_it).data_indices.size() < tree_opts.min_samples_in_leaf){
							illegal_split = true;
							break;
						}
					}
					// in case it wasn't ...
					if (illegal_split){
						// we have to delete the k new temporary nodes
						for (auto i = 0; i<k; i++)
							tmp_nodes.pop_back();
						// and make this node a leaf
						the_nodes[tmp_nodes.front().node_index].make_leaf_node(tmp_nodes.front(),data);
						actual_depth = std::max(actual_depth, tmp_nodes.front().node_level);
						num_leafs++;
					}
				}
			}
			// if it is not 'splitworthy', just turn it into a leaf
			else{
				the_nodes[tmp_nodes.front().node_index].make_leaf_node(tmp_nodes.front(), data);
				actual_depth = std::max(actual_depth, tmp_nodes.front().node_level);
				num_leafs++;
			}
			tmp_nodes.pop_front();
		}
		the_nodes.shrink_to_fit();
	}

	virtual index_type find_leaf(num_type *feature_vector){
		index_type node_index = 0;
		while (! the_nodes[node_index].is_a_leaf()){
			node_index = the_nodes[node_index].falls_into_child(feature_vector);
		}
		return(node_index);
	}
	
	virtual std::vector<response_type> const &leaf_entries (num_type *feature_vector){
		index_type i = find_leaf(feature_vector);
		return(the_nodes[i].responses());
	}
	
	virtual response_type predict (num_type *feature_vector){
		index_type node_index = find_leaf(feature_vector);
		return(the_nodes[node_index].mean());
	}
	
	virtual std::tuple<num_type, num_type, index_type> predict_mean_std_N(num_type *feature_vector){
		index_type node_index = find_leaf(feature_vector);
		return(the_nodes[node_index].mean_variance_N());
	}
	
	
	virtual index_type number_of_nodes() {return(the_nodes.size());}
	virtual index_type number_of_leafs() {return(num_leafs);}
	virtual index_type depth() {return(actual_depth);}
	
	
	void print_info(){
		
		std::cout<<"number of nodes ="<<number_of_nodes()<<"\n";
		std::cout<<"number of leafes="<<number_of_leafs()<<"\n";
		std::cout<<"      depth     ="<<depth()<<"\n";
		
		for (auto i = 0u; i< the_nodes.size(); i++){
			std::cout<<"=========================\nnode "<<i<<"\n";
			the_nodes[i].print_info();
		}
	}
	    
	/** \brief a visualization by generating a LaTeX document that can be compiled
	* 
	* 
	* \param filename Name of the file that will be used. Note that any existing file will be silently overwritten!
	*/
	virtual void save_latex_representation(const char* filename){
		std::fstream str;
		    
		str.open(filename, std::fstream::out);
		std::stack <typename std::pair<std::array<index_type, k>, index_type> > stack;
		    
		// LaTeX headers
		str<<"\\documentclass{standalone}\n\\usepackage{forest}\n\n\\begin{document}\n\\begin{forest}\n";
		str<<"for tree={grow'=east, child anchor = west, draw, calign=center}\n";
		    
		// the root needs special treatment
		if (!the_nodes[0].is_a_leaf()){
			stack.emplace(typename std::pair<std::array<index_type, k>, index_type> (the_nodes[0].get_children(), 0));
			str<<"["<<the_nodes[0].latex_representation(0)<<"\n";
		}
		// 'recursively' add the nodes in a depth first fashion
		while (!stack.empty()){
			if (stack.top().second == k){
				stack.pop();
				for (size_t i=0; i<stack.size(); i++) str << "\t";
				str << "]\n";
			}
			else{
				auto current_index = stack.top().first[stack.top().second++];
				    
				for (size_t i=0; i<stack.size(); i++) str << "\t";
				str << "[" << the_nodes[current_index].latex_representation(current_index);
				    
				if (the_nodes[current_index].is_a_leaf())
					str << "]\n";
				else{
					str << "\n";
					stack.emplace(typename std::pair<std::array<index_type, k>, index_type> (the_nodes[current_index].get_children(), 0));
				}
			}
		}
		str<<"\\end{forest}\n\\end{document}\n";
		str.close();
	}
};

}}//namespace rfr::trees
#endif
