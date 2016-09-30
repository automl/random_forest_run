#ifndef RFR_K_ARY_TREE_HPP
#define RFR_K_ARY_TREE_HPP

#include<vector>
#include<deque>
#include<stack>
#include<utility>       // std::pair
#include<algorithm>     // std::shuffle
#include<numeric>       // std::iota
#include<cmath>         // std::abs
#include<iterator>      // std::advance
#include<fstream>


#include "cereal/cereal.hpp"
#include <cereal/types/bitset.hpp>
#include <cereal/types/vector.hpp>

#include "rfr/data_containers/data_container.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_base.hpp"
#include "rfr/trees/tree_options.hpp"

#include "rfr/forests/regression_forest.hpp"


namespace rfr{ namespace trees{

template <const int k,typename split_type, typename rng_type, typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class k_ary_random_tree : public rfr::trees::tree_base<rng_type, num_type, response_type, index_type> {
	
	friend class rfr::forests::regression_forest<k_ary_random_tree<k,split_type, rng_type,num_type,response_type,index_type>, rng_type,num_type, response_type, index_type>;
	
  private:
	typedef rfr::nodes::k_ary_node<k, split_type, rng_type, num_type, response_type, index_type> node_type;
  
	std::vector<node_type> the_nodes;
	index_type num_leafs;
	index_type actual_depth;
	
  public:
  
	k_ary_random_tree(): the_nodes(0), num_leafs(0), actual_depth(0) {}
	
    /* serialize function for saving forests */
  	template<class Archive>
  	void serialize(Archive & archive){
		archive(the_nodes, num_leafs, actual_depth);
	}
	

	// make overloaded fit function with only 3 arguments from the base class visible here!
	using rfr::trees::tree_base<rng_type, num_type, response_type, index_type>::fit;

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
	virtual void fit(const rfr::data_containers::base<num_type, response_type, index_type> &data,
			 rfr::trees::tree_options<num_type, response_type, index_type> tree_opts,
			 std::vector<index_type> &data_indices,
			 rng_type &rng){
		
		tree_opts.adjust_limits_to_data(data);
		
		// storage for all the temporary nodes
		std::deque<rfr::nodes::temporary_node<num_type, index_type> > tmp_nodes;
		
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
							if (std::abs(data.response(*it)- ref) > tree_opts.epsilon_purity){
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



	/* \brief function to recursively compute the marginalized predictions
	 * 
	 * At any split, this function either goes down one path or averages the
	 * prediction of all children weighted by the fraction of the training data
	 * going into them respectively.
	 * */
	num_type marginalized_prediction(num_type* feature_vector, index_type node_index){
		
		auto n = the_nodes[node_index];	// short hand notation
		
		if (n.is_a_leaf())
			return(n.mean());
		
		// if the feature vector can be split, meaning the corresponding features are not NAN
		// return the marginalized prediction of the corresponding child node
		if (n.can_be_split(feature_vector)){
			return marginalized_prediction( feature_vector, n.falls_into_child(feature_vector));
		}
		
		// otherwise the marginalized prediction consists of the weighted sum of all child nodes
		num_type prediction = 0;
		
		for (auto i = 0u; i<k; i++){
			prediction += n.get_split_fraction(i) * marginalized_prediction(feature_vector, n.get_child_index(i));
		}
		return(prediction);
	}


	/* \brief preditcion for partial input vectors marginalized over unspecified values
	 * 
	 * To compute the fANOVA, the mean prediction over partial assingments is needed.
	 * To accomplish that, feed this function a numerical vector where each element that
	 * is NAN will be marginalized over.
	 */
	num_type marginalized_prediction(num_type *feature_vector){
			return(marginalized_prediction(feature_vector, 0));
	}


	/* \brief finds all the split points for each dimension of the input space
	 * 
	 * This function only makes sense for axis aligned splits!
	 * */
	std::vector<std::vector<num_type> > all_split_values (index_type* types){
		std::vector<std::vector<num_type> > split_values;
		
		for (auto &n: the_nodes){
			if (n.is_a_leaf()) continue;
			
			const auto &s = n.get_split();
			auto fi = s.get_feature_index();

			// as the tree does not know the number of features, the vector
			// might be too small and has to be adjusted
			if (split_values.size() <= fi)
				split_values.resize(fi+1);
			// if a split on a categorical occurs, just add all its possible values
			if((types[fi] > 0) && (split_values[fi].size() == 0)){
				split_values[fi].resize(types[fi]);
				std::iota(split_values[fi].begin(), split_values[fi].end(), 0);
			}
			else{
				split_values[fi].emplace_back(s.get_num_split_value());
			}
		}
		
		for (auto &v: split_values)
			std::sort(v.begin(), v.end());
		return(split_values);
	}


	
	virtual std::tuple<num_type, num_type, index_type> predict_mean_var_N(num_type *feature_vector){
		index_type node_index = find_leaf(feature_vector);
		return(the_nodes[node_index].mean_variance_N());
	}
	
	
	virtual index_type number_of_nodes() {return(the_nodes.size());}
	virtual index_type number_of_leafs() {return(num_leafs);}
	virtual index_type depth() {return(actual_depth);}
	
	/* \brief Function to recursively compute the partition induced by the tree
	 *
	 * Do not call this function from the outside! Needs become private at some point!
	 */
	void partition_recursor (	std::vector<std::vector< std::vector<num_type> > > &the_partition,
							std::vector<std::vector<num_type> > &subspace, num_type node_index){

		// add subspace for a leaf
		if (the_nodes[node_index].is_a_leaf())
			the_partition.push_back(subspace);
		else{
			// compute subspaces of children
			auto subs = the_nodes[node_index].compute_subspaces(subspace);
			// recursively go trough the tree
			for (auto i=0u; i<k; i++){
				partition_recursor(the_partition, subs[i], the_nodes[node_index].get_child_index(i));
			}
		}
	}


	/* \brief computes the partitioning of the input space induced by the tree */
	std::vector<std::vector< std::vector<num_type> > > partition( std::vector<std::vector<num_type> > pcs){
	
		std::vector<std::vector< std::vector<num_type> > > the_partition;
		the_partition.reserve(num_leafs);
		
		partition_recursor(the_partition, pcs, 0);
	
	return(the_partition);
	}

	
	index_type num_samples_in_subtree (index_type node_index){
		index_type N = 0;
		if (the_nodes[node_index].is_a_leaf())
			N = the_nodes[node_index].num_samples();
		else{
			for(auto c: the_nodes[node_index].get_children())
				N += num_samples_in_subtree(c);
		}
		return(N);
	}

	
	bool check_split_fractions(num_type epsilon = 1e-6){
		for ( auto i=0u; i<the_nodes.size(); i++){
			if (the_nodes[i].is_a_leaf()) continue;
			
			index_type N = num_samples_in_subtree(i);
			
			for (auto j = 0u; j<k; j++){
				index_type Nj = num_samples_in_subtree(the_nodes[i].get_child_index(j));
				num_type fj = ((num_type) Nj) / ((num_type) N);
				
				if ((fj - the_nodes[i].get_split_fraction(j))/the_nodes[i].get_split_fraction(j) > epsilon)
					return(false);
			}
		}
		return(true);
	}


	
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
