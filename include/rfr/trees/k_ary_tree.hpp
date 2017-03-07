#ifndef RFR_K_ARY_TREE_HPP
#define RFR_K_ARY_TREE_HPP

#include <vector>
#include <deque>
#include <stack>
#include <utility>       // std::pair
#include <algorithm>     // std::shuffle
#include <numeric>       // std::iota
#include <cmath>         // std::abs
#include <iterator>      // std::advance
#include <fstream>
#include <random>


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

template <const int k,typename node_type, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_type = std::default_random_engine>
class k_ary_random_tree : public rfr::trees::tree_base<num_t, response_t, index_t, rng_type> {
	
  protected:
    typedef rfr::splits::data_info_t<num_t, response_t, index_t> info_t;
  
	std::vector<node_type> the_nodes;
	index_t num_leafs;
	index_t actual_depth;
	
  public:
  
	k_ary_random_tree(): the_nodes(0), num_leafs(0), actual_depth(0) {}

	virtual ~k_ary_random_tree() {}
	
    /* serialize function for saving forests */
  	template<class Archive>
  	void serialize(Archive & archive){
		archive(the_nodes, num_leafs, actual_depth);
	}
	
	/** \brief fits a randomized decision tree to a subset of the data
	 * 
	 * At each node, if it is 'splitworthy', a random subset of all features is considered for the
	 * split. Depending on the split_type provided, greedy or randomized choices can be
	 * made. Just make sure the max_features in tree_opts to a number smaller than the number of features!
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options object that controls certain aspects of "growing" the tree
	 * \param sample_weights vector containing the weights of all allowed datapoints (set to individual entries to zero for subsampling), no checks are done here!
	 * \param rng the random number generator to be used
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data,
			 rfr::trees::tree_options<num_t, response_t, index_t> tree_opts,
			 const std::vector<num_t> &sample_weights,
			 rng_type &rng){
		
		tree_opts.adjust_limits_to_data(data);
		
		// storage for all the temporary nodes
		std::deque<rfr::nodes::temporary_node<num_t, response_t, index_t> > tmp_nodes;
		
		std::vector<index_t> feature_indices(data.num_features());
		std::iota(feature_indices.begin(), feature_indices.end(), 0);
		
        
        std::vector<info_t > data_infos;
        data_infos.reserve(data.num_data_points());
        
        for (auto i=0u; i<data.num_data_points(); ++i){
            if (sample_weights[i] > 0){
                data_infos.emplace_back(i, data.response(i), NAN, data.weight(i) * sample_weights[i]);
            }
        }
        
        
		// add the root to the temporary nodes to get things started
		tmp_nodes.emplace_back(0, 0, 0, data_infos.begin(), data_infos.end());

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
				num_t ref = (*(tmp_nodes.front().begin)).response;
				
				for(auto it = std::next(tmp_nodes.front().begin); it!= tmp_nodes.front().end; it++){
							if (std::abs((*it).response- ref) > tree_opts.epsilon_purity){
									is_not_pure = true;
									break;
							}
				}
			}
			// check if it should be split
			if ((tmp_nodes.front().node_level < tree_opts.max_depth) &&                     // don't grow the tree to deep!
				(std::distance(tmp_nodes.front().begin, tmp_nodes.front().end) >= tree_opts.min_samples_to_split)&& // are enough sample left in the node?
				(is_not_pure) &&                                                            // are not all the values the same?
				(the_nodes.size() <= tree_opts.max_num_nodes-k) &&                          // don't have more nodes than the user specified number
				(tmp_nodes.size() < tree_opts.max_num_leaves - num_leafs)
				){
				// generate a subset of the features to try
				std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
				std::vector<index_t> feature_subset(feature_indices.begin(), std::next(feature_indices.begin(), tree_opts.max_features));

				//split the node
				num_t best_loss = the_nodes[tmp_nodes.front().node_index].make_internal_node(
										tmp_nodes.front(), data, feature_subset,
										the_nodes.size(), tmp_nodes,rng);
				

				// if no split was produces, the node turns itself into a leaf, so nothing else has to be done
				if (best_loss <  std::numeric_limits<num_t>::infinity()){

					// Now, we have to check whether the split was legal
					bool illegal_split = false;
					auto tmp_it = tmp_nodes.end();
					std::advance(tmp_it, -k);
				
					for (; tmp_it != tmp_nodes.end(); tmp_it++){
						if ( std::distance((*tmp_it).begin, (*tmp_it).end)< tree_opts.min_samples_in_leaf){
							illegal_split = true;
							break;
						}
						num_t tmp_weight = 0;
						for (auto mew = (*tmp_it).begin; mew != (*tmp_it).end; mew++){
							tmp_weight += (*mew).weight;
						}

						if (tmp_weight < tree_opts.min_weight_in_leaf){
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

	virtual index_t find_leaf_index(const std::vector<num_t> &feature_vector) const {
		index_t node_index = 0;
		while (! the_nodes[node_index].is_a_leaf()){
			node_index = the_nodes[node_index].falls_into_child(feature_vector);
		}
		return(node_index);
	}

	const node_type&  get_leaf(const std::vector<num_t> &feature_vector) const {
		index_t node_index = find_leaf_index(feature_vector);
		return(the_nodes[node_index]);
	}

	
	virtual std::vector<response_t> const &leaf_entries (const std::vector<num_t> &feature_vector) const {
		index_t i = find_leaf_index(feature_vector);
		return(the_nodes[i].responses());
	}
	

	rfr::util::weighted_running_statistics<num_t> const & leaf_statistic(const std::vector<num_t> &feature_vector) const {
        index_t i = find_leaf_index(feature_vector);
        return(the_nodes[i].leaf_statistic());
    }

    
    virtual response_t predict (const std::vector<num_t> &feature_vector) const {
		leaf_statistic(feature_vector).sum_of_weights();
        return(leaf_statistic(feature_vector).mean());
    }
    
    

	/* \brief function to recursively compute the marginalized predictions
	 * 
	 * To compute the fANOVA, the mean prediction over partial assingments is needed.
	 * To accomplish that, feed this function a numerical vector where each element that
	 * is NAN will be marginalized over.
     * 
	 * At any split, this function either goes down one path or averages the
	 * prediction of all children weighted by the fraction of the training data
	 * going into them respectively.
     * 
     * \param feature_vector the features vector with NAN for dimensions over which is marginalized
     * \param node_index index of root of the computation, default 0 means 'start at the root'
     * 
     * \returns the mean prediction marginalized over the desired inputs according to the training data
	 * */
	num_t marginalized_mean_prediction(const std::vector<num_t> &feature_vector, index_t node_index=0) const{
		
		auto n = the_nodes[node_index];	// short hand notation
		
		if (n.is_a_leaf())
			return(n.leaf_statistic().mean());
		
		// if the feature vector can be split, meaning the corresponding features are not NAN
		// return the marginalized prediction of the corresponding child node
		if (n.can_be_split(feature_vector)){
			return marginalized_mean_prediction( feature_vector, n.falls_into_child(feature_vector));
		}
		
		// otherwise the marginalized prediction consists of the weighted sum of all child nodes
		num_t prediction = 0;
		
		for (auto i = 0u; i<k; i++){
			prediction += n.get_split_fraction(i) * marginalized_mean_prediction(feature_vector, n.get_child_index(i));
		}
		return(prediction);
	}


	
	virtual index_t number_of_nodes() const {return(the_nodes.size());}
	virtual index_t number_of_leafs() const {return(num_leafs);}
	virtual index_t depth()           const {return(actual_depth);}
	
	/* \brief Function to recursively compute the partition induced by the tree
	 *
	 * Do not call this function from the outside! Needs become private at some point!
	 */
	void partition_recursor (	std::vector<std::vector< std::vector<num_t> > > &the_partition,
							std::vector<std::vector<num_t> > &subspace, num_t node_index) const {

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
	std::vector<std::vector< std::vector<num_t> > > partition( std::vector<std::vector<num_t> > pcs) const {
	
		std::vector<std::vector< std::vector<num_t> > > the_partition;
		the_partition.reserve(num_leafs);
		
		partition_recursor(the_partition, pcs, 0);
	
	return(the_partition);
	}

	
	num_t total_weight_in_subtree (index_t node_index) const {
		num_t w = 0;
		if (the_nodes[node_index].is_a_leaf())
			w = the_nodes[node_index].leaf_statistic().sum_of_weights();
		else{
			for(auto c: the_nodes[node_index].get_children())
				w += total_weight_in_subtree(c);
		}
		return(w);
	}

	
	bool check_split_fractions(num_t epsilon = 1e-6) const {

		bool rt = true;
		
		for ( auto i=0u; i<the_nodes.size(); i++){
			if (the_nodes[i].is_a_leaf()) continue;
			
			num_t W = total_weight_in_subtree(i);
			
			for (auto j = 0u; j<k; j++){
				num_t Wj = total_weight_in_subtree(the_nodes[i].get_child_index(j));
				num_t fj =  Wj / W;
				
				rt = rt && ((fj - the_nodes[i].get_split_fraction(j))/the_nodes[i].get_split_fraction(j) < epsilon) ;

			}
		}
		return(rt);
	}

	/* \brief updates the forest by adding all provided datapoints without a complete retraining
	 * 
	 * 
	 * As retraining can be quite expensive, this function can be used to quickly update the forest
	 * by finding the leafs the datapoints belong into and just inserting them. This is, of course,
	 * not the right way to do it for many data points, but it should be a good approximation for a few.
	 * 
	 * \param features a valid feature vector
	 * \param response the corresponding response value
	 * \param weight the associated weight
	 */
	void pseudo_update (std::vector<num_t> features, response_t response, num_t weight){
		index_t index = find_leaf_index(features);
		the_nodes[index].push_response_value(response, weight);
	}


	/* \brief undoing a pseudo update by removing a point
	 * 
	 * This function removes one point from the corresponding leaves into
	 * which the given feature vector falls
	 * 
	 * \param features a valid feature vector
	 * \param response the corresponding response value
	 * \param weight the associated weight
	 */
	void pseudo_downdate(std::vector<num_t> features, response_t response, num_t weight){
		index_t index = find_leaf_index(features);
		the_nodes[index].pop_response_value(response, weight);
	}


	
	void print_info() const {
		
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
	virtual void save_latex_representation(const char* filename) const {
		std::fstream str;
		    
		str.open(filename, std::fstream::out);
		std::stack <typename std::pair<std::array<index_t, k>, index_t> > stack;
		    
		// LaTeX headers
		str<<"\\documentclass{standalone}\n\\usepackage{forest}\n\n\\begin{document}\n\\begin{forest}\n";
		str<<"for tree={grow'=east, child anchor = west, draw, calign=center}\n";
		    
		// the root needs special treatment
		if (!the_nodes[0].is_a_leaf()){
			stack.emplace(typename std::pair<std::array<index_t, k>, index_t> (the_nodes[0].get_children(), 0));
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
					stack.emplace(typename std::pair<std::array<index_t, k>, index_t> (the_nodes[current_index].get_children(), 0));
				}
			}
		}
		str<<"\\end{forest}\n\\end{document}\n";
		str.close();
	}
};

}}//namespace rfr::trees
#endif
