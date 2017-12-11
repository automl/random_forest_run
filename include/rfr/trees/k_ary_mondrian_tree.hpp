#ifndef RFR_K_ARY_MONDRIAN_TREE_HPP
#define RFR_K_ARY_MONDRIAN_TREE_HPP

#include<vector>
#include<deque>
#include<stack>
#include<utility>       // std::pair
#include<algorithm>     // std::shuffle, std::max
#include<numeric>       // std::iota
#include<cmath>         // std::abs
#include<iterator>      // std::advance
#include<fstream>
#include<random>
#include<limits>		//min and max


#include "cereal/cereal.hpp"
#include <cereal/types/bitset.hpp>
#include <cereal/types/vector.hpp>

#include "rfr/data_containers/data_container.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_mondrian_node.hpp"
#include "rfr/trees/tree_base.hpp"
#include "rfr/trees/tree_options.hpp"

#include "rfr/forests/mondrian_forest.hpp"


namespace rfr{ namespace trees{

template <const int k,typename node_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_t = std::default_random_engine>
class k_ary_mondrian_tree : public rfr::trees::tree_base<num_t, response_t, index_t, rng_t> {
  protected:
  
	std::vector<node_t> the_nodes;
	index_t num_leafs;
	index_t max_depth;
	num_t life_time;//
	num_t variance_coef;
	num_t sigmoid_coef;
	num_t sfactor;// //
	num_t prior_variance;
	num_t noise_variance;
	bool smooth_hierarchically;//
	index_t min_samples_node;//
	index_t min_samples_to_split;//
	
  public:
  
	k_ary_mondrian_tree(): the_nodes(0), num_leafs(0), max_depth(0) {}

	virtual ~k_ary_mondrian_tree() {}
	
    /* serialize function for saving forests */
  	template<class Archive>
  	void serialize(Archive & archive){
		archive(the_nodes, num_leafs, max_depth, life_time, variance_coef, sigmoid_coef, sfactor, 
			prior_variance, noise_variance, smooth_hierarchically, min_samples_node, min_samples_to_split);
	}
	
	void set_tree_options(rfr::trees::tree_options<num_t, response_t, index_t> tree_opts){
		life_time = tree_opts.life_time;
		smooth_hierarchically = tree_opts.hierarchical_smoothing;
		min_samples_node = 1;//tree_opts.min_samples_node;
		min_samples_to_split = tree_opts.min_samples_to_split;
		sfactor = 2;
	}

	void partial_fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, 
		rfr::trees::tree_options<num_t, response_t, index_t> tree_opts, index_t new_point, rng_t &rng){
		set_tree_options(tree_opts);
		internal_partial_fit(data, tree_opts, new_point, rng);
		update_gaussian_parameters(data);
		if(smooth_hierarchically){
			update_likelyhood();
		}
		

	}

	/** \brief internal_partial_fit adds a point to the current mondrian tree
	 * 
	 * Finds the place of the new_point in the tree and adds the point in that part of the tree.
	 * 
	 * made. Just make sure the max_features in tree_opts to a number smaller than the number of features!
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options object that controls certain aspects of "growing" the tree
	 * \param new_point index of the point to add
	 * \param rng the random number generator to be used
	 */
	void internal_partial_fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, 
		rfr::trees::tree_options<num_t, response_t, index_t> tree_opts, index_t new_point, rng_t &rng){
		int position = 0;
		std::vector<std::pair<num_t,num_t>> min_max;
		bool end = false;
		std::vector<index_t> points;
		std::vector<num_t> feature_vector = data.retrieve_data_point(new_point);
		response_t response = data.response(new_point);
		std::array<index_t, 3> info_split_its;
		info_split_its[0] = 0;
		info_split_its[2] = 1;
		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> tmp_node(-1, 1), father_parent;
		if(the_nodes.size() == 0){
			the_nodes.resize( the_nodes.size()+1);
				
			std::vector<index_t> selected_elements;
			selected_elements.emplace_back(new_point);
			std::vector<response_t> responses(response);
			std::vector<rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t>> tmp_nodes;
			tmp_nodes.emplace_back(-1,1,info_split_its);
			tmp_node = Sample_Mondrian_Block(data, tmp_nodes, selected_elements, responses, 0, rng);
			the_nodes[0] = tmp_node;
			end = true;
		}
		else{
			tmp_node = the_nodes[position];
		}
		
		num_t min, max, time_parent, split_value, E;

		while(!end){
			num_t sum_E = calculate_nu(tmp_node, feature_vector);
			min_max = tmp_node.get_min_max();
			
			std::exponential_distribution<num_t> distribution(sum_E);// 1/sum_E
			E = distribution(rng);
			
			time_parent = get_parent_split_time(tmp_node);
			
			
			if(time_parent + E < tmp_node.get_split_time()){//compare with the lifetime of the node
				std::uniform_real_distribution<num_t> dist (0,sum_E);
				double dice = dist(rng);
				num_t counter = 0;
				index_t split_dimension = 0u;
				while(counter<dice && split_dimension<data.num_features()){//check maximum j
					counter= counter + std::max(feature_vector[split_dimension] - min_max[split_dimension].second, (num_t)0) +
						std::max(min_max[split_dimension].first - feature_vector[split_dimension], (num_t)0);
					split_dimension++;
				}
				split_dimension--;
				
				if(feature_vector[split_dimension]>min_max[split_dimension].second){//
					min = min_max[split_dimension].second;//first
					max = feature_vector[split_dimension];
				} else{
					if(feature_vector[split_dimension]<min_max[split_dimension].first){
						min = feature_vector[split_dimension];
						max = min_max[split_dimension].first;//scond
					}
					else {
						throw std::runtime_error("impossible partial fit");
					}
				}
				std::uniform_real_distribution<num_t> dist2 (min, max);
				split_value = dist2(rng);
				rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> father_node(-1, tmp_node.get_depth());
				rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> new_leaf_node(-1, tmp_node.get_depth()+1, info_split_its);
				tmp_node.set_depth(tmp_node.get_depth()+1);
				the_nodes[position] = tmp_node;
				father_node.set_split_dimension(split_dimension);
				father_node.set_split_value(split_value);
				father_node.set_split_time(time_parent + E);
				father_node.set_split_cost(E);//update the nwe node cost

				for(index_t i = 0; i<feature_vector.size(); i++){
					if(feature_vector[i] < min_max[i].first)
						min_max[i].first = feature_vector[i];
					if(feature_vector[i] > min_max[i].second)
						min_max[i].second = feature_vector[i];
				}
				father_node.set_min_max(min_max);
				father_node.set_number_of_points(tmp_node.get_number_of_points()+1);
				father_node.set_response_stat(tmp_node.get_response_stat());
				father_node.add_response(response, 1);
				addNewNode(father_node, position, true);

				std::vector<rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t>> tmp_nodes;
				tmp_nodes.emplace_back(new_leaf_node);
				std::vector<index_t> selected_elements;
				selected_elements.emplace_back(new_point);
				std::vector<response_t> responses;
				responses.emplace_back(response);
				new_leaf_node = Sample_Mondrian_Block(data, tmp_nodes, selected_elements, responses, position+1, rng);
				addNewNode(new_leaf_node, position+1,false);

				father_node = the_nodes[position];
				if(feature_vector[split_dimension]<=split_value){
					father_node.set_child(0, position+1);
					father_node.set_child(1, position+2);
				}
				else{
					father_node.set_child(0, position+2);
					father_node.set_child(1, position+1);
				}
				int base_parent_index = the_nodes[position+2].get_parent_index();
				if(base_parent_index>=0){
					father_parent = the_nodes[base_parent_index];
					if( (int) father_parent.get_children()[0] == position+2){
						father_parent.set_child(0, position);
						father_node.set_parent_index(base_parent_index);
					}
					else{
						if( (int) father_parent.get_children()[1] == position+2){
							father_parent.set_child(1, position);
							father_node.set_parent_index(base_parent_index);
						}
						else{
							std::cout << "MONDRIAN_TREE::partial_fit::IMPOSIBLE" << std::endl;
							throw std::runtime_error("partial fit, father doesn't have the correct children");
						}
					}
					the_nodes[base_parent_index] = father_parent;
				}
				the_nodes[position] = father_node;
				the_nodes[position+1].set_parent_index(position);
				the_nodes[position+2].set_parent_index(position);
				update_subtree(position+2);
				end = true;

			}
			else{
				tmp_node.set_number_of_points(tmp_node.get_number_of_points()+1);
				tmp_node.add_response(response, 1);
				for(index_t i = 0; i< feature_vector.size(); i++){
					min_max[i].first = std::min(min_max[i].first, feature_vector[i]);
					min_max[i].second = std::max(min_max[i].second, feature_vector[i]);
				}
				tmp_node.set_min_max(min_max);
				the_nodes[position] = tmp_node;
				if(!tmp_node.is_a_leaf()){
					if(feature_vector[tmp_node.get_split_dimension()] <= tmp_node.get_split_value()){
						position = tmp_node.get_children()[0];
						tmp_node = the_nodes[tmp_node.get_children()[0]];
					}
					else{
						position = tmp_node.get_children()[1];
						tmp_node = the_nodes[tmp_node.get_children()[1]];
					}
				}
				else{
					end = true;
				}
			}
		}
	}

	void update_subtree(index_t start){
		std::vector<index_t> indexes;
		index_t actual;
		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> node = the_nodes[start];
		if(!node.is_a_leaf()){
			indexes.emplace_back(node.get_children()[0]);
			indexes.emplace_back(node.get_children()[1]);
		}
		while(!indexes.empty()){
			actual = indexes.back();
			indexes.pop_back();
			node = the_nodes[actual];
			node.set_depth(node.get_depth()+1);
			if(!node.is_a_leaf()){
				indexes.emplace_back(node.get_children()[0]);
				indexes.emplace_back(node.get_children()[1]);
			}
			the_nodes[actual] = node;

		}
	}


	void addNewNode(rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> new_node,
	 int initial_position, bool adding_parent){
		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> aux, tmp_node, aux_parent, child;
		tmp_node = new_node;
		int aux_parent_index = tmp_node.get_parent_index();
		
		int position = initial_position;
		the_nodes.resize( the_nodes.size()+1);
		std::vector<std::pair<index_t,index_t>> updated(the_nodes.size());
		for(index_t i = 0; i<the_nodes.size();i++){
			updated[i].first = 0;
			updated[i].second = 0;
		}
		while(position< (int) the_nodes.size()){
			if(position< (int) the_nodes.size()-1){
				aux = the_nodes[position];
				if(aux.get_children()[0] != aux.get_children()[1]){
					for(int i = 0; i<2; i++){
						child = the_nodes[aux.get_children()[i]];
						child.set_parent_index(child.get_parent_index()+1);
						the_nodes[aux.get_children()[i]] = child;
					}
				}
				
				aux_parent_index = aux.get_parent_index();
				if(aux_parent_index >= 0){
					if(aux_parent_index == position){
						aux_parent = tmp_node;
					}
					else{
						aux_parent = the_nodes[aux_parent_index];
					}
					
					if( (int) aux_parent.get_children()[0] == position && !updated[aux_parent_index].first){
						aux_parent.set_child(0, position+1);
						updated[aux_parent_index].first = 1;
					}
					else{
						if( (int) aux_parent.get_children()[1] == position && !updated[aux_parent_index].second){
							aux_parent.set_child(1, position+1);
							updated[aux_parent_index].second = 1;
						}
						else{
							//check
						}
					}
					if(aux_parent_index == position){
						tmp_node = aux_parent;
					}
					else {
						the_nodes[aux_parent_index] = aux_parent;
					}
					
				}
				
				if(aux.get_split_cost() == 0){
					//increase the split cost ?
				}
				
			}
			the_nodes[position] = tmp_node;
			tmp_node = aux;
			position++;
		}
		
	}
	//move_update

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
			 rng_t &rng){
		set_tree_options(tree_opts);
		internal_fit(data, tree_opts, sample_weights, rng);
		update_gaussian_parameters(data);
		if(smooth_hierarchically){
			update_likelyhood();
		}
	}

	rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> Sample_Mondrian_Block(
		const rfr::data_containers::base<num_t, response_t, index_t> &data,
		std::vector<rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t>> &tmp_nodes,
		std::vector<index_t> &selected_elements, std::vector<response_t> responses, index_t position, rng_t &rng){
		
		num_t E;
		index_t actual_depth = 1;//modify

        //creates an array of lenght num_features from wiht values from 0 to num_features
		std::vector<index_t> feature_indices(data.num_features());
		std::iota(feature_indices.begin(), feature_indices.end(), 0);
        
		num_t sum_E, time_node, time_parent, split_value;
		index_t split_dimension;
		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> tmp_node;

		std::array<index_t, 3> info_split_its;

		bool create_leaf = true;

		tmp_node = tmp_nodes.back();
		tmp_nodes.pop_back();
		
		if(tmp_node.get_parent_index() != -1){
			time_parent = the_nodes[tmp_node.get_parent_index()].get_split_time();
			if(the_nodes[tmp_node.get_parent_index()].get_child_index(0)==0)
				the_nodes[tmp_node.get_parent_index()].set_child(0, position);
			else
				the_nodes[tmp_node.get_parent_index()].set_child(1, position);
		}
		else{//root of the tree
			time_parent = 0;
		}

		info_split_its = tmp_node.get_info_split_its_index();
		if(info_split_its[0] + min_samples_to_split > info_split_its[2]){
			create_leaf = false;
		}
		else{
			create_leaf = true;
		}
		std::vector<std::pair<num_t,num_t>> min_max = min_max_vector(data, info_split_its, selected_elements, sum_E);

		std::exponential_distribution<num_t> distribution(sum_E);// 1/sum_E
		E = distribution(rng);
		
		if(time_parent + E < life_time && create_leaf){
			time_node = time_parent + E;
			//get split dimension
			std::uniform_real_distribution<num_t> dist (0,sum_E);
			double dice = dist(rng);
			num_t counter = 0;
			split_dimension = 0u;
			while(counter<dice && split_dimension<data.num_features()){//check maximum j
				counter= counter + min_max[split_dimension].second- min_max[split_dimension].first;
				split_dimension++;
			}
			split_dimension--;
	
			num_t min = min_max[split_dimension].first, max = min_max[split_dimension].second;
			std::uniform_real_distribution<num_t> dist2 (min, max);
			split_value = dist2(rng);

			info_split_its[1] =	myPartition(info_split_its[0], info_split_its[2], selected_elements, data, split_dimension, split_value);

			tmp_node.set_info_split_its_index(info_split_its);
			actual_depth = tmp_node.get_depth() + 1;
			//pseudo_leaf, all elements in the same side
			if((info_split_its[0] + min_samples_node) <= info_split_its[1] && 
			(info_split_its[1] + min_samples_node) <= info_split_its[2] ){//que pasa con 2 hojas como minimo?
				//so the first one to pop will be the smaller one
				//continue with right
				std::array<index_t, 3> info_split_its_child;
				info_split_its_child[0] = info_split_its[1];
				info_split_its_child[2] = info_split_its[2];
				tmp_nodes.emplace_back(position, actual_depth, info_split_its_child);
				//continue with left
				info_split_its_child = std::array<index_t, 3>();
				info_split_its_child[0] = info_split_its[0];
				info_split_its_child[2] = info_split_its[1];
				tmp_nodes.emplace_back(position, actual_depth, info_split_its_child);
			}
			else{	
				split_dimension = -1;
				split_value = -1;
				num_leafs++;
				throw std::runtime_error("probably bad set up in the min_samples_node");
				//std::getchar();
				tmp_node.set_child (0,0);
			}
			
		} else{
			if(time_parent + E < life_time){
				time_node = time_parent + E;
			}
			else{
				time_node = life_time;
			}
				
			split_dimension = -1;
			split_value = -1;
			tmp_node.set_child (0,0);
			num_leafs++;
			if(tmp_node.get_depth()>max_depth){
				max_depth = tmp_node.get_depth();
			}
			
			//node is a leaf 
		}
		tmp_node.set_split_time(time_node);
		tmp_node.set_split_cost(E);
		tmp_node.set_split_dimension(split_dimension);
		tmp_node.set_split_value(split_value);
		tmp_node.set_min_max(min_max);
		for(index_t i = info_split_its[0]; i<info_split_its[2] ; i++){
		 	tmp_node.add_response(data.response(selected_elements[i]), data.weight(selected_elements[i]));
		}
		tmp_node.set_number_of_points(info_split_its[2]-info_split_its[0]);
		return tmp_node;
	}

	virtual void internal_fit(const rfr::data_containers::base<num_t, response_t, index_t> &data,
			 rfr::trees::tree_options<num_t, response_t, index_t> tree_opts,
			 const std::vector<num_t> &sample_weights,
			 rng_t &rng){

		tree_opts.adjust_limits_to_data(data);
		the_nodes.clear();
		num_leafs = 0;
		index_t actual_depth = 1;
		
        //creates an array of lenght num_features from wiht values from 0 to num_features
		std::vector<index_t> feature_indices(data.num_features());
		std::iota(feature_indices.begin(), feature_indices.end(), 0);
        
		//vector with the indexes of the boostrap items
		std::vector<index_t> selected_elements;
        std::vector<response_t> responses;

        for (auto i=0u; i<data.num_data_points(); ++i){
            if (sample_weights[i] > 0){
                
				//fill a vector with the indexes of the elements in the boostrap
				selected_elements.emplace_back(i);
				responses.emplace_back(data.response(i));
            }
        }

		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> tmp_node;
		std::vector<rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t>> tmp_nodes;

		std::array<index_t, 3> info_split_its;
		info_split_its[0] = 0;
		info_split_its[2] = selected_elements.size();

		tmp_nodes.emplace_back(-1, actual_depth, info_split_its);
		
		index_t position = 0;
		while (!tmp_nodes.empty()){
			if (position >= the_nodes.size())
				the_nodes.resize( position+1);
			rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> new_node = Sample_Mondrian_Block(data, tmp_nodes, selected_elements, responses, position, rng);
			the_nodes[position] = new_node;
			position++;
		}
	}

	void update_gaussian_parameters(const rfr::data_containers::base<num_t, response_t, index_t> &data){
		num_t n_points = the_nodes[0].get_number_of_points();//points().size();
		num_t n_features = data.num_features();
		num_t coef = std::min(2*n_points, 500.0);
		prior_variance = the_nodes[0].get_response_stat().variance_population();
		variance_coef = 2 * prior_variance * coef /(coef +2);
		//variance_coef = 54;
		sigmoid_coef = n_features / (sfactor * std::log2(n_points));
		noise_variance = prior_variance / coef;
	}

	void update_likelyhood(){
		std::vector<std::pair<response_t,response_t>> message_to_parent(the_nodes.size());
		std::vector<std::pair<response_t,response_t>> message_from_parent(the_nodes.size());
		std::vector<std::pair<response_t,response_t>> child_likelihood(the_nodes.size());

		///noise precision
		num_t variance, mean;
		for(int i = the_nodes.size()-1; i >= 0; i--){
			if(the_nodes[i].is_a_leaf()){
				variance = get_sigmoid_variance(i) + noise_variance / the_nodes[i].get_number_of_points();//points().size();
				mean = the_nodes[i].get_response_stat().mean();
				message_to_parent[i].first = mean;
				message_to_parent[i].second = 1/variance;
				child_likelihood[i].first = mean;
				child_likelihood[i].second = (1/ noise_variance) * the_nodes[i].get_number_of_points();//multiply by points
			}
			else{
				child_likelihood[i] = multiply_gausian(message_to_parent[the_nodes[i].get_children()[0]],
					message_to_parent[the_nodes[i].get_children()[1]]);
				mean = child_likelihood[i].first;
				variance = get_sigmoid_variance(i) + 1/child_likelihood[i].second;
				message_to_parent[i].first = child_likelihood[i].first;
				message_to_parent[i].second = 1/variance;
			}
		}
		message_from_parent[0].first = the_nodes[0].get_response_stat().mean();
		message_from_parent[0].second = get_sigmoid_variance(0);
		for(index_t i = 0 ; i < the_nodes.size(); i++){
			std::pair<response_t,response_t> pred_param = multiply_gausian(message_from_parent[i],
					child_likelihood[i]);
			the_nodes[i].set_mean(pred_param.first);
			the_nodes[i].set_variance(1/pred_param.second + std::pow(pred_param.first,2) + noise_variance );
			if(!the_nodes[i].is_a_leaf()){
				//crossed
				message_from_parent[the_nodes[i].get_children()[0]] = multiply_gausian(message_from_parent[i],
					message_to_parent[the_nodes[i].get_children()[1]]);
				message_from_parent[the_nodes[i].get_children()[1]] = multiply_gausian(message_from_parent[i],
					message_to_parent[the_nodes[i].get_children()[0]]);

				//uncrossed
				// message_from_parent[the_nodes[i].get_children()[0]] = multiply_gausian(message_from_parent[i],
				// 	message_to_parent[the_nodes[i].get_children()[0]]);
				// message_from_parent[the_nodes[i].get_children()[1]] = multiply_gausian(message_from_parent[i],
				// 	message_to_parent[the_nodes[i].get_children()[1]]);
			}
		}
	}

	num_t get_parent_split_time(rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> node){
		if(node.get_parent_index() == -1){
			return 0;
		}
		else{
			return the_nodes[node.get_parent_index()].get_split_time();
		}
	}
	num_t get_sigmoid_variance(index_t node_index){
		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> node = the_nodes[node_index];
		num_t res = variance_coef * (sigmoid(sigmoid_coef * node.get_split_time()) 
			- sigmoid (sigmoid_coef * get_parent_split_time(node)));
		return res;
	}

	num_t sigmoid(num_t x){
		num_t res = 1/(1+std::exp(-x));
		return res;
	}

	std::pair<response_t,response_t> multiply_gausian(std::pair<response_t,response_t> g1, std::pair<response_t,response_t> g2){
		std::pair<response_t,response_t> r;
		r.second = g1.second + g2.second;
		r.first = (g1.first * g1.second + g2.first * g2.second)/r.second;
		return r;
	}

index_t myPartition( index_t it1, index_t it2, std::vector<index_t> &selected_elements,
		const rfr::data_containers::base<num_t, response_t, index_t> &data, index_t split_dimension, num_t split_value ){
		index_t last = it2;
		last--;
		index_t i = it1;
		index_t aux = -1;
		while(i<last){
			if(data.feature(split_dimension, selected_elements[i])>split_value){
				aux = selected_elements[last];
				selected_elements[last] = selected_elements[i];
				selected_elements[i] = aux;
				last--; 
			}
			else{
				i++;
			}
		}
		if(data.feature(split_dimension, selected_elements[last])>split_value){
			//last++;
		}
		else{
			last++;
		}
		
		return last;
	}

	virtual std::vector<std::pair<num_t,num_t>> min_max_vector(const rfr::data_containers::base<num_t, response_t, index_t> &data,
		const std::array<index_t,3> &its, const std::vector<index_t> &selected_elements, num_t &sum_E){
		num_t min = NAN, max = NAN;
		sum_E = 0;
		std::vector<std::pair<num_t,num_t>> min_max(data.num_features());
		std::vector<num_t> feature_values;//data.num_data_points());	
		std::vector<index_t> selected_indexes(selected_elements.begin() + its[0], selected_elements.begin() + its[2]);
		for(auto i =0u; i<data.num_features(); i++){
			feature_values = data.features (i, selected_indexes);
			
			min_max_feature(feature_values, min, max, sum_E);
			min_max[i].first = min;
			min_max[i].second = max;
			
		}
		return min_max;
	}

	//for some reason they only consider the positive numbers
	virtual void min_max_feature(const std::vector<num_t> feature_values,
		num_t &min, num_t &max, num_t &sum_E){

		min = std::numeric_limits<num_t>::max();
		max = std::numeric_limits<num_t>::lowest();//std::numeric_limits<float>::min();//max int and min int
		//min = 0; max = 0;
		for(auto j =0u; j<feature_values.size(); j++){
			if(feature_values[j] < min){
				min = feature_values[j];
			}
			if(feature_values[j] > max){
				max = feature_values[j];
			}
		}
		sum_E = sum_E + max - min;
	}

	virtual index_t find_leaf_index(const std::vector<num_t> &feature_vector) const {
		index_t node_index = 0;
		while (! the_nodes[node_index].is_a_leaf()){
			node_index = the_nodes[node_index].falls_into_child(feature_vector);
		}
		return(node_index);
	}

	const node_t&  get_leaf(const std::vector<num_t> &feature_vector) const {
		index_t node_index = find_leaf_index(feature_vector);
		return(the_nodes[node_index]);
	}


    std::pair<num_t, num_t> predict_mean_var (const std::vector<num_t> &feature_vector){
		if(the_nodes.size() == 0){
			throw std::runtime_error("cannot predict on an empty tree");
		}
		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> tmp_node = the_nodes[0], old_node;

		num_t prob_not_separated_now, prob_separated_now, nu;
		num_t prob_not_separated_yet = 1;
		num_t delta_node = 0;
		bool finished = false;
		num_t w, mean = 0, variance = 0, second_moment = 0, pred_second_moment_temp, pred_mean_temp, expected_split_time, variance_from_mean, expected_cut_time;
		num_t sum_W = 0;
		while (!finished){
			old_node = tmp_node;
			delta_node = tmp_node.get_split_time() - get_parent_split_time(tmp_node);
			
			nu = calculate_nu(tmp_node, feature_vector);
			prob_not_separated_now = exp(- delta_node * nu);
			prob_separated_now = 1 - prob_not_separated_now;	
			if(prob_separated_now>0){
				expected_cut_time = 1/ nu;
				w = prob_not_separated_yet * prob_separated_now;
				if(old_node.get_split_time()<life_time ){
					expected_cut_time -= old_node.get_split_cost() * (1-prob_separated_now) / ( 1- exp(- delta_node * nu) );
					//expected_cut_time -= old_node.get_time_split() * (1-prob_separated_now) / ( 1- exp(- delta_node * nu) );
				}
				if(smooth_hierarchically){
					pred_mean_temp = old_node.get_mean();
					pred_second_moment_temp = old_node.get_variance();
					expected_split_time = expected_cut_time + get_parent_split_time(old_node);
					variance_from_mean = variance_coef * (sigmoid(sigmoid_coef * expected_split_time) 
						- sigmoid (sigmoid_coef * get_parent_split_time(old_node)) );
						
					pred_second_moment_temp += variance_from_mean;
				}
				else{
					pred_mean_temp = old_node.get_response_stat().mean();
					pred_second_moment_temp = old_node.get_response_stat().sum_of_squares() / old_node.get_response_stat().sum_of_weights() + noise_variance;
				}

				mean += w * pred_mean_temp;
				second_moment += w * pred_second_moment_temp;
				sum_W += w;		
			}
			//else if (prob_separated_now == 0 && std::isinf(old_node.get_split_cost()) ){
			else if (prob_separated_now == 0 && old_node.get_split_cost() >= life_time ){
				if(smooth_hierarchically){
					pred_mean_temp = old_node.get_mean();
					pred_second_moment_temp = old_node.get_variance();
				}
				else{
					pred_mean_temp = old_node.get_response_stat().mean();
					pred_second_moment_temp = old_node.get_response_stat().sum_of_squares() / old_node.get_response_stat().sum_of_weights() + noise_variance;
				}
				
				mean += prob_not_separated_yet * pred_mean_temp;
				second_moment += prob_not_separated_yet * pred_second_moment_temp;
				sum_W += prob_not_separated_yet;
			}
			if(tmp_node.is_a_leaf()){
				finished = true;
			}
			else{
				prob_not_separated_yet = prob_not_separated_yet * (1-prob_separated_now);
				if(feature_vector[tmp_node.get_split_dimension()]<tmp_node.get_split_value()){
					tmp_node = the_nodes[tmp_node.get_child_index(0)];
				}
				else{
					tmp_node = the_nodes[tmp_node.get_child_index(1)];
				}
			}	
		}
		variance = second_moment - std::pow(mean,2);
		return(std::pair<num_t, num_t> (mean, variance));
    }

	virtual response_t predict(const std::vector<num_t> &feature_vector) const {
		rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> tmp_node = the_nodes[0];

		while( !tmp_node.is_a_leaf()){
			if(feature_vector[tmp_node.get_split_dimension()] <= tmp_node.get_split_value() )
				tmp_node = the_nodes[tmp_node.get_children()[0]];
			else
				tmp_node = the_nodes[tmp_node.get_children()[1]];
		}
		return tmp_node.get_response_stat().mean();
	}
    
	


	virtual num_t calculate_nu(rfr::nodes::k_ary_mondrian_node_full<k, num_t, response_t, index_t, rng_t> &tmp_node,
		const std::vector<num_t> &feature_vector) const{
		auto min_max = tmp_node.get_min_max();
		num_t nu = 0;
		for(auto i = 0u; i< feature_vector.size(); i++){
			nu += std::max(feature_vector[i] - min_max[i].second,(num_t)0) + std::max(min_max[i].first - feature_vector[i],(num_t)0);
		}
		
		return nu;
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

	virtual std::vector<response_t> const &leaf_entries (const std::vector<num_t> &feature_vector) const {
		throw std::runtime_error("doesn't exists for this class");
	}

	/* \brief finds all the split points for each dimension of the input space
	 * 
	 * This function only makes sense for axis aligned splits!
	 * */
	std::vector<std::vector<num_t> > all_split_values (const std::vector<index_t> &types) const {
		std::vector<std::vector<num_t> > split_values(types.size());
		
		for (auto &n: the_nodes){
			if (n.is_a_leaf()) continue;
			
			const auto &s = n.get_split();
			auto fi = s.get_feature_index();

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
	
	virtual index_t number_of_nodes() const {return(the_nodes.size());}
	virtual index_t number_of_leafs() const {return(num_leafs);}
	virtual index_t depth()           const {return(max_depth);}

	
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
		//the_nodes[index].push_response_value(response, weight);
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
		//the_nodes[index].pop_response_value(response, weight);
	}

	num_t max_life_time()const {return life_time;}

	
	void print_info() const {
		
		std::cout<<"number of nodes ="<<number_of_nodes()<<"\n";
		std::cout<<"number of leaves="<<number_of_leafs()<<"\n";
		std::cout<<"      depth     ="<<depth()<<"\n";
		std::cout<<"max_life_time   ="<<max_life_time()<<"\n";
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

