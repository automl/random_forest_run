#ifndef RFR_TEMPORARY_NODE_HPP
#define RFR_TEMPORARY_NODE_HPP

#include <vector>

#include <rfr/splits/split_base.hpp>

namespace rfr{ namespace nodes{

template <typename num_t = float, typename response_t = float, typename index_t = unsigned int>
/* \brief class to hold necessary information during the fitting of a tree
 *
 * When fitting a tree, every split creates two or more children. This class is used
 * to temporarily store all information needed to continue splitting.
 */
struct temporary_node{

    index_t node_index;
	index_t parent_index;
	typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t> >::iterator begin, end; 
	index_t node_level;


	/* \param note_id the unique ID of the node
	 * \param b iterator to the first element of the data_info vector associated with the node
	 * \param e iterator beyond the last element of the data_info vector associated with the node
	 * \param node_lvl the node's level in the tree
	 */
	temporary_node (index_t node_id, index_t parent_id,	index_t node_lvl,
					typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator b,
					typename std::vector<rfr::splits::data_info_t<num_t, response_t, index_t>>::iterator e):
						node_index(node_id), parent_index(parent_id), begin(b), end(e), node_level(node_lvl)
						{}
	

	num_t total_weight (){
        num_t tw = 0.;
        for (auto it = begin; it != end; ++it){
            tw += (*it).weight;
        }
        return(tw);
    }

	void print_info() const{
		std::cout<<"node_index = "<<node_index <<"\n";
		std::cout<<"parent_index = "<<parent_index <<"\n";
		std::cout<<"node_level = "<< node_level<<"\n";
		std::cout<<"data_indices: ";
        for (auto it = begin; it!=end; it++)
            std::cout<<(*it).index<<" ";
        std::cout<<std::endl;
	}
};

}} // end namespace rfr::nodes
#endif
