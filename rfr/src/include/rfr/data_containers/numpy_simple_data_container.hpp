#ifndef RFR_NUMPY_DATA_CONTAINER_HPP
#define RFR_NUMPY_CONTAINER_BASE_HPP

namespace rfr{
/** \brief The interface for any data container with the minimal functionality
 *
 */
template <typename num_type=float, typename response_type=float, typename index_type=unsigned int>
class numpy_simple_data_container : data_container_base<num_type, response_type, index_type>{
  public:

	virtual num_type feature (index_type feature_index, index_type sample_index){
		
	}
	
	virtual response_type response (index_type sample_index) const{
		
	}
	
	virtual bool add_data_point (num_type* features, index_type num_elements, response_type &response){
		
	}
	
	
	virtual std::vector<num_type> retrieve_data_point (index_type index){
		
	}
	
	
	virtual index_type get_type_of_feature (index_type feature_index){
		
	}

	virtual bool set_type_of_feature (index_type feature_index, index_type feature_type){
		
		
	}

	virtual index_type num_features() const {}
	virtual index_type num_data_points()  const {}
};

} // namespace rfr
#endif // RFR_DATA_CONTAINER_BASE_HPP
