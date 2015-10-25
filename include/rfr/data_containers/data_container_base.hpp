#ifndef RFR_DATA_CONTAINER_BASE_HPP
#define RFR_DATA_CONTAINER_BASE_HPP

#include <vector>


namespace rfr{ namespace data_containers{
/** \brief The interface for any data container with the minimal functionality
 *
 */
template <typename num_type=float, typename response_type=float, typename index_type=unsigned int>
class data_container_base{
  public:
	/** \brief Function for accessing a single feature value, consistency checks might be omitted for performance
	 *
	 * \param feature_index The index of the feature requested
	 * \param sample_index The index of the data point.
	 *
	 * \return the stored value
	 */
	virtual num_type feature (index_type feature_index, index_type sample_index) const = 0;

	/** \brief member function for accessing the feature values of multiple data points at once, consistency checks might be omitted for performance
	 *
	 * \param feature_index The index of the feature requested
	 * \param sample_indices The indices of the data point.
	 *
	 * \return the stored values
	 */
	virtual std::vector<num_type> features (index_type feature_index, std::vector<index_type> &sample_indices) const = 0;

	/** \brief member function to query a single response value, consistency checks might be omitted for performance
	 *
	 * \param sample_index the response of which data point
	 * \return the response value
	 */
	virtual response_type response (index_type sample_index) const = 0;


	/** \brief method to add a single data point
	 *
	 * \param features an array containing all the features
	 * \param num_features length of the array
	 * \param response The corresponding response value
	 * \return bool whether the action was sucessful
	 *
	 */
	virtual bool add_data_point (num_type* features, index_type num_elements, response_type response) = 0;

	/** \brief method to retrieve a data point
	 *
	 * \param index index of the datapoint to extract
	 * \param num_features length of the array
	 * \param response The corresponding response value
	 * \return
	 *
	 */
	virtual std::vector<num_type> retrieve_data_point (index_type index) = 0;

	/** \brief query the type of a feature
	 *
	 * \param feature_index the index of the feature
	 * \return int type of the feature: 0 - numerical value (float or int); n>0 - categorical value with n different values {1,2,...,n}
	 *
	 */
	virtual index_type get_type_of_feature (index_type feature_index) const = 0;

	/** \brief specifying the type of a feature
	 *
	 * \param feature_index the index of the feature whose type is specified
	 * \param feature_type the actual type (0 - numerical, value >0 catergorical with values from {1,2,...value}
	 * \return bool success of the operation (fail do to consistency checks)
	 */
	virtual bool set_type_of_feature (index_type feature_index, index_type feature_type) = 0;

	virtual index_type num_features() const = 0;
	virtual index_type num_data_points()  const = 0;
};

}} // namespace rfr::data_containers
#endif // RFR_DATA_CONTAINER_BASE_HPP
