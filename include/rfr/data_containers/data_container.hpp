#ifndef RFR_DATA_CONTAINER_HPP
#define RFR_DATA_CONTAINER_HPP


#include <vector>
#include <utility>


namespace rfr{ namespace data_containers{
/** \brief The interface for any data container with the minimal functionality
 *
 */
template <typename num_t=float, typename response_t=float, typename index_t=unsigned int>
class base{
  public:

	virtual ~base(){};
  
	/** \brief Function for accessing a single feature value, consistency checks might be omitted for performance
	 *
	 * \param feature_index The index of the feature requested
	 * \param sample_index The index of the data point.
	 *
	 * \return the stored value
	 */
	virtual num_t feature (index_t feature_index, index_t sample_index) const = 0;

	/** \brief member function for accessing the feature values of multiple data points at once, consistency checks might be omitted for performance
	 *
	 * \param feature_index The index of the feature requested
	 * \param sample_indices The indices of the data point.
	 *
	 * \return the stored values
	 */
	virtual std::vector<num_t> features (index_t feature_index, const std::vector<index_t> &sample_indices) const = 0;

	/** \brief member function to query a single response value, consistency checks might be omitted for performance
	 *
	 * \param sample_index the response of which data point
	 * 
	 * \return the response value
	 */
	virtual response_t response (index_t sample_index) const = 0;


	/** \brief function to access the weight attributed to a single data point
	 *
	 * \param sample_index which data point
	 * 
	 * \return the weigth of that sample
	 */
	virtual num_t weight(index_t sample_index) const = 0;


	/** \brief method to add a single data point
	 *
	 * \param features a vector containing the features
	 * \param response the corresponding response value
	 * \param weight the weight of the data point
	 */
	virtual void add_data_point (std::vector<num_t> features, response_t response, num_t weight) = 0;


	/** \brief method to retrieve a data point
	 *
	 * \param index index of the datapoint to extract
	 * 
	 * \return std::vector<num_t> the features of the data point
	 */
	virtual std::vector<num_t> retrieve_data_point (index_t index) const = 0;


	/** \brief query the type of a feature
	 *
	 * \param feature_index the index of the feature
	 * 
	 * \return int type of the feature: 0 - numerical value (float or int); n>0 - categorical value with n different values {0,1,...,n-1}
	 */
	virtual index_t get_type_of_feature (index_t feature_index) const = 0;

	/** \brief query the type of the response
	 *
	 * \return index_t type of the response: 0 - numerical value (float or int); n>0 - categorical value with n different values {0,1,...,n-1}
	 */
	virtual index_t get_type_of_response () const = 0;

	/** \brief specifying the type of a feature
	 *
	 * \param feature_index the index of the feature whose type is specified
	 * \param feature_type the actual type (0 - numerical, value >0 catergorical with values from {0,1,...value-1}
	 */
	virtual void set_type_of_feature (index_t feature_index, index_t feature_type) = 0;

	/** \brief specifying the type of the response
	 *
	 * \param response_type the actual type (0 - numerical, value >0 catergorical with values from {0,1,...value-1}
	 */
	virtual void set_type_of_response (index_t response_type) = 0;


	/** \brief specifies the interval of allowed values for a feature
	 * 
	 * To marginalize out certain feature dimensions using non-i.i.d. data, the numerical bounds
	 * on each variable have to be known. This only applies to numerical features.
	 *
	 * Note: The forest will not check if a datapoint is consistent with the specified bounds!
	 * 
	 * \param feature_index feature_index the index of the feature
	 * \param min the smallest value for the feature
	 * \param max the largest value for the feature
	 */
	virtual void set_bounds_of_feature(index_t feature_index, num_t min, num_t max) = 0;


	/** \brief query the allowed interval for a feature; applies only to continuous variables
	 *
	 * \param feature_index the index of the feature
	 * \return std::pair<num_t,num_t> interval of allowed values
	 */
	virtual std::pair<num_t,num_t> get_bounds_of_feature(index_t feature_index) const = 0;

	/** \brief the number of features of every datapoint in the container */
	virtual index_t num_features() const = 0;
	/** \brief the number of data points in the container */
	virtual index_t num_data_points()  const = 0;
};

}} // namespace rfr::data_containers
#endif // RFR_DATA_CONTAINER_BASE_HPP
