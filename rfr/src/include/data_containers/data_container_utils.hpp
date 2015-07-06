#ifndef RFR_DATA_CONTAINER_UTIL_HPP
#define RFR_DATA_CONTAINER_UTIL_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "boost/tokenizer.hpp"

namespace rfr{


/** \brief A utility function that reads a csv file containing only numerical values
 *
 *  A very common use case should be reading the 'training data' from a file in csv format. This function does that assuming that each row has the same number of entries. It does NOT read any header information.
 *
 * \param filename the CSV file to be read
 * \return The data in a 2d 'array' ready to be used by the data container classes
 *
 */
template <typename num_type = float>
std::vector< std::vector<num_type> > read_csv_file( const char* filename){

	std::vector< std::vector<num_type> > csv_values;

	std::fstream file(filename, std::ios::in);
	if (file)
	{
		typedef boost::tokenizer< boost::char_separator<char> > Tokenizer;
		boost::char_separator<char> sep(",");
		std::string line;
		while (getline(file, line))
		{
			size_t i=0;
			Tokenizer info(line, sep);   // tokenize the line of data
			std::vector<num_type> values;
			for (Tokenizer::iterator it = info.begin(); it != info.end(); ++it)
			{
				if (i==csv_values.size()){
					csv_values.push_back( std::vector<num_type>(1,strtod(it->c_str(), 0)));
				}
				else{
					// convert data into double value, and store
					(csv_values[i]).push_back(strtod(it->c_str(), 0));
				}
				i++;
			}
		}
	}
	else
		std::cerr << "Error: Unable to open file " << filename << std::endl;
	return(csv_values);
}

template<class T>
void print_vector(std::vector<T> &v){
	for(auto it = v.begin(); it!=v.end(); it++)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}


template<class T>
void print_matrix(std::vector<std::vector<T> > &v){
	for(auto it = v.begin(); it!=v.end(); it++)
		print_vector<T>(*it);
}


} // namespace rfr
#endif // RFR_DATA_CONTAINER_UTIL_HPP
