#ifndef RFR_DATA_CONTAINER_HPP
#define RFR_DATA_CONTAINER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "boost/tokenizer.hpp"

namespace rfr{


// helper function that reads a csv file
template <class num_type = float>
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





template<class num_type = float>
class mostly_contiuous_data{
  private:
	std::vector< std::vector<num_type> > feature_values;
	std::vector<num_type> response_values;
	std::map<int, int> categorical_ranges;

  public:
	
	int read_feature_file (const char* filename){
		feature_values = read_csv_file<num_type>(filename);
		//print_matrix<num_type>(feature_values);
		return(feature_values.size());
	}
	
	int read_response_file (const char* filename){
		response_values = (read_csv_file<num_type>(filename))[0];
		//print_vector<num_type>(response_values);
		return(response_values.size());
	}

	// TODO:	every row must have the same number of entries
	//			number of responses must match number of data points
	bool check_consistency(){
		return(true);
	}

	/* method get the responses of a subset of the samples represented by indices*/
	std::vector<num_type> responses_of(std::vector<int> indices){
		std::vector<num_type> return_vector(indices.size());
		for(auto i = 0; i < indices.size(); i++)
			return_vector[i] = response_values[indices[i]];
		return(return_vector);
	}
	/* same as above only with an iterator range of unknown size*/
	std::vector<num_type> responses_of(const std::vector<int>::iterator start, const std::vector<int>::iterator finish){
		std::vector<num_type> return_vector;
		for(std::vector<int>::iterator it = start; it != finish; it++){
			return_vector.push_back(response_values[(*it)]);
		}
		return(return_vector);
	}



	/* method to extract one feature value for all samples in indices*/
	std::vector<num_type> feature_of(int index, std::vector<int> indices){
		std::vector<num_type> return_vector(indices.size());
		for(auto i = 0; i < indices.size(); i++)
			return_vector[i] = feature_values[index][indices[i]];
		return(return_vector);
	}
	
	/* Same as above, but with iterator range of unknown size*/
	std::vector<num_type> feature_of(int index, std::vector<int>::iterator start, std::vector<int>::iterator finish){
		std::vector<num_type> return_vector;
		for(auto it = start; it != finish; it++)
			return_vector.push_back( feature_values[index][*it]);
		return(return_vector);
	}
	
	std::vector<num_type>* feature (int index){
		return (&feature_values[index]);
	}
	
	std::vector<num_type>* responses (){
		return (&response_values);
	}

	int num_features(){return(feature_values.size());}
	int num_data_points(){return(feature_values[0].size());}

	/* As most features are assumed to be categorical, it is actually
	 * beneficial to store only the categorical exceptions in a hash-map.
	 * Type = 0 means continuous, and Type = n >= 1 means categorical with
	 * options \in {1, n}. For consistency, we exclude zero from the categorical
	 * values if anyone wants to add sparse data later on.
	 */
	int type_of_feature (int index){
		auto it = categorical_ranges.find(index);
		if ( it == categorical_ranges.end())
			return(0);
		return(it->second);
	}

	void declare_categorical(int index, int range){
		// TODO:	add some checks, e.g., 0 <= index < num_features, range > 1 ...
		categorical_ranges[index] = range;
	}
};


}//namespace rfr
#endif
