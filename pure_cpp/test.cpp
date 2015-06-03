#include <iostream>
#include <random>


#include "boost/variant.hpp"



template<class T>
void print_vector(std::vector<T> &v){
	for(auto it = v.begin(); it!=v.end(); it++)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}

#include "split.hpp"


int main(){
	
	std::default_random_engine gen;
	//std::uniform_real_distribution<float> dist(0,10);
	std::uniform_int_distribution<int> dist(0,9);
	
	int feature_index = 0;
	int N = 10;
	
	
	std::vector<float> features(N);
	std::vector<float> responses(N);
	std::vector<int> indices(N);
	
	for (size_t i = 0; i<indices.size(); i++){
		indices[i] = i;
		features[i] = dist(gen);
		responses[i] = ((float) i) / ((float) N);
	}
	
	
	print_vector<float>(features);
	print_vector<float>(responses);
	
	
	
	split<float> split_test;
	
	std::cout<<split_test.best_split(features, responses, feature_index, 10, indices.begin(), indices.end());
	//std::cout<<"\n"<<boost::get<float>(split_test.split_criterion);
	
	
	return(0);
}
