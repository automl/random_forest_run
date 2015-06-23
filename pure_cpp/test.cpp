#include <iostream>
#include <random>


#include<map>

#include "boost/variant.hpp"



template<class T>
void print_vector(std::vector<T> &v){
	for(auto it = v.begin(); it!=v.end(); it++)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}

#include "split.hpp"


void test_categorical_split(){
	
	std::default_random_engine gen;
	//std::uniform_real_distribution<float> dist(0,10);
	std::uniform_int_distribution<int> dist(0,9);
	
	int feature_index = 0;
	int N = 20;
	
	
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
	
	
	for (float bla = 0; bla < 10; bla++)
		std::cout<<"\n"<<split_test.apply(bla);
	
}


void test_continuous_split(){
	
	std::default_random_engine gen;
	std::uniform_real_distribution<float> dist(0,1);
	
	int feature_index = 0;
	int N = 20;
	
	std::vector<float> features(N);
	std::vector<float> responses(N);
	std::vector<int> indices(N);
	
	for (size_t i = 0; i<indices.size(); i++){
		indices[i] = i;
		features[i] = i;
		responses[i] = i + dist(gen)*i/10;
	}
	
	
	print_vector<float>(features);
	print_vector<float>(responses);
	
	
	
	split<float> split_test;
	
	std::cout<<split_test.best_split(features, responses, feature_index, 0, indices.begin(), indices.end());
	//std::cout<<"\n"<<boost::get<float>(split_test.split_criterion);
	
	
	for (float bla = 0; bla < 10; bla++)
		std::cout<<"\n"<<split_test.apply(bla);
	
}

int main(){
	//test_continuous_split();
	std::cout<<sizeof(int)<<"\n";
	std::cout<<sizeof(std::map<int, int>)<<"\n";
	std::cout<<sizeof(std::vector<int> (5));
}
