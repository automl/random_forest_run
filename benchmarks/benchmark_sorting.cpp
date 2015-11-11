// g++ -I../include -O3 -pg -o benchmark_sorting -std=c11 benchmark_sorting.cpp 

#include <numeric>
#include <cstring>
#include <random>
#include <algorithm>
#include <functional>
#include <utility>
#include <ctime>


#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"

typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;


typedef rfr::data_containers::mostly_continuous_data<num_type, response_type, index_type> data_type;


template < typename T>
void print_vector( T & v){
	for (auto e: v)
		std::cout<<e<<" ";
	std::cout<<std::endl;
}


num_type rss_split_v1 ( data_type & data, std::vector<index_type> indices, std::vector<index_type> &features_to_try){

	num_type best_loss = std::numeric_limits<num_type>::infinity();
	num_type S_y_total(0), S_y2_total(0);

	for (auto it = indices.begin(); it != indices.end(); it++){
		S_y_total  += data.response(*it);
		S_y2_total += data.response(*it)*data.response(*it);
	}	

	for (auto fi: features_to_try){
		// sort the indices by the value in feature vector
		std::sort(	indices.begin(), indices.end(),
					[&](index_type a, index_type b){return data.feature(fi,a) < data.feature(fi, b);}		//! > uses C++11 lambda function, how exciting :)
		);

		// find the best split by looking at any meaningful value for the feature
		// first some temporary variables
		num_type S_y_left(0), S_y2_left(0);
		num_type N_left(0), N_right(indices.size());
		num_type loss = std::numeric_limits<num_type>::infinity();

		// we start out with everything in the right child
		// so we compute the mean and the variance for that case
		num_type S_y_right(S_y_total), S_y2_right(S_y2_total);

		typename std::vector<index_type>::iterator psii = indices.begin();	// potential split index iterator

		// now we can increase the splitting value to move data points from the right to the left child
		// this way we do not consider a split with everything in the right child
		while (psii != indices.end()){
			num_type psv = data.feature(fi, *psii) + 1e-10; // potential split value add small delta for numerical inaccuracy
			// combine data points that are very close
			while ((psii != indices.end()) && (data.feature(fi,*psii) - psv <= 0)){
				// change the Sum(y) and Sum(y^2) for left and right accordingly
				S_y_left  += data.response(*psii);
				S_y_right -= data.response(*psii);

				S_y2_left += data.response(*psii)*data.response(*psii);
				S_y2_right-= data.response(*psii)*data.response(*psii);
				N_right--;
				N_left++;
				psii++;
			}
			// stop if all data points are now in the left child as this is not a meaningful split
			if (N_right == 0) {break;}

			// compute the loss
			loss = (S_y2_left  - (S_y_left *S_y_left )/N_left) 
			     + (S_y2_right - (S_y_right*S_y_right)/N_right);

			// store the best split
			if (loss < best_loss){
				best_loss = loss;
			}
		}
	}
	return(best_loss);
}





num_type rss_split_v2( data_type & data, std::vector<index_type> indices, std::vector<index_type> &features_to_try){
	num_type best_loss = std::numeric_limits<num_type>::infinity();
	num_type S_y_total(0), S_y2_total(0);

	for (auto it = indices.begin(); it != indices.end(); it++){
		S_y_total  += data.response(*it);
		S_y2_total += data.response(*it)*data.response(*it);
	}

	for (auto fi: features_to_try){

		std::vector< std::pair<num_type, response_type> > sorted_pairs (indices.size());
		for (auto i: indices)
			sorted_pairs.emplace_back(std::pair<num_type, response_type> (data.feature(fi, i), data.response(i)));
		std::sort(sorted_pairs.begin(), sorted_pairs.end());

		// find the best split by looking at any meaningful value for the feature
		// first some temporary variables
		num_type S_y_left(0), S_y2_left(0);
		num_type N_left(0), N_right(indices.size());
		num_type loss = std::numeric_limits<num_type>::infinity();

		// we start out with everything in the right child
		// so we compute the mean and the variance for that case
		num_type S_y_right(S_y_total), S_y2_right(S_y2_total);

		auto spi = sorted_pairs.begin();	// potential split index iterator


		while ( spi != sorted_pairs.end()){
			num_type psv = spi->first;
			while ((spi != sorted_pairs.end()) && (spi->first - psv <= 0)){
				// change the Sum(y) and Sum(y^2) for left and right accordingly
				S_y_left  += spi->second;
				S_y_right -= spi->second;

				S_y2_left += (spi->second)*(spi->second);
				S_y2_right-= (spi->second)*(spi->second);
				N_right--;
				N_left++;
				spi++;
			}

			if (N_right == 0) {break;}

			// compute the loss
			loss = (S_y2_left  - (S_y_left *S_y_left )/N_left) 
			     + (S_y2_right - (S_y_right*S_y_right)/N_right);

			// store the best split
			if (loss < best_loss){
				best_loss = loss;
			}
		}
	}
	return(best_loss);
}




num_type rss_split_v3 ( data_type & data, std::vector<index_type> &indices, std::vector<index_type> &features_to_try){

	num_type best_loss = std::numeric_limits<num_type>::infinity();
	num_type S_y_total(0), S_y2_total(0);


	std::vector<index_type> tmp_indices(indices.size());
	std::iota(tmp_indices.begin(), tmp_indices.end(), 0);
	
	std::vector<num_type> feature; feature.reserve(indices.size());
	std::vector<response_type> responses;responses.reserve(indices.size());

	for (auto it = indices.begin(); it != indices.end(); it++){
		response_type res = data.response(*it);
		S_y_total  += res;
		S_y2_total += res*res;
		responses.push_back(res);
	}	

	for (auto fi: features_to_try){
		feature.clear();
		for (auto it = indices.begin(); it != indices.end(); it++)
			feature.emplace_back(data.feature(fi,*it));
		
		// sort the indices by the value in feature vector
		std::sort(	tmp_indices.begin(), tmp_indices.end(),
					[&](index_type a, index_type b){return feature[a] < feature[b];}		//! > uses C++11 lambda function, how exciting :)
		);

		// find the best split by looking at any meaningful value for the feature
		// first some temporary variables
		num_type S_y_left(0), S_y2_left(0);
		num_type N_left(0), N_right(indices.size());
		num_type loss = std::numeric_limits<num_type>::infinity();

		// we start out with everything in the right child
		// so we compute the mean and the variance for that case
		num_type S_y_right(S_y_total), S_y2_right(S_y2_total);

		auto tmp_i = 0u;

		// now we can increase the splitting value to move data points from the right to the left child
		// this way we do not consider a split with everything in the right child
		while (tmp_i != tmp_indices.size()){
			num_type psv = feature[tmp_indices[tmp_i]]+ 1e-6; // potential split value add small delta for numerical inaccuracy
			// combine data points that are very close
			while ((tmp_i != tmp_indices.size()) && (feature[tmp_indices[tmp_i]] - psv <= 0)){
				// change the Sum(y) and Sum(y^2) for left and right accordingly
				response_type res = responses[tmp_indices[tmp_i]];
				S_y_left  += res;
				S_y_right -= res;

				S_y2_left += res*res;
				S_y2_right-= res*res;
				N_right--;
				N_left++;
				tmp_i++;
			}
			// stop if all data points are now in the left child as this is not a meaningful split
			if (N_right == 0) {break;}

			// compute the loss
			loss = (S_y2_left  - (S_y_left *S_y_left )/N_left) 
			     + (S_y2_right - (S_y_right*S_y_right)/N_right);

			// store the best split
			if (loss < best_loss){
				best_loss = loss;
			}
		}
	}
	return(best_loss);
}



num_type rss_split_v4 ( data_type & data, std::vector<index_type> &indices, std::vector<index_type> &features_to_try){

	num_type best_loss = std::numeric_limits<num_type>::infinity();
	num_type S_y_total(0), S_y2_total(0);


	std::vector<index_type> tmp_indices(indices.size());
	std::iota(tmp_indices.begin(), tmp_indices.end(), 0);
	
	std::vector<num_type> feature; feature.reserve(indices.size());
	std::vector<response_type> responses;responses.reserve(indices.size());

	for (auto it = indices.begin(); it != indices.end(); it++){
		response_type res = data.response(*it);
		S_y_total  += res;
		S_y2_total += res*res;
		responses.push_back(res);
	}	

	for (auto fi: features_to_try){
		feature.clear();
		for (auto it = indices.begin(); it != indices.end(); it++)
			feature.emplace_back(data.feature(fi,*it));
		
		// sort the indices by the value in feature vector
		std::sort(	tmp_indices.begin(), tmp_indices.end(),
					[&](index_type a, index_type b){return feature[a] < feature[b];}		//! > uses C++11 lambda function, how exciting :)
		);

		// find the best split by looking at any meaningful value for the feature
		// first some temporary variables
		num_type S_y_left(0), S_y2_left(0);
		num_type N_left(0), N_right(indices.size());
		num_type loss = std::numeric_limits<num_type>::infinity();

		// we start out with everything in the right child
		// so we compute the mean and the variance for that case
		num_type S_y_right(S_y_total), S_y2_right(S_y2_total);

		// now we can increase the splitting value to move data points from the right to the left child
		// this way we do not consider a split with everything in the right child
		for (auto tmp_i = 0u; tmp_i < tmp_indices.size()-1 ;tmp_i++){
			//num_type psv = feature[tmp_indices[tmp_i]];
			// change the Sum(y) and Sum(y^2) for left and right accordingly
			response_type res = responses[tmp_indices[tmp_i]];
			S_y_left  += res;
			S_y_right -= res;
			S_y2_left += res*res;
			S_y2_right-= res*res;
			N_right--;
			N_left++;

			// compute the loss
			loss = (S_y2_left  - (S_y_left *S_y_left )/N_left) 
			     + (S_y2_right - (S_y_right*S_y_right)/N_right);

			// store the best split
			if (loss < best_loss){
				best_loss = loss;
			}
		}
	}
	return(best_loss);
}





int main (int argc, char** argv){

	index_type num_features = atoi(argv[1]);
	index_type num_data_points = atoi(argv[2]);
	index_type sample_size = atoi(argv[3]);

	data_type data (num_features);

	clock_t t;
	num_type best_loss;

	std::default_random_engine rng;
	std::uniform_real_distribution<response_type> dist1(-1.0,1.0);
	std::uniform_int_distribution<index_type> dist2(0,num_data_points);

	auto random_num = std::bind(dist1, rng);
	auto random_ind = std::bind(dist2, rng);
	
	for (auto i=0u; i < num_data_points; i++){

		num_type feature_vector[num_features];
		std::generate_n(feature_vector, num_features, random_num);
		response_type response = random_num();
		
		data.add_data_point(feature_vector, num_features, response);
	}


	std::vector<index_type> indices (sample_size);
	std::generate_n(indices.begin(), indices.size(), random_ind);
	

	std::vector<index_type> features_to_try(num_features);
	std::iota(features_to_try.begin(), features_to_try.end(), 0);

	t = clock();
	best_loss = rss_split_v1 (data, indices, features_to_try);
	t = clock() - t;
	std::cout << "v1 took "<< t <<" ticks, best_loss = "<<best_loss<<std::endl;
	
	t = clock();
	best_loss = rss_split_v2 (data, indices, features_to_try);
	t = clock() - t;
	std::cout << "v2 took "<< t <<" ticks, best_loss = "<<best_loss<<std::endl;

	t = clock();
	best_loss = rss_split_v3 (data, indices, features_to_try);
	t = clock() - t;
	std::cout << "v3 took "<< t <<" ticks, best_loss = "<<best_loss<<std::endl;

	t = clock();
	best_loss = rss_split_v4 (data, indices, features_to_try);
	t = clock() - t;
	std::cout << "v4 took "<< t <<" ticks, best_loss = "<<best_loss<<std::endl;

    return(0);
}
