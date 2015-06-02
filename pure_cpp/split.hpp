#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <random>


template<class T>
void print_vector(std::vector<T> &v){
	for(auto it = v.begin(); it!=v.end(); it++)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}



template <class f_type>
class continuous_split{
  public:
	int feature_index;
	f_type split_value;
	std::vector<int> indices;
	std::vector<int>::iterator split_index_iterator;

	
	//return the quality of the best
	f_type best_split ( std::vector<f_type> & features, std::vector<f_type> responses, int feature_indx, const std::vector<int>::iterator & indices_start, const std::vector<int>::iterator &indices_end){
		
		feature_index = feature_indx;
		
		// copy the indices so we can manipulate them
		indices = std::vector<int>(indices_start, indices_end);

		// sort the indices by the value in feature vector
		std::sort(	indices.begin(), indices.end(),
					[&](size_t a, size_t b){return features[a] < features[b];}
		);
		
		// find the best split by looking at any meaningful value for the feature 
		// first some temporary variables
		f_type S_y_left(0), S_y2_left(0);
		f_type S_y_right(0), S_y2_right(0);
		f_type N_left(0), N_right(indices.size());
		f_type score(0), best_score(0);
		
		// we start out with everything in the right child
		// so we compute the mean and the variance for that case
		for (auto it = indices.begin(); it != indices.end(); it++){
			S_y_right  += responses[*it];
			S_y2_right += responses[*it]*responses[*it];
		}
		std::vector<int>::iterator psii = indices.begin();	// potential split index iterator

		// Note the score is the negative variance, so that we can maximize it!
		score = - (S_y2_right - (S_y_right*S_y_right)/N_right);
		best_score = score;

		// now we can increase the splitting value incrementaly
		while (psii != indices.end()){
			auto psv = features[*psii]; // potential split value
			// combine data points that are very close
			while ((psii != indices.end()) &&(features[*psii] - psv < 1e-10)){

				// change the <y> and <y^2> for left and right accordingly
				S_y_left  += responses[*psii];
				S_y_right -= responses[*psii];
				
				S_y2_left += responses[*psii]*responses[*psii];
				S_y2_right -= responses[*psii]*responses[*psii];		
				N_right--;
				N_left++;
				psii++;
			}
			// compute the score
			score = -(S_y2_left - (S_y_left*S_y_left)/N_left);
			// the right leaf could be empty!
			if (N_right > 0)
				score  -=  (S_y2_right - (S_y_right*S_y_right)/N_right);
			std::cout<<score<<std::endl;
			
			// store the best split
			if (score > best_score){
				best_score = score;
				split_value = psv;
				split_index_iterator = psii;
			}
		}
		return(best_score);
	}
	
	
	void remove_temp_data(){
		indices.clear();
	}

	int apply ( f_type &value){
		return(value > split_value);
	};
};


template <class f_type = float>
class categorical_split{
  private:
	int feature_index;
	std::set<f_type> values_set;
  public:
};





int main(){
	
	std::default_random_engine gen;
	std::uniform_real_distribution<float> dist(1,6);
	
	int feature_index = 0;
	int N = 100;
	
	
	std::vector<float> features(N);
	std::vector<float> responses(N);
	std::vector<int> indices(N);
	
	for (size_t i = 0; i<indices.size(); i++) indices[i] = i;
	
	
	for (auto it = features.begin(); it != features.end(); it++){
		*it=dist(gen);
	}
	
	for (auto it = responses.begin(); it != responses.end(); it++){
		*it=dist(gen);
	}
	
	print_vector<float>(features);
	
	
	
	
	continuous_split<float> split_test;
	
	std::cout<<split_test.best_split(features, responses, feature_index, indices.begin(), indices.end());
	std::cout<<"\n"<<split_test.split_value;
	
	
	return(0);
}
