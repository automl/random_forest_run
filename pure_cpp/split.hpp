#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <random>


#include "boost/variant/variant.hpp"
#include "boost/bind.hpp"

#include <limits>


/* TODO:
 * replacing boost::variant by adding both splitting criteria to the private variables and checking which one is not meaningful
 * changing the features data type from std::vector<f_type> to f_type* with information about the dimenions (and maybe strides)
 * same for the response data
 */




static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

template <class f_type>
class split{
  public:
	int feature_index;
	boost::variant<f_type, std::set<int> > split_criterion;
	std::vector<int> indices;
	std::vector<int>::iterator split_index_iterator;

	
	//returns the quality of the best split
	f_type best_split_continuous ( std::vector<f_type> & features, std::vector<f_type> & responses){

		// find the best split by looking at any meaningful value for the feature 
		// first some temporary variables
		f_type S_y_left(0), S_y2_left(0);
		f_type S_y_right(0), S_y2_right(0);
		f_type N_left(0), N_right(indices.size());
		f_type loss(0), best_loss(0);
		
		// we start out with everything in the right child
		// so we compute the mean and the variance for that case
		for (auto it = indices.begin(); it != indices.end(); it++){
			S_y_right  += responses[*it];
			S_y2_right += responses[*it]*responses[*it];
		}
		std::vector<int>::iterator psii = indices.begin();	// potential split index iterator

		loss = (S_y2_right - (S_y_right*S_y_right)/N_right);
		best_loss = loss;

		// now we can increase the splitting value incrementaly
		while (psii != indices.end()){
			auto psv = features[*psii] + 1e-10; // potential split value
			// combine data points that are very close
			while ((psii != indices.end()) &&(features[*psii] - psv <= 0)){

				// change the Sum(y) and Sum(y^2) for left and right accordingly
				S_y_left  += responses[*psii];
				S_y_right -= responses[*psii];
				
				S_y2_left += responses[*psii]*responses[*psii];
				S_y2_right -= responses[*psii]*responses[*psii];		
				N_right--;
				N_left++;
				psii++;
			}
			// compute the loss
			loss = (S_y2_left - (S_y_left*S_y_left)/N_left);
			// the right leaf could be empty!
			if (N_right > 0)
				loss  +=  (S_y2_right - (S_y_right*S_y_right)/N_right);
			std::cout<<N_left<<" "<<loss<<" "<<psv<<std::endl;
			
			// store the best split
			if (loss < best_loss){
				best_loss = loss;
				split_criterion = psv;
				split_index_iterator = psii;
			}
		}
		
		std::cout<<"best split at feature <= "<<boost::get<f_type>(split_criterion)<<"\n";
		return(best_loss);
	}

	f_type best_split_categorical ( std::vector<f_type> & features, std::vector<f_type> & responses, int num_categories){

		// auxiliary variables
		std::vector<int> category_ranking(num_categories);
		std::vector<f_type> N_points_in_category(num_categories,0);
		std::vector<f_type> S_y(num_categories, 0);
		std::vector<f_type> S_y2(num_categories, 0);
		
		for (size_t i = 0; i<responses.size() ; i++){
			// find the category for each entry and make it a proper int
			int cat = (int) std::lround(features[i]);
			std::cout<<"element of category "<< cat<<" with response "<< responses[i] <<"\n";
			// collect all the data to compute the loss
			S_y[cat]  += responses[i];
			S_y2[cat] += responses[i]*responses[i];
			N_points_in_category[cat] += 1;
		}

		// take care b/c certain categories might not be encountered (maybe there was a split on the same variable further up in the tree...)
		// sort the categories by whether there were samples or not
		auto it1 = category_ranking.begin();
		auto it2 = category_ranking.end();	// will point to the first category not represented after we are done here
		for (auto i = 0; i < num_categories; i++){
			if (N_points_in_category[i] == 0) {
				it2--;
				*it2 = i;
			}
			else{
				*it1 = i;
				it1++;
			}
		}

		// sort the categories by their individual mean. only consider the ones with actual specimen here
		std::sort(	category_ranking.begin(), it2,
					[&](size_t a, size_t b){return ( (S_y[a]/N_points_in_category[a]) < (S_y[b]/N_points_in_category[b]) );});		// C++11 lambda function, how exciting :)


		//more auxiliary variables
		f_type S_y_left = 0, S_y2_left = 0, N_left = 0;
		f_type S_y_right = 0, S_y2_right = 0, N_right= 0;
		f_type current_loss = 0, best_loss = 0;
		
		// put one category in the left leaf
		auto it_best_split = category_ranking.begin();
		S_y_left  = S_y[*it_best_split];
		S_y2_left = S_y2[*it_best_split];
		N_left    = N_points_in_category[*it_best_split];
		it_best_split++;

		// the rest goes into the right leaf
		for (it1 = it_best_split; it1!=it2; it1++){
			S_y_right  += S_y[*it1];
			S_y2_right += S_y2[*it1];
			N_right    += N_points_in_category[*it1];
		}

		best_loss = 	 (S_y2_right - (S_y_right*S_y_right)/N_right) 
						+ (S_y2_left - (S_y_left*S_y_left)/N_left);


		// now move one category at a time to the left leaf and recompute the loss
		
		// decrease it2 for now to keep at least one category in the right leaf
		it2--;
		
		for (it1 = it_best_split; it1 != it2; it1++){
			S_y_left  += S_y[*it1];
			S_y_right -= S_y[*it1];
			
			S_y2_left  += S_y2[*it1];
			S_y2_right -= S_y2[*it1];

			N_left  += N_points_in_category[*it1];			
			N_right -= N_points_in_category[*it1];

			current_loss 	= (S_y2_right - (S_y_right*S_y_right)/N_right) 
							+ (S_y2_left - (S_y_left*S_y_left)/N_left);		
			std::cout<<current_loss<<"/"<<best_loss<<std::endl;
			// keep the best split
			if (current_loss < best_loss){
				best_loss = current_loss;
				it_best_split = it1;
				it_best_split++;
			}
		}

		// adjust it2 back to the first unobserved category
		it2++;

		// create the split set for the left leaf
		std::set<int> split_set;
		for (it1 = category_ranking.begin(); it1 != it_best_split; it1++)
			split_set.insert(*it1);

		std::cout<<"Original Split set: ";
		std::vector<int> tmp(split_set.begin(), split_set.end());
		print_vector<int>(tmp);

		// add unobserved values randomly to the split_set
		// TODO: consider using one RNG across everything by passing it along.
		if (it2 != category_ranking.end()){
			std::default_random_engine rng;
			std::bernoulli_distribution dist;

			for (it1 = it2; it1 != category_ranking.end(); it1++){
				if (dist(rng))
					split_set.insert(*it1);
			}
		}
		split_criterion = split_set;

		std::cout<<"Split set: ";
		tmp.assign(split_set.begin(), split_set.end());
		print_vector<int>(tmp);
		return(best_loss);
	}


	f_type best_split ( std::vector<f_type> & features, std::vector<f_type> & responses, int feature_indx, int feature_type, const std::vector<int>::iterator & indices_start, const std::vector<int>::iterator &indices_end){

		// store the feature index right away
		feature_index = feature_indx;
		
		// copy the indices so we can manipulate them
		indices = std::vector<int>(indices_start, indices_end);


		// sort the indices by the value in feature vector
		std::sort(	indices.begin(), indices.end(),
					[&](size_t a, size_t b){return features[a] < features[b];}		// C++11 lambda function, how exciting :)
		);


		// feature_type zero means that it is a continous variable
		if (feature_type == 0)
			return(best_split_continuous(features,responses));
		// a positive feature type encodes the number of possible values
		if (feature_type > 0)
			return(best_split_categorical(features, responses, feature_type));
		
		return (-1*std::numeric_limits<f_type>::infinity());
	}
	
	void remove_temp_data(){
		indices.clear();
	}

	bool apply ( f_type & value){
		//return(boost::apply_visitor( boost::bind(apply_split(), value, _1), split_criterion ));
		
		if ( float* pf = boost::get<float>( &split_criterion ) )
			return( value <=  *pf);
		std::set<int> *ps  = boost::get<std::set<int> > (&split_criterion);
		return((bool) (ps->count ((int) std::lround(value)))  );
	};
};


template <class f_type = float>
class categorical_split{
  private:
	int feature_index;
	std::set<f_type> values_set;
  public:
};
