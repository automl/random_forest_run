#ifndef RFR_UTIL_HPP
#define RFR_UTIL_HPP


#include <iostream>

namespace rfr{ namespace util{

/* Merges f1 and f2 into dest without copying NaNs. This allows for easy marginalization */
template <typename num_t, typename index_type>
inline void merge_two_vectors ( num_t* f1, num_t* f2, num_t* dest, index_type n){
	// make full copy of first vector
	std::copy_n(f1,n, dest);
			for (index_type j=0u; j <n; ++j){
				// copy everything from the second vector that is not NaN
				if (!std::isnan(f2[j]))
					dest[j] = f2[j];
				else if (std::isnan(dest[j]))
					throw std::runtime_error("Merged feature vector still contains a NaN");
			}
}



template<typename num_t>
class running_statistics{
  private:
	long unsigned int N;
	num_t avg, var;
  public:
	running_statistics(): N(0), avg(0), var(0) {}
	
	void push(num_t x){
		++N;
		num_t delta = x - avg;
		// adjust mean
		avg += delta/N;
		// adjust variance
		var += delta*(x-avg);
	}

	void pop (num_t x){
		--N;
		num_t delta = x - avg;
		// adjust mean
		avg -= delta/N;
		// adjust variance
		var -= delta*(x-avg);
	}

	
	long unsigned int number_of_points() {return(N);}
	num_t mean(){ return(N>0 ? avg : NAN);}
	num_t variance(){return(N>1 ? std::max<double>(0.,var/(N-1)) : NAN);}
};


template<typename num_t, typename weight_t>
class weighted_running_statistics{
  private:
	weight_t sum_weights;
	unsigned long int N;
	num_t avg, var;
  public:
	weighted_running_statistics(): sum_weights(0), N(0), avg(0), var(0) {}
	
	void push (num_t x, weight_t weight){
		++N;
		if (weight > 0){
			// helper
			num_t delta = x - avg;
			// update the weights' sum
			sum_weights += weight;
			// adjust mean
			avg += delta * weight / sum_weights;
			// adjust variance
			var += weight*delta*(x-avg);
		}
	}

	void pop (num_t x, weight_t weight){
		--N;
		if (weight > 0){
			// helper
			num_t delta = (x - avg);
			// update the weights' sum
			sum_weights -= weight;
			// adjust mean
			avg -= delta * weight / sum_weights;
			// adjust variance
			var -= weight*delta*(x-avg);
		}
	}
	
	long unsigned int number_of_points() {return(N);}
	num_t mean(){ return(N>0?avg:NAN);}
	num_t variance(){return(N>1?std::max<double>(0.,var/ sum_weights * num_t(N)/num_t(N-1)):NAN);}
};




template <typename num_t>
class running_covariance{
  private:
	long unsigned int N;
	num_t m1, m2;
	num_t cov;

  public:
	running_covariance(): N(0), m1(0), m2(0), cov(0) {}
	
	void push (num_t x1, num_t x2){
		N++;
		num_t delta1 = (x1-m1)/N;
		m1 += delta1;
		num_t delta2 = (x2-m2)/N;
		m2 += delta2;
		
		cov += (N-1) * delta1 * delta2 - cov/N;
	}
	
	long unsigned int number_of_points(){return(N);}
	num_t covariance(){return(num_t(N)/num_t(N-1)*cov);}
};






}}//namespace rfr::util
#endif
