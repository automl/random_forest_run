#ifndef RFR_UTIL_HPP
#define RFR_UTIL_HPP


#include <iostream>
#include <stdexcept>

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
	typedef long unsigned int luint;
	luint N;
	num_t avg, sdm; // the mean and the squared distance from the mean


  public:
	running_statistics(): N(0), avg(0), sdm(0) {}

	running_statistics( luint n, num_t a, num_t s): N(n), avg(a), sdm(s) {}
	
	void push(num_t x){
		++N;
		num_t delta = x - avg;
		// adjust mean
		avg += delta/N;
		// adjust variance
		sdm += delta*(x-avg);
	}

	void pop (num_t x){
		--N;
		num_t delta = x - avg;
		// adjust mean
		avg -= delta/N;
		// adjust variance
		sdm -= delta*(x-avg);
	}

	num_t	divide_sdm_by(num_t fraction)	const	{ return(N>1 ? std::max<num_t>(0.,sdm/fraction) : NAN);}


	
	luint	number_of_points()		const	{return(N);}
	num_t	mean()					const	{return(N>0 ? avg : NAN);}
	num_t	sum()					const	{return(avg*N);}
	num_t	sum_of_squares()		const	{return(sdm + N*mean());}


	num_t	variance_population()	const	{return(divide_sdm_by(N));}
	num_t	variance_sample()		const	{return(divide_sdm_by(N-1));}
	num_t	variance_MSE()			const	{return(divide_sdm_by(N+1));}

	num_t	std_population() 		const	{return( std::sqrt(variance_population()));}
	num_t	std_sample() 			const	{return( std::sqrt(variance_sample()));}

	/* Source: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation*/
	num_t	std_unbiased_gaussian() {
		num_t n(N);
		num_t c4 = 1 - 1/(4*n) - 7/(32*n*n) - 19/(128*n*n*n);
		return( std_sample()/c4);
	}


	running_statistics operator+ ( const running_statistics &other) const{

		num_t n1(N), n2(other.N), nt(N_total);

		// total number of points is trivial
		luint N_total = N + other.N;
		// the total mean is also pretty easy to figure out
		num_t avg_total = avg * (n1/nt) + other.avg * (n2/nt);
		// the total sdm looks a bit tricky, but is straight forward to derive
		num_t sdm_total = sdm + other.sdm + n1*std::pow(avg-avg_total,2) + n2*std::pow(other.avg-avg_total,2);

		return(	running_statistics(	N_total, avg_total, sdm_total));
	}

	running_statistics operator- ( const running_statistics &other) const{
		
		if (other.N >= N-1)
			throw std::runtime_error("Second statistics must not contain as many points as first one!");

		// new number of points is trivial
		luint N1 = N - other.N;

		num_t n1(N1), n2(other.N), nt(N);
		// the total mean is also pretty easy to figure out
		num_t avg1 = avg * (nt/n1) - other.avg * (n2/n1);
		// the sdm looks a bit tricky, but is straight forward to derive
		num_t sdm1 = sdm - other.sdm - n2*std::pow(other.avg - avg, 2) - n1*std::pow(avg1-avg,2);

		return(running_statistics( N1, avg1, sdm1));
	}


	bool numerically_equal (const running_statistics other, num_t rel_error){
		if (N != other.N) return(false);

		// inline lambda expression for the relative error
		auto relerror = [] (num_t a, num_t b) {return(std::abs(a-b)/(a+b));};

		// all the numerical values are allowed to be slightly off
		if ( relerror(avg, other.avg) > rel_error) return(false);
		if ( relerror(sdm, other.sdm) > rel_error) return(false);

		return(true);
	}

};


template<typename num_t>
class weighted_running_statistics{
  private:
	typedef long unsigned int luint;
	luint N;
	num_t avg, sdm;
	running_statistics<num_t> weight_stat;

  public:
	weighted_running_statistics(): N(0), avg(0), sdm(0), weight_stat() {}
	weighted_running_statistics( luint n, num_t m, num_t s, running_statistics w_stat):
		N(n), avg(m), sdm(v), weight_stat(w_stat) {}
	
	void push (num_t x, weight_t weight){
		if (weight > 0){
			++N;
			// helper
			num_t delta = x - avg;
			// update the weights' sum
			sum_weights += weight;
			// adjust mean
			avg += delta * weight / sum_weights;
			// adjust variance
			sdm += weight*delta*(x-avg);
		}
	}

	void pop (num_t x, weight_t weight){
		if (weight > 0){
			if (weight > weight_stat.sum())
				throw std::runtime_error("Cannot remove item, weight too large");

			if (N >=2)
				throw std::runtime_error("Cannot remove item, statistics doesn't contain more than three elements.");
			--N;
			// helper
			num_t delta = (x - avg);
			// update the weights' sum
			sum_weights -= weight;
			// adjust mean
			avg -= delta * weight / sum_weights;
			// adjust variance
			sdm -= weight*delta*(x-avg);

			if (sdm < 0)
				throw std::runtime_error("Squared Distance from the mean is now negative; Abort!")
		}
	}

	num_t	divide_sdm_by(num_t fraction) const { return(N>1 ? std::max<num_t>(0.,sdm/fraction) : NAN);}

	luint	number_of_points()				const 	{return(N);}
	num_t 	mean() 							const	{return(N>0?avg:NAN);}
	num_t 	variance_population()			const	{return(divide_sdm_by(weight_stat.sum()));}
	num_t	variance_unbiased_frequency()	const	{return(divide_sdm_by(weight_stat.sum()));}
	num_t	variance_unbiased_importance()	const	{return(divide_sdm_by(weight_stat.sum() - (weight_stat.sum_of_squares() / weight_stat.sum())));}




// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
// ==========================================================================================



	weighted_running_statistics operator+ ( const weighted_running_statistics &other) const{

		// total number of points is trivial
		unsigned long int N_total = N + other.number_of_points();

		num_t n1(N), n2(other.number_of_points()), nt(N_total);

		// the total mean is also pretty easy to figure out
		num_t avg_total = avg * (n1/nt) + other.mean() * (n2/nt);


		return(	weighted_running_statistics(
					N_total,
					avg_total,
					// the total variance looks a bit tricky, but is straight forward to derive
					(n1/nt) * var + ((n2-1)/nt)* other.variance() + (n1/nt)*avg + (n2/nt)*other.mean() - avg_total,
					sum_weights + other.sum_of_weights()
				)
		);
	}

	weighted_running_statistics operator- ( const weighted_running_statistics &other) const{
		unsigned long int N2 = other.number_of_points();
		if (N2 >= N-1)
			throw std::runtime_error("Second statistics must not contain as many points as first one!");

		if (other.sum_of_weights() >= sum_weights)
			throw std::runtime_error("Second statistics must not have a greater sum of weights!");

		// new number of points is trivial
		unsigned long int N1 = N - N2;
		// the total mean is also pretty easy to figure out
		num_t avg_total = (N* avg - N2 * other.mean())/ (num_t) N1;

		return	(weighted_running_statistics(
					N1,
					avg_total,
					// the total variance looks a bit tricky, but is straight forward to derive
					((N * var + (N2-1)* other.variance() + N*avg - N2*other.mean())/(num_t) N1) - avg_total,
					sum_weights - other.sum_of_weights()
				)
		);
	}


	bool numerically_equal (weighted_running_statistics other, num_t rel_error){

		if (N != other.number_of_points()) return(false);

		// inline lambda expression for the relative error
		auto relerror = [] (num_t a, num_t b) {return(std::abs(a-b)/(a+b));};

		// all the numerical values are allowed to be slightly off
		if ( relerror(avg, other.mean()) > rel_error) return(false);
		if ( relerror(var(), other.var) > rel_error) return(false);
		if ( relerror(sum_weights, other.sum_of_weights()) > rel_error) return(false);

		return(true);
	}
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
